import pathlib
import argparse
import torch
import h5py
import json

import matplotlib.pyplot as plt
import numpy as np

from artist.util import config_dictionary, paint_loader, set_logger_config
from artist.util.kinematic_optimizer import KinematicOptimizer
from artist.util.scenario import Scenario

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import paint_utils

# Plot Parameters
FIGSIZE = (6,4)
LEGEND_FONTSIZE = 8

def extract_and_filter_calibration_data(
    heliostat_name: str,
    paint_dir: pathlib.Path,
    power_plant_position: torch.Tensor,
    aim_point_identifier: str,
    device,
    max_length: int = 100,
):
    calibration_dir = paint_dir / heliostat_name / "Calibration"
    calibration_properties_paths = list(calibration_dir.glob("*calibration-properties*.json"))

    if len(calibration_properties_paths) > max_length:
        calibration_properties_paths = calibration_properties_paths[-max_length:]
        
    if not calibration_properties_paths:
        raise FileNotFoundError(
            f"No calibration-properties found for {heliostat_name} in {calibration_dir}"
        )

    # Filter: only keep paths where both 'HeliOS' and 'UTIS' are present under 'focal_spot'
    valid_paths = []
    for path in calibration_properties_paths:
        with open(path, "r") as file:
            try:
                calibration_dict = json.load(file)
                focal_data = calibration_dict.get(config_dictionary.paint_focal_spot, {})
                if (
                    config_dictionary.paint_helios in focal_data
                    and config_dictionary.paint_utis in focal_data
                ):
                    valid_paths.append(path)
            except Exception as e:
                print(f"Warning: Skipping {path.name} due to error: {e}")

    if not valid_paths:
        raise ValueError(f"No valid calibration files with both Helios and UTIS for {heliostat_name}")

    # Proceed with loading calibration data
    return paint_loader.extract_paint_calibration_data(
        calibration_properties_paths=valid_paths,
        power_plant_position=power_plant_position,
        aim_point_identifier=aim_point_identifier,
        device=device,
    )
    
def configure_optimizer(parameters, initial_lr=0.01):
    optimizer = torch.optim.Adam(parameters, lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=20,
        threshold=0.1,
        threshold_mode="abs",
    )
    return optimizer, scheduler


def calibrate_single_heliostat(
    scenario: Scenario,
    heliostat_index: int,
    heliostat_name: str,
    paint_dir: pathlib.Path,
    aim_point_identifier: str,
    device,
    max_epoch: int = 150,
    tolerance: float = 1e-7,
):
    power_plant_position = scenario.power_plant_position
    # Extract calibration data
    (
        calibration_target_names,
        center_calibration_images,
        sun_positions,
        motor_positions,
    ) = extract_and_filter_calibration_data(heliostat_name, paint_dir, power_plant_position, aim_point_identifier, device)

    # Normed incident ray direction
    incident_ray_directions = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device) - sun_positions

    # Create calibration scenario for a single heliostat
    calibration_scenario = scenario.create_calibration_scenario(heliostat_index=heliostat_index, device=device)

    # Inject power plant position (needed for calibration extraction earlier)
    calibration_scenario.power_plant_position = scenario.power_plant_position

    # Select parameters to optimize
    optimizable_parameters = [
        calibration_scenario.heliostat_field.all_kinematic_deviation_parameters.requires_grad_(),
        calibration_scenario.heliostat_field.all_actuator_parameters.requires_grad_(),
        calibration_scenario.heliostat_field.all_heliostat_positions.requires_grad_(),
    ]

    # Setup optimizer and scheduler
    optimizer, scheduler = configure_optimizer(optimizable_parameters)

    # Run optimization
    kinematic_optimizer = KinematicOptimizer(
        scenario=calibration_scenario,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    print(f"Optimizing heliostat: {heliostat_name} (Index {heliostat_index}) on aimpoints {aim_point_identifier}")
    
    all_losses_mrad = kinematic_optimizer._optimize_kinematic_parameters_with_motor_positions(
        tolerance=tolerance,
        max_epoch=max_epoch,
        center_calibration_images=center_calibration_images,
        incident_ray_directions=incident_ray_directions,
        all_motor_positions=motor_positions,
        num_log=max_epoch,
        return_final_losses=True,
        device=device,
    )

    return all_losses_mrad

def optimize_kinematics(scenario_path: str, paint_dir: str, results_path: str, device: str = "cuda:0"):

    device = torch.device(device)

    scenario_path = pathlib.Path(scenario_path+'.h5')
    paint_dir = pathlib.Path(paint_dir)

    # Load scenario
    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(scenario_file, device=device)

    heliostat_names = scenario.heliostat_field.all_heliostat_names

    results_dict = {}

    for index, name in enumerate(heliostat_names):
        results_dict[name] = {}

        for aim_point_identifier in ["HeliOS", "UTIS"]:
            losses_mrad = calibrate_single_heliostat(
                scenario=scenario,
                heliostat_index=index,
                heliostat_name=name,
                paint_dir=paint_dir,
                aim_point_identifier=aim_point_identifier,
                device=device,
            )

            results_dict[name][f"mrad_losses_{aim_point_identifier.lower()}"] = (
                losses_mrad.clone().detach().cpu().tolist()
            )

        # Add heliostat position
        position = scenario.heliostat_field.all_heliostat_positions[index]
        results_dict[name]["position"] = position.clone().detach().cpu().tolist()

    torch.save(results_dict,results_path)

def plot_mrad_error_distributions(results_dict, save_path=None):
    """
    Plots histograms + KDEs of clipped mrad losses (0–11 mrad) for HeliOS and UTIS across all heliostats.

    Parameters
    ----------
    results_dict : dict
        Dictionary with structure: {
            heliostat_name: {
                "mrad_losses_helios": [...],
                "mrad_losses_utis": [...],
                ...
            }
        }
    save_path : Path or str, optional
        If given, saves the plot to this path.
    """
    from scipy.stats import gaussian_kde

    # Gather and process all losses
    helios_losses = []
    utis_losses = []

    for data in results_dict.values():
        helios_losses.extend(data["mrad_losses_helios"])
        utis_losses.extend(data["mrad_losses_utis"])
        
    x_vals = np.linspace(0, 10, 100)

    # KDEs
    kde_helios = gaussian_kde(helios_losses, bw_method="scott")
    kde_utis = gaussian_kde(utis_losses, bw_method="scott")

    kde_vals_helios = kde_helios(x_vals)
    kde_vals_utis = kde_utis(x_vals)

    mean_helios = np.mean(helios_losses)
    mode_helios = x_vals[np.argmax(kde_vals_helios)]

    mean_utis = np.mean(utis_losses)
    mode_utis = x_vals[np.argmax(kde_vals_utis)]

    print("Min HeliOS loss:", np.min(helios_losses))
    print("Min UTIS loss:", np.min(utis_losses))

    # Plot
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.hist(helios_losses, bins=25, range=(0, 10), density=True, alpha=0.3, label="HeliOS Histogram", color="tab:blue")
    ax.plot(x_vals, kde_vals_helios, label="HeliOS KDE", color="tab:blue")
    ax.axvline(mode_helios, color="tab:blue", linestyle="--", label=f"HeliOS Mode: {mode_helios:.2f} mrad")
    ax.axvline(mean_helios, color="tab:blue", linestyle=":", label=f"HeliOS Mean: {mean_helios:.2f} mrad")

    ax.hist(utis_losses, bins=25, range=(0, 10), density=True, alpha=0.3, label="UTIS Histogram", color="tab:orange")
    ax.plot(x_vals, kde_vals_utis, label="UTIS KDE", color="tab:orange")
    ax.axvline(mode_utis, color="tab:orange", linestyle="--", label=f"UTIS Mode: {mode_utis:.2f} mrad")
    ax.axvline(mean_utis, color="tab:orange", linestyle=":", label=f"UTIS Mean: {mean_utis:.2f} mrad")

    ax.set_xlabel("Pointing Error / mrad")
    ax.set_ylabel("Density / -")
    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.grid(True)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved error distribution plot to {save_path}")

    plt.close(fig)
    return fig

# def plot_mrad_vs_distance(results_dict, save_path=None):
#     """
#     Plots mean pointing error (mrad) vs. heliostat distance in XY plane from origin (0, 0).

#     Parameters
#     ----------
#     results_dict : dict
#         Dictionary with structure:
#             {
#                 heliostat_name: {
#                     "mrad_losses_helios": [...],
#                     "mrad_losses_utis": [...],
#                     "position": [x, y, z]
#                 },
#                 ...
#             }
#     save_path : str or Path, optional
#         If given, saves the plot to this path.
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np

#     distances = []
#     helios_means = []
#     utis_means = []

#     for data in results_dict.values():
#         pos = np.array(data["position"])  # [x, y, z]
#         distance = np.linalg.norm(pos[:2])  # distance in XY plane

#         helios_losses = np.array(data["mrad_losses_helios"])
#         utis_losses = np.array(data["mrad_losses_utis"])

#         # Clean + mean
#         helios_mean = np.nanmean(helios_losses)
#         utis_mean = np.nanmean(utis_losses)

#         distances.append(distance)
#         helios_means.append(helios_mean)
#         utis_means.append(utis_mean)

#     # Plot
#     fig, ax = plt.subplots(figsize=FIGSIZE)
#     ax.scatter(distances, helios_means, color="tab:blue", label="HeliOS Mean Error", alpha=0.7)
#     ax.scatter(distances, utis_means, color="tab:orange", label="UTIS Mean Error", alpha=0.7)

#     ax.set_xlabel("Heliostat Distance to Tower / m")
#     ax.set_ylabel("Mean Pointing Error / m")
#     ax.grid(True)
#     ax.legend(fontsize=LEGEND_FONTSIZE)

#     if save_path:
#         fig.savefig(save_path, dpi=300, bbox_inches="tight")
#         print(f"Saved distance vs. error plot to {save_path}")

#     plt.close(fig)
#     return fig

def plot_mrad_vs_distance(results_dict, save_path=None):
    """
    Plots mean pointing error (mrad) vs. heliostat distance in XY plane from origin (0, 0),
    with unique markers per heliostat and trendlines.

    Parameters
    ----------
    results_dict : dict
        Dictionary with structure:
            {
                heliostat_name: {
                    "mrad_losses_helios": [...],
                    "mrad_losses_utis": [...],
                    "position": [x, y, z]
                },
                ...
            }
    save_path : str or Path, optional
        If given, saves the plot to this path.
    """
    # Marker cycle (repeatable)
    marker_styles = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'H', '>', '<']
    
    distances = []
    helios_means = []
    utis_means = []

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (name, data) in enumerate(results_dict.items()):
        pos = np.array(data["position"])
        distance = np.linalg.norm(pos[:2])

        helios_mean = np.nanmean(data["mrad_losses_helios"])
        utis_mean = np.nanmean(data["mrad_losses_utis"])

        distances.append(distance)
        helios_means.append(helios_mean)
        utis_means.append(utis_mean)

        marker = marker_styles[idx % len(marker_styles)]
        ax.scatter(distance, helios_mean, color="tab:blue", marker=marker, label=f"{name} HeliOS", alpha=0.7)
        ax.scatter(distance, utis_mean, color="tab:orange", marker=marker, label=f"{name} UTIS", alpha=0.7)

    # Convert to arrays for fitting
    distances = np.array(distances)
    helios_means = np.array(helios_means)
    utis_means = np.array(utis_means)

    # Trendlines
    helios_fit = np.poly1d(np.polyfit(distances, helios_means, 1))
    utis_fit = np.poly1d(np.polyfit(distances, utis_means, 1))
    x_vals = np.linspace(distances.min(), distances.max(), 200)

    ax.plot(x_vals, helios_fit(x_vals), color="tab:blue", linestyle="--", label="HeliOS Trend")
    ax.plot(x_vals, utis_fit(x_vals), color="tab:orange", linestyle="--", label="UTIS Trend")

    ax.set_xlabel("Heliostat Distance to Tower / m")
    ax.set_ylabel("Mean Pointing Error / mrad")
    ax.set_title("Pointing Error vs. Distance to Tower")
    ax.grid(True)
    ax.legend(fontsize=8, loc="best", ncol=2)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved distance vs. error plot to {save_path}")

    plt.close(fig)
    return fig

 

def plot_results(results_path, save_path=None):
            
    # Load results (losses, etc.)
    results_dict = torch.load(results_path)
    save_path = pathlib.Path(save_path)
    
    plot_mrad_error_distributions(results_dict,  save_path / "01a_angular_error_distribution.png")
    plot_mrad_vs_distance(results_dict,  save_path / "01b_error_vs_distance.png")
                                  
def main():
    parser = argparse.ArgumentParser(
        description="Generate and calibrate a PAINT scenario."
    )
    parser.add_argument(
        "--paint_dir",
        type=str,
        default="/workVERLEIHNIX/share/PAINT",
        help="Base directory for PAINT data",
    )
    parser.add_argument(
        "--scenario_path",
        type=str,
        default="examples/data/test_scenario_paint",
        help="Path to save the generated scenario",
    )
    parser.add_argument(
        "--tower_file",
        type=str,
        default="examples/data/tower-measurements.json",
        help="Path to tower-measurements.json",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for tensor operations",
    )
    parser.add_argument(
        "--result_file",
        type=str,
        default="examples/data/results_mrad_losses.pt",
        help="Path to result file",
    )
    parser.add_argument(
        "--plot_save_path",
        type=str,
        default="examples/plots",
        help="Path to save plots",
    )
    args = parser.parse_args()
    
    with open("examples/data/heliostat_list.json", "r") as f:
        heliostat_names = json.load(f)
    # Set logger
    set_logger_config()
      
    # Set seed
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    
    # Step 1: Generate scenario only if it doesn't exist
    if not pathlib.Path(args.scenario_path).exists():
        print(f"Scenario file not found at {args.scenario_path}. Generating scenario...")
        paint_utils.generate_paint_scenario(
            paint_dir=args.paint_dir,
            scenario_path=args.scenario_path,
            tower_file=args.tower_file,
            heliostat_names=heliostat_names,
            device=args.device
        )
    else:
        print(f"Scenario already exists at {args.scenario_path}. Skipping generation.")

    # Step 2: Optimize kinematics only if results file doesn't exist
    if not pathlib.Path(args.result_file).exists():
        print(f"Result file not found at {args.result_file}. Running kinematic optimization...")
        optimize_kinematics(
            scenario_path=args.scenario_path,
            paint_dir=args.paint_dir,
            results_path=args.result_file,
            device=args.device,
        )
    else:
        print(f"Results already exist at {args.result_file}. Skipping optimization.")

    # Step 3: Plot results (always run)
    print("Plotting results...")
    plot_results(
        results_path=args.result_file,
        save_path=args.plot_save_path
    )

if __name__ == "__main__":
    main()