import pathlib
import argparse
import torch
import h5py
import json

import matplotlib.pyplot as plt
import numpy as np

from artist.data_loader import paint_loader
from artist.util import config_dictionary, set_logger_config
from artist.core.kinematic_optimizer import KinematicOptimizer
from artist.scenario.scenario import Scenario

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import paint_utils

# Plot Parameters
FIGSIZE = (6,4)
LEGEND_FONTSIZE = 8

helmholtz_colors = {
    "hgfblue": "#005AA0",
    "hgfdarkblue": "#0A2D6E",
    "hgfgreen": "#8CB423",
    "hgfgray": "#5A696E",
    "hgfaerospace": "#50C8AA",
    "hgfearthandenvironment": "#326469",
    "hgfenergy": "#FFD228",
    "hgfhealth": "#D23264",
    "hgfkeytechnologies": "#A0235A",
    "hgfmatter": "#F0781E",
}


def extract_and_filter_calibration_data(
    scenario,
    heliostat_name: str,
    paint_dir: pathlib.Path,
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

    heliostat_data_mapping = [(heliostat_name, valid_paths)]
    # Proceed with loading calibration data
    return paint_loader.extract_paint_calibration_properties_data(
                heliostat_calibration_mapping=[
                    (heliostat_name, calibration_properties_paths)
                    for heliostat_name, calibration_properties_paths in heliostat_data_mapping
                    if heliostat_name in scenario.heliostat_field.heliostat_groups[0].names
                ],
                heliostat_names=scenario.heliostat_field.heliostat_groups[0].names,
                target_area_names=scenario.target_areas.names,
                power_plant_position=scenario.power_plant_position,
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


def optimize_kinematics(scenario_path: str, paint_dir: str, results_path: str, device: str = "cuda:0"):

    device = torch.device(device)

    scenario_path = pathlib.Path(scenario_path)
    paint_dir = pathlib.Path(paint_dir)

    # Load scenario
    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(scenario_file, device=device)

    heliostat_names = scenario.heliostat_field.heliostat_groups[0].names

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
    
    # Plot
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.hist(helios_losses, bins=25, range=(0, 10), density=True, alpha=0.3, label="HeliOS Histogram", color=helmholtz_colors["hgfblue"])
    ax.plot(x_vals, kde_vals_helios, label="HeliOS KDE", color=helmholtz_colors["hgfblue"])
    ax.axvline(mode_helios, color=helmholtz_colors["hgfblue"], linestyle="--", label=f"HeliOS Mode: {mode_helios:.2f} mrad")
    ax.axvline(mean_helios, color=helmholtz_colors["hgfblue"], linestyle=":", label=f"HeliOS Mean: {mean_helios:.2f} mrad")

    ax.hist(utis_losses, bins=25, range=(0, 10), density=True, alpha=0.3, label="UTIS Histogram", color=helmholtz_colors["hgfenergy"])
    ax.plot(x_vals, kde_vals_utis, label="UTIS KDE", color=helmholtz_colors["hgfenergy"])
    ax.axvline(mode_utis, color=helmholtz_colors["hgfenergy"], linestyle="--", label=f"UTIS Mode: {mode_utis:.2f} mrad")
    ax.axvline(mean_utis, color=helmholtz_colors["hgfenergy"], linestyle=":", label=f"UTIS Mean: {mean_utis:.2f} mrad")

    ax.set_xlabel("Pointing Error / mrad")
    ax.set_ylabel("Density / -")
    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.grid(True)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved error distribution plot to {save_path}")

    plt.close(fig)
    return fig

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

    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    for idx, (name, data) in enumerate(results_dict.items()):
        pos = np.array(data["position"])
        distance = np.linalg.norm(pos[:2])

        helios_mean = np.nanmean(data["mrad_losses_helios"])
        utis_mean = np.nanmean(data["mrad_losses_utis"])

        distances.append(distance)
        helios_means.append(helios_mean)
        utis_means.append(utis_mean)

        marker = marker_styles[idx % len(marker_styles)]

        ax.scatter(
            distance,
            helios_mean,
            color=helmholtz_colors["hgfblue"],
            marker=marker,
            label="HeliOS Mean Error per Heliostat" if idx == 0 else None,
            alpha=0.7,
        )
        ax.scatter(
            distance,
            utis_mean,
            color=helmholtz_colors["hgfenergy"],
            marker=marker,
            label="UTIS Mean Error per Heliostat" if idx == 0 else None,
            alpha=0.7,
        )

    # Convert to arrays for fitting
    distances = np.array(distances)
    helios_means = np.array(helios_means)
    utis_means = np.array(utis_means)

    # Trendlines
    helios_fit = np.poly1d(np.polyfit(distances, helios_means, 1))
    utis_fit = np.poly1d(np.polyfit(distances, utis_means, 1))
    x_vals = np.linspace(distances.min(), distances.max(), 200)

    ax.plot(x_vals, helios_fit(x_vals), color=helmholtz_colors["hgfblue"], linestyle="--", label="HeliOS Trend")
    ax.plot(x_vals, utis_fit(x_vals), color=helmholtz_colors["hgfenergy"], linestyle="--", label="UTIS Trend")

    ax.set_xlabel("Heliostat Distance to Tower / m")
    ax.set_ylabel("Mean Pointing Error / mrad")
    ax.grid(True)
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.set_ylim([0,16])
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved distance vs. error plot to {save_path}")

    plt.close(fig)
    return fig

 

def plot_results(results_path, save_path=None):
            
    # Load results (losses, etc.)
    results_dict = torch.load(results_path)
    save_path = pathlib.Path(save_path)
    
    plot_mrad_error_distributions(results_dict,  save_path / "01a_angular_error_distribution.pdf")
    plot_mrad_vs_distance(results_dict,  save_path / "01b_error_vs_distance.pdf")
                                  
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
        default="examples/data/paint_scenario_50heliostats",
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

    device = torch.device(args.device)
    scenario_path = pathlib.Path(args.scenario_path+'.h5')
    paint_dir = pathlib.Path(args.paint_dir)
    result_file = pathlib.Path(args.result_file)

    
    plt.rcParams["font.family"] = "sans-serif"
    
    # Step 1: Generate scenario only if it doesn't exist
    scenario_file = args.scenario_path + '.h5'
    if not pathlib.Path(scenario_file).exists():
        print(f"Scenario file not found at {args.scenario_path}. Generating scenario...")
        paint_utils.generate_paint_scenario(
            paint_dir=paint_dir,
            scenario_path=scenario_path,
            tower_file=args.tower_file,
            heliostat_names=heliostat_names,
            use_deflectometry=False,
            device=device
        )
    else:
        print(f"Scenario already exists at {args.scenario_path}. Skipping generation.")

    # Step 2: Optimize kinematics only if results file doesn't exist
    if not pathlib.Path(args.result_file).exists():
        print(f"Result file not found at {args.result_file}. Running kinematic optimization...")
        optimize_kinematics(
            scenario_path=scenario_path,
            paint_dir=paint_dir,
            results_path=result_file,
            device=device,
        )
    else:
        print(f"Results already exist at {args.result_file}. Skipping optimization.")

    # Step 3: Plot results (always run)
    print("Plotting results...")
    plot_results(
        results_path=result_file,
        save_path=args.plot_save_path
    )

if __name__ == "__main__":
    main()