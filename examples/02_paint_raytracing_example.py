
import pathlib
import argparse
import logging

import h5py
import matplotlib.pyplot as plt
import torch
from matplotlib.pyplot import tight_layout

from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util import set_logger_config
from artist.util.scenario import Scenario
from artist.util import paint_loader
import numpy as np


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import paint_utils

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

MEASUREMENT_IDS = {
    "AA39": 149576,
   "AY26": 247613,
   "BC34": 82084
}
    
def generate_scenarios(args):
    """
    Generates one scenario file per heliostat in args.heliostats,
    skipping those that already exist.
    """
    for name in args.heliostats:
        scenario_path = pathlib.Path(args.scenario_path) / f"paint_scenario_{name}.h5"

        if scenario_path.exists():
            print(f"Scenario for {name} already exists at {scenario_path}. Skipping.")
            continue

        print(f"Generating scenario for {name} at {scenario_path}...")
        paint_utils.generate_paint_scenario(
            paint_dir=args.paint_dir,
            scenario_path=scenario_path,
            tower_file=args.tower_file,
            heliostat_names=[name],
            device=args.device
        )
        
def generate_flux_images(args):
    
    results_dict = {}

    for i,name in enumerate(args.heliostats):
        
        results_dict[name] = {}
        
        scenario_path = f"examples/data/paint_scenario_{name}.h5"
        measurements_id = MEASUREMENT_IDS[name]
        
        # Load scenario
        with h5py.File(scenario_path, "r") as scenario_file:
            scenario = Scenario.load_scenario_from_hdf5(scenario_file, device=args.device)
            
        # Generate list
        calibration_properties_paths = [
            pathlib.Path(f"{args.paint_dir}/{name}/Calibration/{measurements_id}-calibration-properties.json")
        ]

        # Load the calibration data.
        (
            calibration_target_name,
            center_calibration_image,
            sun_position,
            _,
        ) = paint_loader.extract_paint_calibration_data(
            calibration_properties_paths=calibration_properties_paths,
            power_plant_position=scenario.power_plant_position,
            device=args.device,
        )

        scenario.heliostat_field.all_aim_points = center_calibration_image

        # The incident ray direction needs to be normed.
        incident_ray_direction = (
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=args.device) - sun_position
        )
            
        # Align all heliostats.
        scenario.heliostat_field.align_surfaces_with_incident_ray_direction(
            incident_ray_direction=incident_ray_direction, 
            mode="all",
            device=args.device
        )

        # Create a ray tracer.
        ray_tracer = HeliostatRayTracer(scenario=scenario, batch_size=1)

        # Perform heliostat-based ray tracing.
        flux_deflectometry = ray_tracer.trace_rays(
            incident_ray_direction=incident_ray_direction,
            target_area=scenario.get_target_area(calibration_target_name[0]),
            device=args.device,
        )
        
        # Perform ideal heliostat Caluclation
        flux_ideal = ray_tracer.trace_rays(
           incident_ray_direction=incident_ray_direction,
           target_area=scenario.get_target_area(calibration_target_name[0]),
           surface_mode= "ideal",
           device=args.device,
        )
        
        # Load raw and UTIS image
        utis_image = paint_utils.load_image_as_tensor(name, args.paint_dir, measurements_id, "flux") 
        raw_image = paint_utils.load_image_as_tensor(name, args.paint_dir, measurements_id, "cropped")
        
        results_dict[name]["image_raw"] = raw_image.cpu().detach().numpy()
        results_dict[name]["utis_image"] = utis_image.cpu().detach().numpy()
        results_dict[name]["flux_ideal"] = flux_ideal.cpu().detach().numpy()
        results_dict[name]["flux_deflectometry"] = flux_deflectometry.cpu().detach().numpy()
        results_dict[name]["surface"] = scenario.heliostat_field.all_surface_points[0,:,:3].reshape(100,100,3).cpu().detach().numpy()
        
    torch.save(results_dict, args.result_file)

def plot_results_flux_comparision(args):
    """
    Plots a multi-row comparison of flux and calibration images from the results_dict.

    Parameters
    ----------
    results_dict : dict
        Dictionary where each key is a heliostat name and each value is a dict containing:
            - image_raw
            - utis_image
            - flux_ideal
            - flux_deflectometry
    save_path : str
        File path to save the final figure.
    """
    results_dict = torch.load(args.result_file, weights_only=False)
    
    number_of_heliostats = len(results_dict)
    fig, ax = plt.subplots(number_of_heliostats, 5, figsize=(12.5, 2.5 * number_of_heliostats), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.2]}  )

    if number_of_heliostats == 1:
        ax = [ax]  # Handle single-row case

    colormaps = ["gray", "hot", "hot", "hot", "jet"]  
    
    for i, (name, data) in enumerate(results_dict.items()):
        images = [
            data["image_raw"],
            data["utis_image"],
            data["flux_ideal"],
            data["flux_deflectometry"],
            data["surface"]
        ]

        for j, img in enumerate(images):
            ax[i][j].imshow(img, cmap=colormaps[j])
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            ax[i][j].set_xticklabels([])
            ax[i][j].set_yticklabels([])
            
        ax[i][0].set_ylabel(name, rotation=90, labelpad=10, va='center')
        ax[i][4].set_aspect(1/1.2)
        
    # Set column titles
    ax[0][0].set_title("Raw Image")
    ax[0][1].set_title("Flux - UTIS")
    ax[0][2].set_title("Flux - Ideal")
    ax[0][3].set_title("Flux - Deflectometry")
    ax[0][4].set_title("Surface - Deflectometry")
    
    fig.tight_layout()
    save_path = args.plot_save_path + '/02a_flux_comparision.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved flux comparison plot to {save_path}")
    plt.close(fig)


def plot_error_distributions(args):
    """
    Loads results_dict and plots surface z-coordinate distributions for each heliostat.

    Parameters
    ----------
    args : argparse.Namespace
        Should contain `results_path` (path to .pt file) and `plot_save_path` (optional).
    """
    from scipy.stats import gaussian_kde
    
    # Load the results dict from .pt file
    results_dict = torch.load(args.result_file, weights_only=False)

    fig, axs = plt.subplots(2, 1, figsize=(4, 7.5), sharex=False)

    # --- Upper subplot: surface Z KDEs ---
    ax_surface = axs[0]
    z_vals_all = []

    ax_surface.axvline(x=0.0, color="black", linestyle="--", label="Ideal")
    
    colors = [helmholtz_colors['hgfblue'],helmholtz_colors['hgfgreen'],helmholtz_colors['hgfaerospace']]
    
    for idx, (name, data) in enumerate(results_dict.items()):
        surface_points = data["surface"]  # shape [N, 3]
        z_vals = surface_points[:, 2].flatten()

        # KDE
        kde = gaussian_kde(z_vals)
        x_range = np.linspace(z_vals.min(), z_vals.max(), 200)
        kde_vals = kde(x_range)

        # Plot
        ax_surface.plot(x_range, kde_vals, label=name, color=colors[idx])
        z_vals_all.append(z_vals)

    # ax_surface.set_title("Surface Z-Coordinate Distribution per Heliostat")
    ax_surface.set_xlabel("Surface Coordinate in Z / m")
    ax_surface.set_ylabel("Density / -")
    ax_surface.legend()
    ax_surface.grid(True)

    # --- Lower subplot: placeholder for additional data (optional) ---
    # --- Plot 2: Flux Deviation Error ---
    ax_flux = axs[1]
    heliostat_names = list(results_dict.keys())
    x = np.arange(len(heliostat_names))

    flux_error_ideal = []
    flux_error_deflectometry = []

    for name in heliostat_names:
        utis = results_dict[name]["utis_image"]
        flux_ideal = results_dict[name]["flux_ideal"]
        flux_deflectometry = results_dict[name]["flux_deflectometry"]

        error_ideal = paint_utils.calculate_flux_deviation(utis, flux_ideal)
        error_deflec = paint_utils.calculate_flux_deviation(utis, flux_deflectometry)

        flux_error_ideal.append(error_ideal)
        flux_error_deflectometry.append(error_deflec)

    ax_flux.scatter(x, flux_error_ideal, label="Ideal", color=helmholtz_colors['hgfgray'], marker="o")
    ax_flux.scatter(x, flux_error_deflectometry, label="Deflectometry", color=helmholtz_colors['hgfdarkblue'], marker="x")

    ax_flux.set_xticks(x)
    ax_flux.set_xticklabels(heliostat_names, rotation=45, ha="right")
    ax_flux.set_ylabel("Relative Flux Error / -")
    ax_flux.legend()
    ax_flux.grid(True)

    # --- Save or show ---
    if hasattr(args, "plot_save_path") and args.plot_save_path:
        save_path = args.plot_save_path + '/02b_error_distributions.pdf'
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close(fig)
    
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
        "--tower_file",
        type=str,
        default="examples/data/tower-measurements.json",
        help="Path to tower-measurements.json",
    )
    parser.add_argument(
        "--heliostats",
        type=str,
        nargs="+",
        default=["AA39", ],#"AY26", "BC34"
        help="List of heliostat names",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:3",
        help="Device for tensor operations",
    )
    parser.add_argument(
        "--scenario_path",
        type=str,
        default="examples/data",
        help="Path to scenario folder",
    )
    parser.add_argument(
        "--result_file",
        type=str,
        default="examples/data/raytracing_results.pt",
        help="Path to result file",
    )
    parser.add_argument(
        "--plot_save_path",
        type=str,
        default="examples/plots",
        help="Path to save plots",
    )
    args = parser.parse_args()
    
    # Set logger
    set_logger_config()
    
    # Set seed
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    
    
    # Step 1: Generate scenarios
    generate_scenarios(args)

    generate_flux_images(args)
    # Step 2: Generate flux images only if results file doesn't exist yet
    # if not pathlib.Path(args.result_file).exists():
    #     print(f"Flux results not found at {args.result_file}. Generating flux images...")
    #     generate_flux_images(args)
    # else:
    #     print(f"Flux results already exist at {args.result_file}. Skipping generation.")

    # Step 3: Plot image and flux map comparison
    plot_results_flux_comparision(args)

    # Step 4: Plot error distributions (surface Z and flux deviation)
    plot_error_distributions(args)
    
    
    

if __name__ == "__main__":      
    main()