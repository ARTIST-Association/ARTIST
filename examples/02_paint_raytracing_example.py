import pathlib
import sys
from pathlib import Path
from typing import Union

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.data_loader import paint_loader
from artist.scenario.scenario import Scenario
from artist.scenario.surface_generator import SurfaceGenerator
from artist.util import set_logger_config

sys.path.append(str(Path(__file__).resolve().parent))
import json
import os

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


MEASUREMENT_IDS = {"AA39": 149576, "AY26": 247613, "BC34": 82084}


def align_and_trace_rays(
    scenario: Scenario,
    aimpoints: torch.Tensor,
    light_direction: torch.Tensor,
    active_heliostats_mask: torch.Tensor,
    target_area_mask: torch.Tensor,
    device: Union[torch.device, str] = "cuda",
) -> torch.Tensor:
    """
    Align the heliostat and perform heliostat ray tracing.

    Parameters
    ----------
    light_direction : torch.Tensor
        The direction of the incoming light on the heliostat.
    active_heliostats_mask : torch.Tensor
        A mask for the active heliostats.
    target_area_mask : torch.Tensor
        The indices of the target areas for each active heliostat.
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Returns
    -------
    torch.Tensor
        A tensor containing the distribution strengths used to generate the image on the receiver.
    """
    # Activate heliostats
    scenario.heliostat_field.heliostat_groups[0].activate_heliostats(
        active_heliostats_mask=active_heliostats_mask
    )

    # Align all heliostats.
    scenario.heliostat_field.heliostat_groups[
        0
    ].align_surfaces_with_incident_ray_directions(
        aim_points=aimpoints,
        incident_ray_directions=light_direction,
        active_heliostats_mask=active_heliostats_mask,
        device=device,
    )

    # Create a ray tracer.
    ray_tracer = HeliostatRayTracer(
        scenario=scenario,
        heliostat_group=scenario.heliostat_field.heliostat_groups[0],
    )

    # Perform heliostat-based ray tracing.
    return ray_tracer.trace_rays(
        incident_ray_directions=light_direction,
        active_heliostats_mask=active_heliostats_mask,
        target_area_mask=target_area_mask,
        device=device,
    )


def generate_flux_images(scenario_path, heliostats, paint_dir, result_file, device):
    """Generate flux images via alignment and ray tracing and save to result_file."""
    results_dict = {}

    scenario_path = pathlib.Path(scenario_path).with_suffix(".h5")

    # Load scenario
    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(scenario_file, device=device)

    scenario.light_sources.light_source_list[0].number_of_rays = 6000
    # Generate list
    heliostat_data_mapping = []
    for name in heliostats:
        heliostat_data_mapping.append(
            (
                name,
                [
                    pathlib.Path(
                        f"{paint_dir}/{name}/Calibration/{MEASUREMENT_IDS[name]}-calibration-properties.json"
                    )
                ],
            )
        )
    # Load the calibration data.
    (
        focal_spots_calibration,
        incident_ray_directions_calibration,
        motor_positions_calibration,
        heliostats_mask_calibration,
        target_area_mask_calibration,
    ) = paint_loader.extract_paint_calibration_properties_data(
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

    # Perform heliostat-based ray tracing.
    # Perform alignment and ray tracing to generate flux density images.
    flux = align_and_trace_rays(
        scenario=scenario,
        aimpoints=focal_spots_calibration,
        light_direction=incident_ray_directions_calibration,
        active_heliostats_mask=heliostats_mask_calibration,
        target_area_mask=target_area_mask_calibration,
        device=device,
    )

    # Load raw and UTIS image
    for i, name in enumerate(heliostats):
        results_dict[name] = {}
        utis_image = paint_utils.load_image_as_tensor(
            name, paint_dir, MEASUREMENT_IDS[name], "flux"
        )
        raw_image = paint_utils.load_image_as_tensor(
            name, paint_dir, MEASUREMENT_IDS[name], "cropped"
        )

        results_dict[name]["image_raw"] = raw_image.cpu().detach().numpy()
        results_dict[name]["utis_image"] = utis_image.cpu().detach().numpy()
        results_dict[name]["flux_deflectometry"] = flux[i].cpu().detach().numpy()
        facets = (
            scenario.heliostat_field.heliostat_groups[0]
            .surface_points[i]
            .view(4, 2500, 4)
        )
        facets_decanted = SurfaceGenerator.perform_inverse_canting_and_translation(
            canted_points=facets,
            translation=scenario.heliostat_field.heliostat_groups[
                0
            ].active_facet_translations[i],
            canting=scenario.heliostat_field.heliostat_groups[0].active_cantings[i],
            device=device,
        )

        facets_decanted = (
            facets_decanted[:, :, 2].view(4, 50, 50).cpu().detach().numpy()
        )
        surface = np.block(
            [
                [facets_decanted[2], facets_decanted[0]],
                [facets_decanted[3], facets_decanted[1]],
            ]
        )

        results_dict[name]["surface"] = surface

    torch.save(results_dict, result_file)


def simulate_overexposure(
    img: np.ndarray,
    exposure: float = 5.0,
    i_max: float = 1.0,
    thresh: float = 0.9,
    blur_ksize: int = 31,
    blur_sigma: float = 10,
) -> np.ndarray:
    """Simulate sensor overexposure and bloom effect on a linear image.

    Parameters
    ----------
    img : np.ndarray
        Input image in linear intensity space (normalized to [0, 1]).
    exposure : float
        Exposure multiplier to simulate "overexposure".
    i_max : float
        Maximum sensor capacity (full well) before clipping.
    thresh : float
        Threshold (in normalized units) above which highlights are extracted for bloom.
    blur_ksize : int
        Kernel size for Gaussian blur (must be odd).
    blur_sigma : float
        Sigma value for Gaussian blur.

    Returns
    -------
    np.ndarray
        Overexposed image with bloom/glare effect applied (clamped to [0, i_max]).
    """
    # Scale and hard clip
    i_exp = img * exposure
    i_clip = np.clip(i_exp, 0, i_max)

    # Extract highlights for bloom
    highlights = np.clip(i_exp - thresh, 0, None)

    # Apply Gaussian blur to highlights
    bloom = cv2.GaussianBlur(highlights, (blur_ksize, blur_ksize), blur_sigma)

    # Composite clipped image + bloom, then clamp
    return np.clip(i_clip + bloom, 0, i_max)


def plot_results_flux_comparision(result_file, plot_save_path):
    """Plot a multi-row comparison of raw, UTIS, deflectometry (with overexposure), and surface images for each heliostat.

    Parameters
    ----------
    result_file : path to torch .pt file with results_dict
    plot_save_path : directory to save the figure
    """
    # Load results
    results_dict = torch.load(result_file, weights_only=False)
    num_hel = len(results_dict)

    # Create figure
    fig, ax = plt.subplots(
        num_hel,
        4,
        figsize=(12.5, 2.5 * num_hel),
        gridspec_kw={"width_ratios": [1, 1, 1, 1.2]},
    )
    if num_hel == 1:
        ax = [ax]

    # Define colormaps
    colormaps = ["gray", "hot", "hot", "jet"]

    for i, (name, data) in enumerate(results_dict.items()):
        # Extract images
        raw = data["image_raw"]
        utis = data["utis_image"]
        flux_def = data["flux_deflectometry"]
        surface = data.get("surface", np.zeros_like(raw))

        # Simulate overexposure on deflectometry image
        # flux_def = simulate_overexposure(
        #     flux_def,
        #     exposure=0.18,
        #     i_max=1.0,
        #     thresh=0.5,
        #     blur_ksize=None,
        #     blur_sigma=100
        # )

        # Normalize intensities so each image has max=1
        def normalize(img):
            mx = img.max()
            return img / mx if mx > 0 else img

        raw = normalize(raw)
        utis = normalize(utis)
        flux_def = normalize(flux_def)

        images = [raw, utis, flux_def, surface]

        for j, img in enumerate(images):
            if j < 3:
                ax[i][j].imshow(img, cmap=colormaps[j])
            else:
                # Surface deviation map
                ax[i][j].imshow(
                    img, cmap=colormaps[j], origin="lower", vmin=-0.003, vmax=0.003
                )
            ax[i][j].axis("off")

        # Label rows and adjust aspect
        ax[i][0].set_ylabel(name, rotation=90, labelpad=10, va="center")
        ax[i][3].set_aspect(1 / 1.2)

    # Column titles
    ax[0][0].set_title("Raw Image")
    ax[0][1].set_title("Flux - UTIS")
    ax[0][2].set_title("Flux - Deflectometry")
    ax[0][3].set_title("Surface - Deflectometry")

    fig.tight_layout()
    out_file = f"{plot_save_path}/02a_flux_comparison.pdf"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overexposed flux comparison to {out_file}")

    plt.close(fig)


def plot_error_distributions(result_file, plot_save_path=None):
    """Load results_dict and plot surface z-coordinate distributions for each heliostat.

    Parameters
    ----------
    result_file : path to .pt file
    plot_save_path : directory to save the figure (optional)
    """
    from scipy.stats import gaussian_kde

    # Load the results dict from .pt file
    results_dict = torch.load(result_file, weights_only=False)

    fig, axs = plt.subplots(2, 1, figsize=(4, 7.5), sharex=False)

    # --- Upper subplot: surface Z KDEs ---
    ax_surface = axs[0]
    z_vals_all = []

    ax_surface.axvline(x=0.0, color="black", linestyle="--", label="Ideal")

    colors = [
        helmholtz_colors["hgfblue"],
        helmholtz_colors["hgfgreen"],
        helmholtz_colors["hgfaerospace"],
    ]

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
        flux_deflectometry = results_dict[name]["flux_deflectometry"]

        error_deflec = paint_utils.calculate_flux_deviation(utis, flux_deflectometry)

        flux_error_deflectometry.append(error_deflec)

    ax_flux.scatter(
        x,
        flux_error_ideal,
        label="Ideal",
        color=helmholtz_colors["hgfgray"],
        marker="o",
    )
    ax_flux.scatter(
        x,
        flux_error_deflectometry,
        label="Deflectometry",
        color=helmholtz_colors["hgfdarkblue"],
        marker="x",
    )

    ax_flux.set_xticks(x)
    ax_flux.set_xticklabels(heliostat_names, rotation=45, ha="right")
    ax_flux.set_ylabel("Relative Flux Error / -")
    ax_flux.legend()
    ax_flux.grid(True)

    # --- Save or show ---
    if plot_save_path:
        save_path = plot_save_path + "/02b_error_distributions.pdf"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def load_config():
    """Load local example configuration from config.local.json."""
    script_dir = os.path.dirname(__file__)
    candidates = [
        os.path.join(script_dir, "config.local.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError(
        "No config.local.json found. "
        "Copy config.example.json to config.local.json and customize it."
    )


def main():
    """Run ray tracing example: generate flux images and plots from config."""
    cfg = load_config()
    # Map config -> args using new keys and base paths
    paint_base = pathlib.Path(cfg["paint_repository_base_path"])
    examples_base = pathlib.Path(cfg["examples_data_base_path"])

    def join_safe(
        base: pathlib.Path, maybe_rel: Union[str, pathlib.Path]
    ) -> pathlib.Path:
        s = str(maybe_rel)
        return base / s.lstrip("/\\")

    paint_dir = paint_base
    tower_file = join_safe(paint_base, cfg["paint_tower_file"])
    heliostats = cfg["heliostats"]
    device_str = cfg["device"]
    scenario_base = join_safe(examples_base, cfg["examples_raytracing_scenario_path"])
    result_file = join_safe(examples_base, cfg["examples_results_path"])
    plot_save_path = cfg["examples_save_plot_path"]

    device = torch.device(device_str)
    set_logger_config()
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    plt.rcParams["font.family"] = "sans-serif"

    scenario_h5 = pathlib.Path(scenario_base).with_suffix(".h5")
    if scenario_h5.exists():
        print(
            f"Scenario found at {scenario_h5}... continue without generating scenario."
        )
    else:
        print(f"Scenario not found. Generating a new one at {scenario_h5}...")
        paint_utils.generate_paint_scenario(
            paint_dir=str(paint_dir),
            scenario_path=pathlib.Path(scenario_base),
            tower_file=str(tower_file),
            heliostat_names=heliostats,
            device=device,
        )

    generate_flux_images(
        scenario_path=scenario_base,
        heliostats=heliostats,
        paint_dir=str(paint_dir),
        result_file=str(result_file),
        device=device,
    )
    plot_results_flux_comparision(
        result_file=str(result_file), plot_save_path=str(plot_save_path)
    )
    # plot_error_distributions(result_file=result_file, plot_save_path=plot_save_path)


if __name__ == "__main__":
    main()
