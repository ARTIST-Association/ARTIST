import argparse
import pathlib
import warnings

import torch
import yaml
from matplotlib import pyplot as plt

from artist.core.loss_functions import KLDivergenceLoss
from artist.util import utils
from artist.util.environment_setup import get_device


def plot_kinematic_reconstruction(
    results,
    save_dir: pathlib.Path,
) -> None:
    for i, reconstruction in enumerate(["kinematic_reconstruction_ideal_surface", "kinematic_reconstruction_reconstructed_surface"]):
        number_of_heliostats = len(results[reconstruction])
        fig, axes = plt.subplots(number_of_heliostats, 3, figsize=(15, 15))
        for flux_index, heliostat_name in enumerate(results[reconstruction]):
            flux_data = results[reconstruction][heliostat_name]["fluxes"].cpu().detach()
            axes[flux_index, 0].imshow(flux_data[0], cmap="inferno")
            axes[flux_index, 0].set_title("Calibration Flux")
            axes[flux_index, 0].axis("off") 
            
            axes[flux_index, 1].imshow(flux_data[1], cmap="inferno")
            axes[flux_index, 1].set_title("Kinematic not reconstructed")
            axes[flux_index, 1].axis("off")

            axes[flux_index, 2].imshow(flux_data[2], cmap="inferno")
            axes[flux_index, 2].set_title("Kinematic reconstructed")
            axes[flux_index, 2].axis("off")
        
        plt.savefig(save_dir / f"results_kinematic_reconstruction_fluxes_{i}.png", bbox_inches='tight', pad_inches=1)
        plt.close(fig)

    #TODO loss is MSE, for RMSE i need to take the sqrt, for euclidean rmse sqrt(3* MSE)?
    for index, case in enumerate(["ablation_study_case_3", "ablation_study_case_7"]):
        losses = results[case]["kinematic_reconstruction_loss_per_heliostat"].detach().cpu()

        fig, ax = plt.subplots(1, figsize=(12, 6))

        ax.hist(losses, bins=50, edgecolor="black", alpha=0.7, density=True)
        mean_val = losses.mean()
        median_val = losses.median()
        ax.axvline(mean_val, color="green", linestyle="--", linewidth=1.5, label=f"Mean = {mean_val:.3f}")
        ax.axvline(median_val, color="orange", linestyle="--", linewidth=1.5, label=f"Median = {median_val:.3f}")
        ax.set_title("Histogram Losses")
        ax.set_xlabel("Loss")
        ax.set_ylabel("Count")
        ax.legend()

        plt.tight_layout()
        plt.savefig(save_dir / f"results_kinematic_reconstruction_errors_{index}.png", bbox_inches='tight', pad_inches=1)



def plot_surface_reconstruction(
    results,
    save_dir: pathlib.Path,
) -> None:
    number_of_heliostats = len(results["surface_reconstruction"])
    fig, axes = plt.subplots(number_of_heliostats, 7, figsize=(35, 15))
    for index, heliostat_name in enumerate(results["surface_reconstruction"]):
        heliostat_data = results["surface_reconstruction"][heliostat_name]
        axes[index, 0].imshow(heliostat_data["fluxes"][0].cpu().detach(), cmap="inferno")
        axes[index, 0].set_title("Calibration Flux")
        axes[index, 0].axis("off")

        axes[index, 1].imshow(heliostat_data["fluxes"][1].cpu().detach(), cmap="inferno")
        axes[index, 1].set_title("Surface not reconstructed")
        axes[index, 1].axis("off")

        axes[index, 2].imshow(heliostat_data["fluxes"][2].cpu().detach(), cmap="inferno")
        axes[index, 2].set_title("Surface reconstructed")
        axes[index, 2].axis("off") 

        reference_direction = torch.tensor([0.0, 0.0, 1.0], device=torch.device("cpu"))
        canting = heliostat_data["canting"].cpu().detach()
        deflectometry_uncanted_fitted = utils.perform_canting(
            canting=canting.expand(2, -1, -1, -1),
            data=results["deflectometry_fitted"][heliostat_name].reshape(2, 4, -1, 4).cpu().detach(),
            inverse=True,
            device=torch.device("cpu"),
        )
        deflectometry_original = results["deflectometry_original"][heliostat_name].cpu().detach()
        ones = torch.ones(2, 4, 80762, 1, device=torch.device("cpu"))
        deflectometry_original = torch.cat(
            (deflectometry_original, ones), dim=-1
        )
        deflectometry_uncanted_original = utils.perform_canting(
            canting=canting.expand(2, -1, -1, -1),
            data=deflectometry_original,
            inverse=True,
            device=torch.device("cpu"),
        )
        points_uncanted = utils.perform_canting(
            canting=canting.expand(2, -1, -1, -1),
            data=heliostat_data["surface_points"].cpu().detach(),
            inverse=True,
            device=torch.device("cpu"),
        )
        normals_uncanted = utils.perform_canting(
            canting=canting.expand(2, -1, -1, -1),
            data=heliostat_data["surface_normals"].cpu().detach(),
            inverse=True,
            device=torch.device("cpu"),
        )

        deflectometry_normals_fitted = torch.nn.functional.normalize(deflectometry_uncanted_fitted[1, :, :, :3], dim=-1).reshape(-1, 3)
        deflectometry_normals_original = torch.nn.functional.normalize(deflectometry_uncanted_original[1, :, :, :3], dim=-1).reshape(-1, 3)
        ideal_normals = torch.nn.functional.normalize(normals_uncanted[0, :, :, :3], dim=-1).reshape(-1, 3)
        reconstructed_normals = torch.nn.functional.normalize(normals_uncanted[1, :, :, :3], dim=-1).reshape(-1, 3)
        deflectometry_points_fitted = deflectometry_uncanted_fitted[0, :, :, :3].reshape(-1, 3)
        deflectometry_points_original = deflectometry_uncanted_original[0, :, :, :3].reshape(-1, 3)
        ideal_points = points_uncanted[0, :, :, :3].reshape(-1, 3)
        reconstructed_points = points_uncanted[1, :, :, :3].reshape(-1, 3)

        cos_theta_d_fitted = deflectometry_normals_fitted @ reference_direction
        angles_d_fitted =torch.clip( torch.arccos(torch.clip(cos_theta_d_fitted, -1.0, 1.0)),  -0.1, 0.1)
        cos_theta_d_original = deflectometry_normals_original @ reference_direction
        angles_d_original =torch.clip( torch.arccos(torch.clip(cos_theta_d_original, -1.0, 1.0)),  -0.1, 0.1)
        cos_theta_i = ideal_normals @ reference_direction
        angles_i =torch.clip( torch.arccos(torch.clip(cos_theta_i, -1.0, 1.0)),  -0.1, 0.1)
        cos_theta_r = reconstructed_normals @ reference_direction
        angles_r =torch.clip( torch.arccos(torch.clip(cos_theta_r, -1.0, 1.0)),  -0.1, 0.1)

        # sc3 = axes[index, 3].scatter(x=deflectometry_points_fitted[:, 0], y=deflectometry_points_fitted[:, 1], c=angles_d_fitted, cmap="inferno", vmin=0.0, vmax=0.005)
        # axes[index, 3].set_title("Deflectometry normals fitted")
        # axes[index, 3].axis("off") 
        # axes[index, 3].set_aspect("equal", adjustable="box")
        # cbar3 = fig.colorbar(
        #     sc3, ax=axes[index, 3], orientation="horizontal", fraction=0.046, pad=0.1
        # )
        # cbar3.set_label("Angle (rad)")

        sc3 = axes[index, 3].scatter(x=deflectometry_points_original[:, 0], y=deflectometry_points_original[:, 1], c=angles_d_original, cmap="inferno", vmin=0.0, vmax=0.005)
        axes[index, 3].set_title("Deflectometry normals original")
        axes[index, 3].axis("off") 
        axes[index, 3].set_aspect("equal", adjustable="box")
        cbar3 = fig.colorbar(
            sc3, ax=axes[index, 3], orientation="horizontal", fraction=0.046, pad=0.1
        )
        cbar3.set_label("Angle (rad)")

        sc4 = axes[index, 4].scatter(x=reconstructed_points[:, 0], y=reconstructed_points[:, 1], c=angles_r, cmap="inferno", vmin=0.0, vmax=0.005)
        axes[index, 4].set_title("Reconstructed normals")
        axes[index, 4].axis("off") 
        axes[index, 4].set_aspect("equal", adjustable="box")
        cbar4 = fig.colorbar(
            sc4, ax=axes[index, 4], orientation="horizontal", fraction=0.046, pad=0.1
        )
        cbar4.set_label("Angle (rad)")

        # loss_normals = (1 - torch.sum(deflectometry_normals * reconstructed_normals, dim=-1)).reshape(-1)
        # sc5 = axes[index, 5].scatter(deflectometry_points[:, 0], deflectometry_points[:, 1], c=loss_normals, cmap="inferno", vmin=-0.00001, vmax=0.00001)
        # axes[index, 5].set_title("Cosine Similarity")
        # axes[index, 5].set_aspect("equal", adjustable="box")
        # axes[index, 5].axis("off")
        # cbar5 = fig.colorbar(
        #     sc5, ax=axes[index, 5], orientation="horizontal", fraction=0.046, pad=0.1
        # )
        # cbar5.set_label("difference")
        #cbar5.ax.tick_params(labelsize=24)

        # sc6 = axes[index, 6].scatter(x=deflectometry_points_fitted[:, 0], y=deflectometry_points_fitted[:, 1], c=deflectometry_points_fitted[:, 2], cmap="inferno", vmin=0.033, vmax=0.0375)
        # axes[index, 6].set_title("Deflectometry Points fitted")
        # axes[index, 6].axis("off") 
        # axes[index, 6].set_aspect("equal", adjustable="box")
        # cbar6 = fig.colorbar(
        #     sc6, ax=axes[index, 6], orientation="horizontal", fraction=0.046, pad=0.1
        # )
        # cbar6.set_label("m")
    
        sc5 = axes[index, 5].scatter(x=deflectometry_points_original[:, 0], y=deflectometry_points_original[:, 1], c=deflectometry_points_original[:, 2], cmap="inferno", vmin=reconstructed_points[:, 2].min(), vmax=reconstructed_points[:, 2].max())
        axes[index, 5].set_title("Deflectometry Points original")
        axes[index, 5].axis("off") 
        axes[index, 5].set_aspect("equal", adjustable="box")
        cbar5 = fig.colorbar(
            sc5, ax=axes[index, 5], orientation="horizontal", fraction=0.046, pad=0.1
        )
        cbar5.set_label("m")
        cbar5.set_ticks(torch.linspace(reconstructed_points[:, 2].min(), reconstructed_points[:, 2].max(), 6))


        sc6 = axes[index, 6].scatter(x=reconstructed_points[:, 0], y=reconstructed_points[:, 1], c=reconstructed_points[:, 2], cmap="inferno", vmin=reconstructed_points[:, 2].min(), vmax=reconstructed_points[:, 2].max())
        axes[index, 6].set_title("Reconstructed Surface (Points)")
        axes[index, 6].axis("off") 
        axes[index, 6].set_aspect("equal", adjustable="box")
        cbar6 = fig.colorbar(
            sc6, ax=axes[index, 6], orientation="horizontal", fraction=0.046, pad=0.1
        )
        cbar6.set_label("m")
        cbar6.set_ticks(torch.linspace(reconstructed_points[:, 2].min(), reconstructed_points[:, 2].max(), 6))


    plt.tight_layout()
    plt.savefig(save_dir / "results_surface_reconstruction.png", bbox_inches='tight', pad_inches=1)

    losses = results["ablation_study_case_5"]["surface_reconstruction_loss_per_heliostat"]
    losses = torch.cat([loss[~torch.isinf(loss)] for loss in losses]).cpu().detach()

    fig, ax = plt.subplots(1, figsize=(12, 6))

    ax.hist(losses, bins=50, edgecolor="black", alpha=0.7, density=True)
    mean_val = losses.mean()
    median_val = losses.median()
    ax.axvline(mean_val, color="green", linestyle="--", linewidth=1.5, label=f"Mean = {mean_val:.3f}")
    ax.axvline(median_val, color="orange", linestyle="--", linewidth=1.5, label=f"Median = {median_val:.3f}")
    ax.set_title("Histogram Losses")
    ax.set_xlabel("Loss")
    ax.set_ylabel("Count")
    ax.legend()


    plt.tight_layout()
    plt.savefig(save_dir / "results_surface_reconstruction_errors.png", bbox_inches='tight', pad_inches=1)



def plot_aim_point_optimization(
    results,
    save_dir: pathlib.Path,
) -> None:
    measured_flux = results["measured_flux"]
    homogeneous_distribution = results["homogeneous_distribution"]
    unoptimized_flux = results["ablation_study_case_7"]["flux"].squeeze(0)
    optimized_flux = results["ablation_study_case_8"]["flux"].squeeze(0)

    # Norm bitmaps.
    factor = unoptimized_flux.sum() / measured_flux.sum()
    unoptimized_flux_normed = unoptimized_flux / factor
    optimized_flux_normed = optimized_flux / factor
    homogeneous_distribution_normed = homogeneous_distribution * optimized_flux.sum() / factor

    # 1. KL-Divergence between measured flux and homogeneous distribution.
    # 2. KL-Divergence between reconstructed field but unoptimized aim points and homogenous distribution.
    # 3. KL-Divergence between reconstructed field with optimized aim points and homogeneous distribution.
    kl_divergence = KLDivergenceLoss()
    kl_div_measured_homogeneous = kl_divergence(
        measured_flux.unsqueeze(0),
        homogeneous_distribution_normed.unsqueeze(0),
        reduction_dimensions=(1, 2)
    )
    kl_div_unoptimized_homogeneous = kl_divergence(
        unoptimized_flux_normed.unsqueeze(0),
        homogeneous_distribution_normed.unsqueeze(0),
        reduction_dimensions=(1, 2)
    )
    kl_div_optimized_homogeneous = kl_divergence(
        optimized_flux_normed.unsqueeze(0),
        homogeneous_distribution_normed.unsqueeze(0),
        reduction_dimensions=(1, 2)
    )
    
    images = [
        measured_flux.cpu().detach(),
        homogeneous_distribution_normed.cpu().detach(),
        unoptimized_flux_normed.cpu().detach(),
        optimized_flux_normed.cpu().detach()
    ]
    vmin = min(img.min() for img in images)
    vmax = max(img.max() for img in images)

    fig, axes = plt.subplots(1, 4, figsize=(30, 6))

    im0 = axes[0].imshow(images[0], cmap="inferno", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"kl_div_measured_homogeneous: {kl_div_measured_homogeneous.item():.4f}\nBaseline")

    axes[1].imshow(images[1], cmap="inferno", vmin=vmin, vmax=vmax)
    axes[1].set_title("homogeneous_distribution\nExample Optimal Flux")

    axes[2].imshow(images[2], cmap="inferno", vmin=vmin, vmax=vmax)
    axes[2].set_title(f"kl_div_unoptimized_homogeneous: {kl_div_unoptimized_homogeneous.item():.4f}\nIntensity: 100%")

    axes[3].imshow(images[3], cmap="inferno", vmin=vmin, vmax=vmax)
    axes[3].set_title(f"kl_div_optimized_homogeneous: {kl_div_optimized_homogeneous.item():.4f}\nIntensity: {images[3].sum() / images[2].sum() * 100:.4f}%")

    for ax in axes:
        ax.axis("off")

    fig.colorbar(im0, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)

    plt.savefig(save_dir / "results.png", bbox_inches='tight', pad_inches=1)
    plt.close(fig)


if __name__ == "__main__":
    """
    Generate plots based on the kinematic reconstruction results.

    This script loads the results from the ``ARTIST`` reconstruction and generates two plots, one comparing the loss when
    using different centroid extraction methods and one comparing the loss as a function of distance from the tower.

    Parameters
    ----------
    config : str
        Path to the configuration file.
    device : str
        Device to use for the computation.
    results_dir : str
        Path to directory where the results are saved.
    plots_dir : str
        Path to the directory where the plots are saved.
    number_of_points_to_plot : int
        Number of data points to plot in the distance error plot.
    random_seed : int
        Random seed for the selection of points to plot.
    """

    # Set default location for configuration file.
    script_dir = pathlib.Path(__file__).resolve().parent
    default_config_path = script_dir / "config.yaml"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML configuration file.",
        default=default_config_path,
    )

    # Parse the config argument first to load the configuration.
    args, unknown = parser.parse_known_args()
    config_path = pathlib.Path(args.config)
    config = {}
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            warnings.warn(f"Error parsing YAML file: {exc}.")
    else:
        warnings.warn(
            f"Warning: Configuration file not found at {config_path}. Using defaults."
        )

    # Add remaining arguments to the parser with defaults loaded from the config.
    device_default = config.get("device", "cuda")
    results_dir_default = config.get("results_dir", "./results")
    plots_dir_default = config.get("plots_dir", "./plots")

    parser.add_argument(
        "--device",
        type=str,
        help="Device to use.",
        default=device_default,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Path to load the results.",
        default=results_dir_default,
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        help="Path to save the plots.",
        default=plots_dir_default,
    )

    # Re-parse the full set of arguments.
    args = parser.parse_args(args=unknown)

    device = get_device(torch.device(args.device))

    # for case in ["baseline", "full_field"]:
    for case in ["baseline"]:

        plots_path = pathlib.Path(args.plots_dir) / case
        
        plots_path.mkdir(parents=True, exist_ok=True)

        results_path = (
            pathlib.Path(args.results_dir) / case / "results_3.pt"
        )

        if not results_path.exists():
            raise FileNotFoundError(
                f"Results file not found: {results_path}. Please run ``reconstruction_generate_results.py``"
                f"or adjust the location of the results file and try again!"
            )

        results = torch.load(
            results_path,
            weights_only=False,
            map_location=device,
        )

        plot_aim_point_optimization(
            results=results, save_dir=plots_path
        )

        plot_kinematic_reconstruction(
            results=results, save_dir=plots_path
        )
        
        plot_surface_reconstruction(
            results=results, save_dir=plots_path
        )