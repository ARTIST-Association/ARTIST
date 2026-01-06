import argparse
import pathlib
import warnings

import numpy as np
import torch
import yaml
from matplotlib import gridspec
from matplotlib import pyplot as plt

from artist.util.environment_setup import get_device


def plot_reconstruction_results(
    results_file: pathlib.Path, plots_path: pathlib.Path, device: torch.device
) -> None:
    """
    Plot the flux prediction results.

    Parameters
    ----------
    results_file : pathlib.Path
        Path to the results file.
    plots_path : pathlib.Path
        Path to save the plot to.
    device : torch.device
        Device to use.
    """
    device = get_device(device)

    # Load results.
    results_dicts: list[dict[str, dict[str, np.ndarray]]] = torch.load(
        results_file,
        weights_only=False,
        map_location=device,
    )

    num_surfaces = results_dicts["reconstructed"]["measured_flux"].shape[0]

    fig = plt.figure(figsize=(34, 20))
    gs = gridspec.GridSpec(3, 6, width_ratios=[1, 1, 1, 1, 1, 1])
    gs.update(left=0.03, right=0.97, wspace=0.005)
    axes = [fig.add_subplot(gs[i]) for i in range(18)]

    # Reference flux.
    for i in range(num_surfaces):
        reference_flux = results_dicts["reconstructed"]["measured_flux"][i]
        reference_flux_normalized = (reference_flux - reference_flux.min()) / (
            reference_flux.max() - reference_flux.min()
        )
        axes[i % 6 + (i * 5)].imshow(reference_flux.cpu().detach(), cmap="gray")
        axes[i % 6 + (i * 5)].set_title("Reference", fontsize=28)
        axes[i % 6 + (i * 5)].axis("off")

        # Ideal flux.
        ideal_flux = results_dicts["ideal"]["ideal_flux"][i]
        ideal_flux_normalized = (ideal_flux - ideal_flux.min()) / (
            ideal_flux.max() - ideal_flux.min()
        )
        rmse_ideal = torch.sqrt(
            torch.mean((reference_flux_normalized - ideal_flux_normalized) ** 2)
        )
        axes[i % 6 + 1 + (i * 5)].imshow(ideal_flux.cpu().detach(), cmap="gray")
        axes[i % 6 + 1 + (i * 5)].set_title("Ideal", fontsize=28)
        axes[i % 6 + 1 + (i * 5)].axis("off")
        axes[i % 6 + 1 + (i * 5)].text(
            0.5,
            -0.05,
            f"RMSE(Ref, Ideal)={rmse_ideal:.4f}",
            ha="center",
            va="top",
            transform=axes[i % 6 + 1 + (i * 6)].transAxes,
            fontsize=26,
        )

        # Reconstructed flux.
        reconstructed_flux = results_dicts["reconstructed"]["reconstructed_flux"][i]
        reconstructed_flux_normalized = (
            reconstructed_flux - reconstructed_flux.min()
        ) / (reconstructed_flux.max() - reconstructed_flux.min())
        rmse_reconstructed = torch.sqrt(
            torch.mean((reference_flux_normalized - reconstructed_flux_normalized) ** 2)
        )
        axes[i % 6 + 2 + (i * 5)].imshow(reconstructed_flux.cpu().detach(), cmap="gray")
        axes[i % 6 + 2 + (i * 5)].set_title("Reconstructed", fontsize=28)
        axes[i % 6 + 2 + (i * 5)].axis("off")
        axes[i % 6 + 2 + (i * 5)].text(
            0.5,
            -0.05,
            f"RMSE(Ref, Recon)={rmse_reconstructed:.4f}",
            ha="center",
            va="top",
            transform=axes[i % 6 + 2].transAxes,
            fontsize=26,
        )

        # Angle maps.
        reference_direction = torch.tensor([0.0, 0.0, 1.0], device=device).cpu()
        normals_r = (
            (
                results_dicts["reconstructed"]["normals"][..., :3]
                / torch.linalg.norm(
                    results_dicts["reconstructed"]["normals"][..., :3],
                    axis=-1,
                    keepdims=True,
                )
            )
            .cpu()
            .detach()
        )
        normals_d = (
            (
                results_dicts["deflectometry"]["normals"][..., :3]
                / torch.linalg.norm(
                    results_dicts["deflectometry"]["normals"][..., :3],
                    axis=-1,
                    keepdims=True,
                )
            )
            .cpu()
            .detach()
        )
        # normals_i = (
        #     (
        #         results_dicts["ideal_surface"]["normals"][..., :3]
        #         / torch.linalg.norm(
        #             results_dicts["ideal_surface"]["normals"][..., :3],
        #             axis=-1,
        #             keepdims=True,
        #         )
        #     )
        #     .cpu()
        #     .detach()
        # )

        facet_points_flat_r = (
            results_dicts["reconstructed"]["points"][:, :, :, :3]
            .reshape(num_surfaces, -1, 3)
            .cpu()
            .detach()
        )
        facet_normals_flat_r = normals_r.reshape(num_surfaces, -1, 3).cpu().detach()
        x_r = facet_points_flat_r[:, :, 0]
        y_r = facet_points_flat_r[:, :, 1]
        cos_theta_r = facet_normals_flat_r @ reference_direction
        angles_r = torch.clip(
            torch.arccos(torch.clip(cos_theta_r, -1.0, 1.0)), -0.1, 0.1
        )

        facet_points_flat_d = (
            results_dicts["deflectometry"]["points"][:, :, :, :3]
            .reshape(num_surfaces, -1, 3)
            .cpu()
            .detach()
        )
        facet_normals_flat_d = normals_d.reshape(num_surfaces, -1, 3).cpu().detach()
        x_d = facet_points_flat_d[:, :, 0]
        y_d = facet_points_flat_d[:, :, 1]
        cos_theta_d = facet_normals_flat_d @ reference_direction
        angles_d = torch.clip(
            torch.arccos(torch.clip(cos_theta_d, -1.0, 1.0)), -0.1, 0.1
        )

        # facet_points_flat_i = results_dicts["ideal_surface"]["points"][
        #     :, :, :, :3
        # ].reshape(num_surfaces, -1, 3)
        # facet_normals_flat_i = normals_i.reshape(num_surfaces, -1, 3)
        # x_i = facet_points_flat_i[:, :, 0]
        # y_i = facet_points_flat_i[:, :, 1]
        # cos_theta_i = facet_normals_flat_i @ reference_direction
        # angles_i = torch.clip(
        #     torch.arccos(torch.clip(cos_theta_i, -1.0, 1.0)), -0.1, 0.1
        # )

        sc3 = axes[i % 6 + 3 + (i * 5)].scatter(
            x_d[i], y_d[i], c=angles_d[i], cmap="viridis", s=20, vmin=0.0, vmax=0.006
        )
        axes[i % 6 + 3 + (i * 5)].set_title("Measured Angle Map", fontsize=16)
        axes[i % 6 + 3 + (i * 5)].set_aspect("equal", adjustable="box")
        axes[i % 6 + 3 + (i * 5)].axis("off")
        cbar3 = fig.colorbar(
            sc3,
            ax=axes[i % 6 + 3 + (i * 5)],
            orientation="horizontal",
            fraction=0.046,
            pad=0.1,
        )
        cbar3.set_label("Angle (rad)")

        sc4 = axes[i % 6 + 4 + (i * 5)].scatter(
            x_r[i], y_r[i], c=angles_r[i], cmap="viridis", s=20, vmin=0.0, vmax=0.006
        )
        axes[i % 6 + 4 + (i * 5)].set_title("Reconstructed Angle Map", fontsize=28)
        axes[i % 6 + 4 + (i * 5)].set_aspect("equal", adjustable="box")
        axes[i % 6 + 4 + (i * 5)].axis("off")
        cbar4 = fig.colorbar(
            sc4,
            ax=axes[i % 6 + 4 + (i * 5)],
            orientation="horizontal",
            fraction=0.046,
            pad=0.01,
        )
        cbar4.set_label("Angle (rad)", fontsize=26)
        cbar4.ax.tick_params(labelsize=24)

        loss_normals = (1 - torch.sum(normals_d[i] * normals_r[i], dim=-1)).reshape(-1)
        sc5 = axes[i % 6 + 5 + (i * 5)].scatter(
            x_r[i], y_r[i], c=loss_normals, cmap="viridis", s=20
        )
        axes[i % 6 + 5 + (i * 5)].set_title("Cosine Similarity", fontsize=28)
        axes[i % 6 + 5 + (i * 5)].set_aspect("equal", adjustable="box")
        axes[i % 6 + 5 + (i * 5)].axis("off")
        cbar5 = fig.colorbar(
            sc5,
            ax=axes[i % 6 + 5 + (i * 5)],
            orientation="horizontal",
            fraction=0.046,
            pad=0.01,
        )
        cbar5.set_label("Angle (rad)", fontsize=26)
        cbar5.ax.tick_params(labelsize=24)

    plt.tight_layout()
    plt.savefig(plots_path, dpi=300, bbox_inches="tight")

    print(f"Saved flux comparison to {plots_path}.")

    plt.close("all")


if __name__ == "__main__":
    """
    Generate plots based on the reconstruction results.

    This script loads the results from the ``ARTIST`` reconstruction and generates a plot comparing the fluxes,
    from the ideal, reconstructed and measured images, as well as the measured surface with the reconstructed
    surface.

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
    """
    # Set default location for configuration file.
    script_dir = pathlib.Path(__file__).resolve().parent
    default_config_path = script_dir / "hpo_config.yaml"

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
            warnings.warn(f"Error parsing YAML file: {exc}")
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

    results_path = pathlib.Path(args.results_dir) / "surface_reconstruction_results.pt"

    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_path}. Please run ``surface_reconstruction_results.py``"
            f"or adjust the location of the results file and try again!"
        )

    plots_path = pathlib.Path(args.plots_dir) / "surface_reconstruction.png"
    if not plots_path.parent.is_dir():
        plots_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate and save plots.
    plot_reconstruction_results(
        results_file=results_path, plots_path=plots_path, device=device
    )
