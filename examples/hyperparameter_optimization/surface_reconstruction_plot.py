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
    results_dict: dict[str, dict[str, np.ndarray]] = torch.load(
        results_file,
        weights_only=False,
        map_location=device,
    )

    fig = plt.figure(figsize=(26, 6))
    gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 1])
    gs.update(left=0.03, right=0.97, wspace=0.005)
    axes = [fig.add_subplot(gs[i]) for i in range(5)]

    # Reference flux.
    reference_flux = results_dict["reconstructed"]["measured_flux"][0]
    reference_flux_normalized = (reference_flux - reference_flux.min()) / (
        reference_flux.max() - reference_flux.min()
    )
    axes[0].imshow(reference_flux.cpu().detach(), cmap="gray")
    axes[0].set_title("Reference", fontsize=28)
    axes[0].axis("off")

    # Ideal flux.
    ideal_flux = results_dict["ideal"]["ideal_flux"][0]
    ideal_flux_normalized = (ideal_flux - ideal_flux.min()) / (
        ideal_flux.max() - ideal_flux.min()
    )
    rmse_ideal = torch.sqrt(
        torch.mean((reference_flux_normalized - ideal_flux_normalized) ** 2)
    )
    axes[1].imshow(ideal_flux.cpu().detach(), cmap="gray")
    axes[1].set_title("Ideal", fontsize=28)
    axes[1].axis("off")
    axes[1].text(
        0.5,
        -0.05,
        f"RMSE(Ref, Ideal)={rmse_ideal:.4f}",
        ha="center",
        va="top",
        transform=axes[1].transAxes,
        fontsize=26,
    )

    # Reconstructed flux.
    reconstructed_flux = results_dict["reconstructed"]["reconstructed_flux"][0]
    reconstructed_flux_normalized = (reconstructed_flux - reconstructed_flux.min()) / (
        reconstructed_flux.max() - reconstructed_flux.min()
    )
    rmse_reconstructed = torch.sqrt(
        torch.mean((reference_flux_normalized - reconstructed_flux_normalized) ** 2)
    )
    axes[2].imshow(reconstructed_flux.cpu().detach(), cmap="gray")
    axes[2].set_title("Reconstructed", fontsize=28)
    axes[2].axis("off")
    axes[2].text(
        0.5,
        -0.05,
        f"RMSE(Ref, Recon)={rmse_reconstructed:.4f}",
        ha="center",
        va="top",
        transform=axes[2].transAxes,
        fontsize=26,
    )

    # Angle maps.
    reference_direction = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device)
    normals_r = (
        (
            results_dict["reconstructed"]["normals_reconstructed"][..., :3]
            / torch.linalg.norm(
                results_dict["reconstructed"]["normals_reconstructed"][..., :3],
                axis=-1,
                keepdims=True,
            )
        )
        .cpu()
        .detach()
    )
    normals_d = (
        (
            results_dict["deflectometry"]["normals_deflectometry"][..., :3]
            / torch.linalg.norm(
                results_dict["deflectometry"]["normals_deflectometry"][..., :3],
                axis=-1,
                keepdims=True,
            )
        )
        .cpu()
        .detach()
    )
    ref = (
        (reference_direction[..., :3] / torch.linalg.norm(reference_direction[..., :3]))
        .cpu()
        .detach()
    )

    all_x_r, all_y_r, all_angles_r = [], [], []
    all_x_d, all_y_d, all_angles_d = [], [], []

    for facet_points_r, facet_normals_r, facet_points_d, facet_normals_d in zip(
        results_dict["reconstructed"]["points_reconstructed"][0].cpu().detach(),
        normals_r[0],
        results_dict["deflectometry"]["points_deflectometry"][0].cpu().detach(),
        normals_d[0],
    ):
        # Reconstructed.
        x_r, y_r = facet_points_r[:, 0], facet_points_r[:, 1]
        cos_theta_r = facet_normals_r @ ref
        angles_r = torch.arccos(torch.clip(cos_theta_r, -1.0, 1.0))
        angles_r = torch.clip(angles_r, -0.1, 0.1)
        all_x_r.append(x_r)
        all_y_r.append(y_r)
        all_angles_r.append(angles_r)

        # Deflectometry.
        x_d, y_d = facet_points_d[:, 0], facet_points_d[:, 1]
        cos_theta_d = facet_normals_d @ ref
        angles_d = torch.arccos(torch.clip(cos_theta_d, -1.0, 1.0))
        angles_d = torch.clip(angles_d, -0.1, 0.1)
        all_x_d.append(x_d)
        all_y_d.append(y_d)
        all_angles_d.append(angles_d)

    all_x_r = torch.cat(all_x_r)
    all_y_r = torch.cat(all_y_r)
    all_angles_r = torch.cat(all_angles_r)
    all_x_d = torch.cat(all_x_d)
    all_y_d = torch.cat(all_y_d)
    all_angles_d = torch.cat(all_angles_d)

    sc3 = axes[3].scatter(all_x_d, all_y_d, c=all_angles_d, cmap="viridis", s=20)
    axes[3].set_title("Angle Map (Measured Normals)", fontsize=16)
    axes[3].set_aspect("equal", adjustable="box")
    axes[3].axis("off")
    cbar3 = fig.colorbar(
        sc3, ax=axes[3], orientation="horizontal", fraction=0.046, pad=0.1
    )
    cbar3.set_label("Angle (rad)")

    sc4 = axes[4].scatter(all_x_r, all_y_r, c=all_angles_r, cmap="viridis", s=20)
    axes[4].set_title("Angle Map\n(Reconstructed Normals)", fontsize=28)
    axes[4].set_aspect("equal", adjustable="box")
    axes[4].axis("off")
    cbar4 = fig.colorbar(
        sc4, ax=axes[4], orientation="horizontal", fraction=0.046, pad=0.01
    )
    cbar4.set_ticks([0.000, 0.016])
    cbar4.set_label("Angle (rad)", fontsize=26)
    cbar4.ax.tick_params(labelsize=24)

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
            f"Results file not found: {results_path}. Please run ``surface_reconstruction.py``"
            f"or adjust the location of the results file and try again!"
        )

    plots_path = pathlib.Path(args.plots_dir) / "surface_reconstruction.pdf"
    if not plots_path.parent.is_dir():
        plots_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate and save plots.
    plot_reconstruction_results(
        results_file=results_path, plots_path=plots_path, device=device
    )
