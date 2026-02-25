import argparse
import pathlib
import warnings
from typing import Any

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

from artist.util import utils
from artist.util.environment_setup import get_device

plot_colors = {
    "darkblue": "#002864",
    "lightblue": "#14c8ff",
    "darkred": "#cd5c5c",
    "darkgray": "#686868",
}


def plot_kinematic_reconstruction_fluxes(
    reconstruction_results: dict[str, dict[str, Any]], save_dir: pathlib.Path
) -> None:
    """
    Plot the reconstructed fluxes.

    This function plots histograms and kernel density estimations of the pointing errors in reconstruction when comparing
    HeliOS and UTIS as methods for focal spot centroid extraction.

    Parameters
    ----------
    reconstruction_results : dict[str, dict[str, Any]]
        The reconstruction results.
    save_dir : pathlib.Path
        Directory used for saving the plot.
    """
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"
    plt.rcParams["text.latex.preamble"] = r"\setlength{\parindent}{0pt}"

    cmap = "inferno"

    results = reconstruction_results["flux"]

    n_rows = 1
    n_cols = 3

    fig = plt.figure(figsize=(6, 4))
    gs = GridSpec(
        n_rows,
        n_cols,
        figure=fig,
        left=0.02,
        right=0.98,
        top=0.99,
        bottom=0.02,
        wspace=0.01,
        hspace=0.01,
        width_ratios=[1, 1, 1],
    )

    axes = np.empty((n_rows, n_cols), dtype=object)

    for i in range(n_rows):
        for j in range(n_cols):
            axes[i, j] = fig.add_subplot(gs[i, j])
            axes[i, j].axis("off")

    col_labels = [
        "Calibration Flux",
        "Default\\\\Kinematic",
        "Reconstructed\\\\Kinematic",
    ]
    heliostat_names = [list(results.keys())[-1]]
    positions = [
        reconstruction_results["loss"][heliostat]["position"]
        for heliostat in heliostat_names
    ]

    for col_index in range(n_cols):
        axes[0, col_index].set_title(
            rf"\textbf{{{col_labels[col_index]}}}", fontsize=18, ha="center"
        )

    for row_index in range(n_rows):
        flux_data = results[heliostat_names[row_index]]["fluxes"].cpu().detach()
        for col_index in range(n_cols):
            position = positions[row_index]
            position_str = ", ".join(f"{x:.2f}" for x in position[:3])

            axes[row_index, col_index].imshow(flux_data[col_index], cmap=cmap)
            axes[row_index, 0].text(
                -0.05,
                0.5,
                rf"\textbf{{Heliostat: {heliostat_names[row_index]}}}",
                transform=axes[row_index, 0].transAxes,
                fontsize=18,
                ha="right",
                va="center",
            )
            axes[row_index, 0].text(
                -0.05,
                0.4,
                r"\textit{ENU Position:}",
                transform=axes[row_index, 0].transAxes,
                fontsize=12,
                color=plot_colors["darkgray"],
                ha="right",
                va="center",
            )
            axes[row_index, 0].text(
                -0.05,
                0.30,
                rf"\textit{{{position_str}}}",
                transform=axes[row_index, 0].transAxes,
                fontsize=12,
                color=plot_colors["darkgray"],
                ha="right",
                va="center",
            )

    if not save_dir.is_dir():
        save_dir.mkdir(parents=True, exist_ok=True)
    filename = save_dir / "reconstruction_kinematic_fluxes.pdf"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved reconstruction flux plot at: {filename}.")


def plot_error_distribution(
    reconstruction_results: dict[str, dict[str, Any]], save_dir: pathlib.Path
) -> None:
    """
    Plot the distribution of reconstruction errors.

    This function plots histograms and kernel density estimations of the pointing errors in reconstruction when comparing
    HeliOS and UTIS as methods for focal spot centroid extraction.

    Parameters
    ----------
    reconstruction_results : dict[str, dict[str, Any]]
        A dictionary containing the reconstruction results.
    save_dir : pathlib.Path
        Directory used for saving the plot.
    """
    # Set plot style.
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"
    plt.rcParams["text.latex.preamble"] = r"\setlength{\parindent}{0pt}"

    # Convert losses to list.
    errors_in_meters = [
        data["loss"] for data in reconstruction_results["loss"].values()
    ]

    # Convert to angular error in mrad
    positions = np.array(
        [data["position"] for data in reconstruction_results["loss"].values()],
        dtype=float,
    )
    distances = np.linalg.norm(positions[:, :2], axis=1)
    errors_in_mrad = (errors_in_meters / distances) * 1000

    for errors, name, color in zip(
        [errors_in_meters, errors_in_mrad], ["meters", "mrad"], ["lightblue", "darkred"]
    ):
        x_max = max(errors)
        x_vals = np.linspace(0, x_max, 100)
        kde = gaussian_kde(errors, bw_method="scott")
        kde_values = kde(x_vals)
        mean = np.mean(errors)

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.hist(
            errors,
            bins=25,
            range=(0, x_max),
            density=True,
            alpha=0.3,
            label="Loss Histogram",
            color=plot_colors[color],
        )
        ax.plot(
            x_vals,
            kde_values,
            label="KDE",
            color=plot_colors[color],
        )
        ax.axvline(
            mean,
            color=plot_colors[color],
            linestyle="--",
            label=f"Mean: {mean:.2f} {name}",
        )

        ax.set_xlabel(f"\\textbf{{Pointing Error}} \n{{\\small {name}}}")
        ax.set_ylabel("\\textbf{Density}")
        ax.legend(fontsize=8)
        ax.grid(True)

        if not save_dir.is_dir():
            save_dir.mkdir(parents=True, exist_ok=True)
        filename = save_dir / f"error_distribution_{name}.pdf"
        fig.savefig(filename, dpi=300, bbox_inches="tight")

        print(f"Saved reconstruction error distribution plot at: {filename}.")


def plot_linear_and_angular_error_against_distance(
    reconstruction_results: dict[str, dict[str, Any]],
    number_of_points_to_plot: int,
    save_dir: pathlib.Path,
    random_seed: int,
) -> None:
    """
    Plot both reconstruction error in meters (left y-axis) and mrad (right y-axis) against the distance from the tower.

    Parameters
    ----------
    reconstruction_results : dict[str, dict[str, Any]]
        A dictionary containing the reconstruction results.
    number_of_points_to_plot : int
        Number of points to randomly select and plot.
    save_dir : pathlib.Path
        Directory used for saving the plot.
    random_seed : int
        Random seed for reproducibility.
    """
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"
    plt.rcParams["text.latex.preamble"] = r"\setlength{\parindent}{0pt}"

    positions_list = [
        data["position"] for data in reconstruction_results["loss"].values()
    ]
    error_list_in_meters = [
        data["loss"] for data in reconstruction_results["loss"].values()
    ]

    positions = np.array(positions_list, dtype=float)
    errors_in_meters = np.array(error_list_in_meters, dtype=float)

    distances = np.linalg.norm(positions[:, :2], axis=1)

    np.random.seed(random_seed)
    total_data_points = len(distances)
    if number_of_points_to_plot >= total_data_points:
        selected_indices = np.arange(total_data_points)
    else:
        selected_indices = np.random.choice(
            total_data_points, number_of_points_to_plot, replace=False
        )

    distances = distances[selected_indices]
    errors_in_meters = errors_in_meters[selected_indices]
    errors_in_mrad = (errors_in_meters / distances) * 1000

    fig, ax_m = plt.subplots(figsize=(7, 4))
    ax_m.scatter(
        distances,
        errors_in_meters,
        color=plot_colors["lightblue"],
        marker="o",
        label="Error (m)",
        alpha=0.7,
    )

    fit_meters = np.poly1d(np.polyfit(distances, errors_in_meters, 1))
    x_vals = np.linspace(distances.min(), distances.max(), 200)
    ax_m.plot(
        x_vals, fit_meters(x_vals), color=plot_colors["lightblue"], linestyle="--"
    )
    ax_m.set_xlabel("\\textbf{Heliostat Distance from Tower [m]}")
    ax_m.set_ylabel(
        "\\textbf{Mean Pointing Error [m]}",
        color=plot_colors["lightblue"],
    )
    ax_m.grid(True)

    ax_a = ax_m.twinx()
    ax_a.scatter(
        distances,
        errors_in_mrad,
        color=plot_colors["darkred"],
        marker="^",
        label="Error (mrad)",
        alpha=0.7,
    )

    fit_a = np.poly1d(np.polyfit(distances, errors_in_mrad, 1))
    ax_a.plot(x_vals, fit_a(x_vals), color="darkred", linestyle="--")
    ax_a.set_ylabel("\\textbf{Mean Pointing Error [mrad]}", color="darkred")
    ax_a.tick_params(axis="y", labelcolor="black")

    handles_m, labels_m = ax_m.get_legend_handles_labels()
    handles_a, labels_a = ax_a.get_legend_handles_labels()
    ax_m.legend(
        handles_m + handles_a,
        labels_m + labels_a,
        fontsize=8,
        loc="upper right",
        ncol=2,
    )

    save_path = save_dir / "reconstruction_error_distance_dual_axis.pdf"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved dual-axis plot at: {save_path}")


def plot_motor_pos_fluxes(
    reconstruction_results: dict[str, Any], save_dir: pathlib.Path
) -> None:
    """
    Plot the distribution of reconstruction errors.

    This function plots histograms and kernel density estimations of the pointing errors in reconstruction when comparing
    HeliOS and UTIS as methods for focal spot centroid extraction.

    Parameters
    ----------
    reconstruction_results : dict[str, dict[str, Any]]
        A dictionary containing the reconstruction results.
    save_dir : pathlib.Path
        Directory used for saving the plot.
    """
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"

    cmap = "inferno"
    n_cols = 3

    fig = plt.figure(figsize=(9.5, 8))
    gs = GridSpec(
        2,
        n_cols,
        figure=fig,
        left=0.05,
        right=0.95,
        top=0.95,
        bottom=0.15,
        height_ratios=[1, 0.05],
        wspace=0.01,
        hspace=0.01,
    )
    axes = []

    # Compute global min and max for shared color scale
    all_flux_data = [
        reconstruction_results[key].cpu().detach()
        for key in ["flux_before", "flux_after", "target_distribution"]
    ]
    vmin = min([data.min() for data in all_flux_data])
    vmax = max([data.max() for data in all_flux_data])

    for i, key in enumerate(["flux_before", "flux_after", "target_distribution"]):
        ax = fig.add_subplot(gs[0, i])
        ax.axis("off")
        flux_data = reconstruction_results[key].cpu().detach()
        im = ax.imshow(flux_data, cmap=cmap, vmin=vmin, vmax=vmax)  # Shared color scale
        axes.append(ax)

        pos = ax.get_position()
        fig.text(
            x=pos.x0 + pos.width / 2,
            y=pos.y0 - 0.03,
            s=str(flux_data.sum()),
            ha="center",
            va="top",
            fontsize=18,
        )

    axes[0].set_title(r"\textbf{Aim Points Centered}", fontsize=18, ha="center")
    axes[1].set_title(r"\textbf{Aim Points Optimized}", fontsize=18, ha="center")
    axes[2].set_title(r"\textbf{Target Distribution}", fontsize=18, ha="center")

    # Add a single horizontal colorbar beneath all subplots
    cbar_ax = fig.add_subplot(gs[1, :])  # spans all columns
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=14)

    if not save_dir.is_dir():
        save_dir.mkdir(parents=True, exist_ok=True)
    filename = save_dir / "motor_pos_plots.pdf"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved reconstruction flux plot at: {filename}.")


def plot_surface_reconstruction(
    reconstruction_results: dict[str, Any],
    save_dir: pathlib.Path,
) -> None:
    """
    Plot the surface reconstruction results.

    Parameters
    ----------
    results : dict[str, Any]
        Results of the surface reconstruction.
    save_dir : pathlib.Path
        Path to the location where the plots are saved.
    """
    fig, axes = plt.subplots(3, 7, figsize=(35, 15))
    for index, heliostat_name in enumerate(list(reconstruction_results["flux"])[:3]):
        heliostat_data = reconstruction_results["flux"][heliostat_name]
        axes[index, 0].imshow(
            heliostat_data["fluxes"][0].cpu().detach(), cmap="inferno"
        )
        axes[index, 0].set_title("Calibration Flux")
        axes[index, 0].axis("off")

        axes[index, 1].imshow(
            heliostat_data["fluxes"][1].cpu().detach(), cmap="inferno"
        )
        axes[index, 1].set_title("Surface not reconstructed")
        axes[index, 1].axis("off")

        axes[index, 2].imshow(
            heliostat_data["fluxes"][2].cpu().detach(), cmap="inferno"
        )
        axes[index, 2].set_title("Surface reconstructed")
        axes[index, 2].axis("off")

        reference_direction = torch.tensor([0.0, 0.0, 1.0], device=torch.device("cpu"))
        canting = heliostat_data["canting"].cpu().detach()

        # Process original deflectometry data.
        deflectometry = torch.stack(
            (
                reconstruction_results["deflectometry"][heliostat_name][
                    "surface_points"
                ]
                .cpu()
                .detach(),
                reconstruction_results["deflectometry"][heliostat_name][
                    "surface_normals"
                ]
                .cpu()
                .detach(),
            )
        ).reshape(2, 4, -1, 4)
        deflectometry_uncanted = utils.perform_canting(
            canting_angles=canting.expand(2, -1, -1, -1),
            data=deflectometry,
            inverse=True,
            device=torch.device("cpu"),
        )
        deflectometry_points_original = deflectometry_uncanted[0, :, :, :3].reshape(
            -1, 3
        )
        deflectometry_normals_original = torch.nn.functional.normalize(
            deflectometry_uncanted[1, :, :, :3], dim=-1
        ).reshape(-1, 3)
        cos_theta_deflectometry_original = (
            deflectometry_normals_original @ reference_direction
        )
        angles_deflectometry_original = torch.clip(
            torch.arccos(torch.clip(cos_theta_deflectometry_original, -1.0, 1.0)),
            -0.1,
            0.1,
        )
        sc3 = axes[index, 3].scatter(
            x=deflectometry_points_original[:, 0],
            y=deflectometry_points_original[:, 1],
            c=deflectometry_points_original[:, 2],
            cmap="inferno",
            vmin=0.0345,
            vmax=0.036,
        )
        axes[index, 3].set_title("Deflectometry Points original")
        axes[index, 3].axis("off")
        axes[index, 3].set_aspect("equal", adjustable="box")
        cbar3 = fig.colorbar(
            sc3, ax=axes[index, 3], orientation="horizontal", fraction=0.046, pad=0.1
        )
        cbar3.set_label("m")

        sc4 = axes[index, 4].scatter(
            x=deflectometry_points_original[:, 0],
            y=deflectometry_points_original[:, 1],
            c=angles_deflectometry_original,
            cmap="inferno",
            vmin=0.0,
            vmax=0.005,
        )
        axes[index, 4].set_title("Deflectometry normals")
        axes[index, 4].axis("off")
        axes[index, 4].set_aspect("equal", adjustable="box")
        cbar4 = fig.colorbar(
            sc4, ax=axes[index, 4], orientation="horizontal", fraction=0.046, pad=0.1
        )
        cbar4.set_label("Angle (rad)")

        # Process reconstructed data.
        points_uncanted = utils.perform_canting(
            canting_angles=canting.expand(2, -1, -1, -1),
            data=heliostat_data["surface_points"].cpu().detach().reshape(2, 4, -1, 4),
            inverse=True,
            device=torch.device("cpu"),
        )
        normals_uncanted = utils.perform_canting(
            canting_angles=canting.expand(2, -1, -1, -1),
            data=heliostat_data["surface_normals"].cpu().detach().reshape(2, 4, -1, 4),
            inverse=True,
            device=torch.device("cpu"),
        )
        reconstructed_points = points_uncanted[1, :, :, :3].reshape(-1, 3)
        reconstructed_normals = torch.nn.functional.normalize(
            normals_uncanted[1, :, :, :3], dim=-1
        ).reshape(-1, 3)
        cos_theta_reconstructed = reconstructed_normals @ reference_direction
        angles_reconstructed = torch.clip(
            torch.arccos(torch.clip(cos_theta_reconstructed, -1.0, 1.0)), -0.1, 0.1
        )
        sc5 = axes[index, 5].scatter(
            x=reconstructed_points[:, 0],
            y=reconstructed_points[:, 1],
            c=reconstructed_points[:, 2],
            cmap="inferno",
            vmin=0.0345,
            vmax=0.036,
        )
        axes[index, 5].set_title("Reconstructed Surface (Points)")
        axes[index, 5].axis("off")
        axes[index, 5].set_aspect("equal", adjustable="box")
        cbar5 = fig.colorbar(
            sc5, ax=axes[index, 5], orientation="horizontal", fraction=0.046, pad=0.1
        )
        cbar5.set_label("m")

        sc6 = axes[index, 6].scatter(
            x=reconstructed_points[:, 0],
            y=reconstructed_points[:, 1],
            c=angles_reconstructed,
            cmap="inferno",
            vmin=0.0,
            vmax=0.005,
        )
        axes[index, 6].set_title("Reconstructed normals")
        axes[index, 6].axis("off")
        axes[index, 6].set_aspect("equal", adjustable="box")
        cbar6 = fig.colorbar(
            sc6, ax=axes[index, 6], orientation="horizontal", fraction=0.046, pad=0.1
        )
        cbar6.set_label("Angle (rad)")

    plt.tight_layout()
    plt.savefig(
        save_dir / "results_surface_reconstruction.png",
        bbox_inches="tight",
        pad_inches=1,
    )


def plot_heliostat_positions(
    surface_scenario: dict[str, Any],
    kinematic_scenario: dict[str, Any],
    save_dir: pathlib.Path,
) -> None:
    """
    Plot heliostat positions.

    Parameters
    ----------
    surface_scenario : dict[str, Any]
        Results of surface reconstruction.
    kinematic_scenario : dict[str, Any]
        Results of kinematic reconstruction.
    save_dir : pathlib.Path
        Directory to save the plots.
    """
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"
    plt.rcParams["text.latex.preamble"] = r"\setlength{\parindent}{0pt}"

    for scenario in [surface_scenario, kinematic_scenario]:
        positions_list = [data["position"] for data in scenario["loss"].values()]

        index = [i for i, d in enumerate(scenario["loss"].keys()) if "BD32" in d]

        fig, ax = plt.subplots(figsize=(6, 4))

        x = [row[0] for row in positions_list]
        y = [row[1] for row in positions_list]

        ax.scatter(
            x=x,
            y=y,
            c=plot_colors["lightblue"],
            s=2,
        )

        ax.scatter(
            [x[index[0]]],
            [y[index[0]]],
            facecolors="none",
            edgecolors="red",
            s=2,
            linewidths=2,
            label="BD32",
        )

        ax.plot([-2 / 2, 2 / 2], [0, 0], color="red", linewidth=2)
        ax.grid(True)

        ax.set_xlabel("\\textbf{East-West distance to tower [m]}")
        ax.set_ylabel("\\textbf{North-South distance to tower [m]}")
        ax.legend(fontsize=8)
        ax.grid(True)

        if not save_dir.is_dir():
            save_dir.mkdir(parents=True, exist_ok=True)
        filename = save_dir / f"heliostat_positions_{len(positions_list)}.pdf"
        fig.savefig(filename, dpi=300, bbox_inches="tight")

        print(f"Saved position plot at: {filename}.")


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
    results_dir_default = config.get("results_dir", "./examples/paint_plots/results")
    plots_dir_default = config.get("plots_dir", "./examples/paint_plots/plots")
    number_of_points_to_plot_default = config.get("number_of_points_to_plot", 100)
    random_seed_default = config.get("random_seed", 7)

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
    parser.add_argument(
        "--number_of_points_to_plot",
        type=int,
        help="Number of data points to plot in the distance error plot.",
        default=number_of_points_to_plot_default,
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="Random seed for the selection of points to plot.",
        default=random_seed_default,
    )

    # Re-parse the full set of arguments.
    args = parser.parse_args(args=unknown)

    device = get_device(torch.device(args.device))

    results_path = (
        pathlib.Path(args.results_dir) / "kinematic_reconstruction_results.pt"
    )
    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_path}. Please run ``reconstruction_generate_results.py``"
            f"or adjust the location of the results file and try again!"
        )

    reconstruction_results = torch.load(
        results_path,
        weights_only=False,
        map_location=device,
    )

    results_path_motor_pos = (
        pathlib.Path(args.results_dir) / "motor_position_optimization_results.pt"
    )
    if not results_path_motor_pos.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_path_motor_pos}. Please run ``reconstruction_generate_results.py``"
            f"or adjust the location of the results file and try again!"
        )

    results_motor_pos = torch.load(
        results_path_motor_pos,
        weights_only=False,
        map_location=device,
    )

    results_path_surface = (
        pathlib.Path(args.results_dir) / "surface_reconstruction_results.pt"
    )
    if not results_path_surface.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_path_surface}. Please run ``reconstruction_generate_results.py``"
            f"or adjust the location of the results file and try again!"
        )

    results_surface = torch.load(
        results_path_surface,
        weights_only=False,
        map_location=device,
    )

    plots_path = pathlib.Path(args.plots_dir)

    plot_error_distribution(
        reconstruction_results=reconstruction_results, save_dir=plots_path
    )

    plot_linear_and_angular_error_against_distance(
        reconstruction_results=reconstruction_results,
        number_of_points_to_plot=args.number_of_points_to_plot,
        save_dir=plots_path,
        random_seed=args.random_seed,
    )

    plot_kinematic_reconstruction_fluxes(
        reconstruction_results=reconstruction_results, save_dir=plots_path
    )

    plot_surface_reconstruction(
        reconstruction_results=results_surface, save_dir=plots_path
    )

    plot_motor_pos_fluxes(reconstruction_results=results_motor_pos, save_dir=plots_path)

    plot_heliostat_positions(
        surface_scenario=results_surface,
        kinematic_scenario=reconstruction_results,
        save_dir=plots_path,
    )
