import argparse
import math
import pathlib
import warnings
from typing import Any

import numpy as np
import paint.util.paint_mappings as paint_mappings
import torch
import yaml
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

from artist.util.environment_setup import get_device

plot_colors = {
    paint_mappings.HELIOS_KEY: "#002864",
    paint_mappings.UTIS_KEY: "#14c8ff",
}


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
    # Set Plot style.
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"

    # Convert losses to list.
    helios_losses_in_meters = [
        math.sqrt(data[paint_mappings.HELIOS_KEY])
        for data in reconstruction_results.values()
    ]
    utis_losses_in_meters = [
        math.sqrt(data[paint_mappings.UTIS_KEY])
        for data in reconstruction_results.values()
    ]
    x_max = max(utis_losses_in_meters + helios_losses_in_meters)
    x_vals = np.linspace(0, x_max, 100)

    kde_helios = gaussian_kde(helios_losses_in_meters, bw_method="scott")
    kde_utis = gaussian_kde(utis_losses_in_meters, bw_method="scott")

    kde_values_helios = kde_helios(x_vals)
    kde_values_utis = kde_utis(x_vals)

    mean_helios = np.mean(helios_losses_in_meters)

    mean_utis = np.mean(utis_losses_in_meters)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Best practice is to plot the histogram with the highest mean first.
    if mean_helios > mean_utis:
        ax.hist(
            helios_losses_in_meters,
            bins=25,
            range=(0, x_max),
            density=True,
            alpha=0.3,
            label="HeliOS Histogram",
            color=plot_colors[paint_mappings.HELIOS_KEY],
        )
        ax.hist(
            utis_losses_in_meters,
            bins=25,
            range=(0, x_max),
            density=True,
            alpha=0.3,
            label="UTIS Histogram",
            color=plot_colors[paint_mappings.UTIS_KEY],
        )
    else:
        ax.hist(
            utis_losses_in_meters,
            bins=25,
            range=(0, x_max),
            density=True,
            alpha=0.3,
            label="UTIS Histogram",
            color=plot_colors[paint_mappings.UTIS_KEY],
        )
        ax.hist(
            helios_losses_in_meters,
            bins=25,
            range=(0, x_max),
            density=True,
            alpha=0.3,
            label="HeliOS Histogram",
            color=plot_colors[paint_mappings.HELIOS_KEY],
        )
    ax.plot(
        x_vals,
        kde_values_helios,
        label="HeliOS KDE",
        color=plot_colors[paint_mappings.HELIOS_KEY],
    )
    ax.axvline(
        mean_helios,
        color=plot_colors[paint_mappings.HELIOS_KEY],
        linestyle="--",
        label=f"HeliOS Mean: {mean_helios:.2f} meter",
    )
    ax.plot(
        x_vals,
        kde_values_utis,
        label="UTIS KDE",
        color=plot_colors[paint_mappings.UTIS_KEY],
    )
    ax.axvline(
        mean_utis,
        color=plot_colors[paint_mappings.UTIS_KEY],
        linestyle="--",
        label=f"UTIS Mean: {mean_utis:.2f} meter",
    )

    ax.set_xlabel("\\textbf{Pointing Error} \n{\\small meter}")
    ax.set_ylabel("\\textbf{Density}")
    ax.legend(fontsize=8)
    ax.grid(True)

    if not save_dir.is_dir():
        save_dir.mkdir(parents=True, exist_ok=True)
    filename = save_dir / "error_distribution.pdf"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved reconstruction error distribution plot at: {filename}.")


def plot_error_against_distance(
    reconstruction_results: dict[str, dict[str, Any]],
    number_of_points_to_plot: int,
    save_dir: pathlib.Path,
    random_seed: int,
) -> None:
    """
    Plot the reconstruction error against the distance.

    This function plots the reconstruction error, in meter, against the distance of that heliostat from the tower.

    Parameters
    ----------
    reconstruction_results : dict[str, dict[str, Any]]
        A dictionary containing the reconstruction results.
    number_of_points_to_plot : int
        Number of points to randomly select and plot.
    save_dir : pathlib.Path
        Directory used for saving the plot.
    random_seed : int
        Random seed for the selection of points to plot.
    """
    # Set plot style.
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"

    # Load as lists.
    positions_list = [data["Position"] for data in reconstruction_results.values()]
    helios_loss_list_in_meters = [
        math.sqrt(data[paint_mappings.HELIOS_KEY])
        for data in reconstruction_results.values()
    ]
    utis_loss_list_in_meters = [
        math.sqrt(data[paint_mappings.UTIS_KEY])
        for data in reconstruction_results.values()
    ]

    # Convert to arrays for plotting.
    positions = np.array(positions_list, dtype=float)
    helios_losses = np.array(helios_loss_list_in_meters, dtype=float)
    utis_losses = np.array(utis_loss_list_in_meters, dtype=float)

    # Vectorized calculation of distances.
    distances = np.linalg.norm(positions[:, :2], axis=1)

    # Randomly select indices to plot.
    np.random.seed(42)
    total_data_points = len(distances)
    if number_of_points_to_plot >= total_data_points:
        selected_indices = np.arange(total_data_points)
    else:
        selected_indices = np.random.choice(
            total_data_points, number_of_points_to_plot, replace=False
        )

    # Select subset of data points to plot.
    distances = distances[selected_indices]
    helios_losses = helios_losses[selected_indices]
    utis_losses = utis_losses[selected_indices]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(
        distances,
        helios_losses,
        color=plot_colors[paint_mappings.HELIOS_KEY],
        marker="o",
        label="HeliOS Mean Error",
        alpha=0.7,
    )

    ax.scatter(
        distances,
        utis_losses,
        color=plot_colors[paint_mappings.UTIS_KEY],
        marker="^",
        label="UTIS Mean Error",
        alpha=0.7,
    )

    # Trendlines.
    helios_fit = np.poly1d(np.polyfit(distances, helios_losses, 1))
    utis_fit = np.poly1d(np.polyfit(distances, utis_losses, 1))
    x_vals = np.linspace(distances.min(), distances.max(), 200)

    ax.plot(
        x_vals,
        helios_fit(x_vals),
        color=plot_colors[paint_mappings.HELIOS_KEY],
        linestyle="--",
        label="HeliOS Trend",
    )
    ax.plot(
        x_vals,
        utis_fit(x_vals),
        color=plot_colors[paint_mappings.UTIS_KEY],
        linestyle="--",
        label="UTIS Trend",
    )

    ax.set_xlabel("\\textbf{Heliostat Distance from Tower} \n{m}")
    ax.set_ylabel("\\textbf{Mean Pointing Error} \n{meter}")
    ax.grid(True)
    ax.legend(fontsize=8, loc="upper right", ncol=2)

    save_path = save_dir / "reconstruction_error_distance.pdf"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(
        f"Saved plot comparing the reconstruction error to the distance at: {save_path}."
    )


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
    default_config_path = script_dir / "paint_plot_config.yaml"

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
    plots_path = pathlib.Path(args.plots_dir)

    plot_error_distribution(
        reconstruction_results=reconstruction_results, save_dir=plots_path
    )
    plot_error_against_distance(
        reconstruction_results=reconstruction_results,
        number_of_points_to_plot=args.number_of_points_to_plot,
        save_dir=plots_path,
        random_seed=args.random_seed,
    )
