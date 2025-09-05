import pathlib
from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt

from artist.util import config_dictionary, set_logger_config
from examples.paint_plots.helpers import join_safe, load_config

torch.manual_seed(7)
torch.cuda.manual_seed(7)

FIGSIZE = (6, 4)
LEGEND_FONTSIZE = 8


# Set up logger.
set_logger_config()


def plot_mrad_error_distributions(
    results_dict: dict, save_path: Optional[str | pathlib.Path] = None
) -> plt.Figure:
    """Plot histograms and KDEs of mrad losses (0–10 mrad) for HeliOS and UTIS across all heliostats.

    Parameters
    ----------
    results_dict : dict
        Mapping from heliostat name to entries for config_dictionary.paint_helios and
        config_dictionary.paint_utis (each an array-like of losses in radians).
    save_path : str | pathlib.Path | None, default=None
        Directory to save the PDF plot if provided.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    from scipy.stats import gaussian_kde

    helios_losses: list[float] = []
    utis_losses: list[float] = []

    for data in results_dict.values():
        helios_losses.extend(np.asarray(data[config_dictionary.paint_helios]) * 1000.0)
        utis_losses.extend(np.asarray(data[config_dictionary.paint_utis]) * 1000.0)

    # Handle empty inputs.
    if len(helios_losses) == 0 or len(utis_losses) == 0:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.set_title("No data available to plot.")
        ax.axis("off")
        if save_path:
            save_dir = pathlib.Path(save_path)
            filename = (save_dir / "plot_mrad_distribution").with_suffix(".pdf")
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Saved error distribution plot to {filename}")
        plt.close(fig)
        return fig

    x_vals = np.linspace(0, 10, 100)

    kde_helios = gaussian_kde(helios_losses, bw_method="scott")
    kde_utis = gaussian_kde(utis_losses, bw_method="scott")

    kde_vals_helios = kde_helios(x_vals)
    kde_vals_utis = kde_utis(x_vals)

    mean_helios = np.mean(helios_losses)
    mode_helios = x_vals[np.argmax(kde_vals_helios)]

    mean_utis = np.mean(utis_losses)
    mode_utis = x_vals[np.argmax(kde_vals_utis)]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.hist(
        helios_losses,
        bins=25,
        range=(0, 10),
        density=True,
        alpha=0.3,
        label="HeliOS Histogram",
        color=config_dictionary.helmholtz_colors["hgfblue"],
    )
    ax.plot(
        x_vals,
        kde_vals_helios,
        label="HeliOS KDE",
        color=config_dictionary.helmholtz_colors["hgfblue"],
    )
    ax.axvline(
        mode_helios,
        color=config_dictionary.helmholtz_colors["hgfblue"],
        linestyle="--",
        label=f"HeliOS Mode: {mode_helios:.2f} mrad",
    )
    ax.axvline(
        mean_helios,
        color=config_dictionary.helmholtz_colors["hgfblue"],
        linestyle=":",
        label=f"HeliOS Mean: {mean_helios:.2f} mrad",
    )

    ax.hist(
        utis_losses,
        bins=25,
        range=(0, 10),
        density=True,
        alpha=0.3,
        label="UTIS Histogram",
        color=config_dictionary.helmholtz_colors["hgfenergy"],
    )
    ax.plot(
        x_vals,
        kde_vals_utis,
        label="UTIS KDE",
        color=config_dictionary.helmholtz_colors["hgfenergy"],
    )
    ax.axvline(
        mode_utis,
        color=config_dictionary.helmholtz_colors["hgfenergy"],
        linestyle="--",
        label=f"UTIS Mode: {mode_utis:.2f} mrad",
    )
    ax.axvline(
        mean_utis,
        color=config_dictionary.helmholtz_colors["hgfenergy"],
        linestyle=":",
        label=f"UTIS Mean: {mean_utis:.2f} mrad",
    )

    ax.set_xlabel("Pointing Error / mrad")
    ax.set_ylabel("Density / -")
    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.grid(True)

    if save_path:
        save_dir = pathlib.Path(save_path)
        filename = (save_dir / "plot_mrad_distribution").with_suffix(".pdf")
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved error distribution plot to {filename}")

    plt.close(fig)
    return fig


def plot_mrad_vs_distance(
    results_dict: dict, save_path: Optional[str | pathlib.Path] = None
) -> plt.Figure:
    """Plot mean pointing error (mrad) vs. heliostat XY distance with trendlines.

    Parameters
    ----------
    results_dict : dict
        Dictionary with structure:
            {
                heliostat_name: {
                    config_dictionary.paint_helios: [...],
                    config_dictionary.paint_utis: [...],
                    "position": [x, y, z]
                },
                ...
            }
    save_path : str | pathlib.Path | None, default=None
        Directory to save the PDF plot if provided.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    distances = []
    helios_means = []
    utis_means = []

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for idx, (name, data) in enumerate(results_dict.items()):
        pos = np.array(data["position"])
        distance = np.linalg.norm(pos[:2])

        helios_mean = (
            np.mean(data[config_dictionary.paint_helios]) * 1000
        )  # results are calculated in rad but plotted in mrad.
        utis_mean = (
            np.mean(data[config_dictionary.paint_utis]) * 1000
        )  # results are calculated in rad but plotted in mrad.

        distances.append(distance)
        helios_means.append(helios_mean)
        utis_means.append(utis_mean)

        ax.scatter(
            distance,
            helios_mean,
            color=config_dictionary.helmholtz_colors["hgfblue"],
            marker="o",
            label="HeliOS Mean Error per Heliostat" if idx == 0 else None,
            alpha=0.7,
        )
        ax.scatter(
            distance,
            utis_mean,
            color=config_dictionary.helmholtz_colors["hgfenergy"],
            marker="o",
            label="UTIS Mean Error per Heliostat" if idx == 0 else None,
            alpha=0.7,
        )

    distances = np.array(distances)
    helios_means = np.array(helios_means)
    utis_means = np.array(utis_means)

    # Trendlines
    helios_fit = np.poly1d(np.polyfit(distances, helios_means, 1))
    utis_fit = np.poly1d(np.polyfit(distances, utis_means, 1))
    x_vals = np.linspace(distances.min(), distances.max(), 200)

    ax.plot(
        x_vals,
        helios_fit(x_vals),
        color=config_dictionary.helmholtz_colors["hgfblue"],
        linestyle="--",
        label="HeliOS Trend",
    )
    ax.plot(
        x_vals,
        utis_fit(x_vals),
        color=config_dictionary.helmholtz_colors["hgfenergy"],
        linestyle="--",
        label="UTIS Trend",
    )

    ax.set_xlabel("Heliostat Distance to Tower / m")
    ax.set_ylabel("Mean Pointing Error / mrad")
    ax.grid(True)
    ax.legend(fontsize=8, loc="best", ncol=2)

    if save_path:
        save_dir = pathlib.Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = (save_dir / "mrad_vs_distance").with_suffix(".pdf")
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved distance vs. error plot to {filename}")

    plt.close(fig)
    return fig


config = load_config()

paint_plots_base_path = pathlib.Path(config["base_path"])
results_path = join_safe(paint_plots_base_path, config["results_calibration_dict_path"])
save_plot_path = join_safe(paint_plots_base_path, config["results_plot_path"])

if results_path.exists():
    results_dict = torch.load(results_path, weights_only=False)
else:
    print(
        f"Did not found existing results at {results_path}. please run 02_run_calibration first."
    )


plot_mrad_vs_distance(results_dict, save_path=save_plot_path)
plot_mrad_error_distributions(results_dict, save_path=save_plot_path)
