import argparse
import pathlib
import warnings
from typing import Dict

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt


def normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0,1].

    Parameters
    ----------
    image : np.ndarray
        Image to be normalized.

    Returns
    -------
    np.ndarray
        Normalized image.
    """
    maximum_value = float(image.max()) if image.size > 0 else 0.0
    return image / maximum_value if maximum_value > 0 else image


def plot_flux_prediction(results_file: pathlib.Path, plots_path: pathlib.Path) -> None:
    """
    Plot the flux prediction results.

    Parameters
    ----------
    results_file : pathlib.Path
        Path to the results file.
    plots_path : pathlib.Path
        Path to save the plot to.
    """
    # Set Plot style.
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"

    # Load results.
    results_dict: Dict[str, Dict[str, np.ndarray]] = torch.load(
        results_file, weights_only=False
    )
    number_of_heliostats = len(results_dict)

    # Create figure.
    fig, ax = plt.subplots(
        number_of_heliostats,
        4,
        figsize=(12, 2.5 * number_of_heliostats),
        gridspec_kw={
            "width_ratios": [1, 1, 1, 1],
            "wspace": 0.01,
            "hspace": 0.1,
        },
    )

    # Ensure 'ax' is always an array of arrays for consistency
    if number_of_heliostats == 1:
        ax = [ax]

    # Define colormaps.
    colormaps: list[str] = ["gray", "hot", "hot", "plasma"]

    for i, (heliostat_name, data) in enumerate(results_dict.items()):
        # Extract images
        utis = normalize(data["utis"])
        ideal = normalize(data["ideal"])
        deflectometry = normalize(data["deflectometry"])
        surface = data.get("surface", np.zeros_like(utis))

        images = [utis, ideal, deflectometry, surface]

        for j, img in enumerate(images):
            if j < 3:
                ax[i][j].imshow(img, cmap=colormaps[j])
            else:
                # Surface deviation map.
                ax[i][j].imshow(
                    img, cmap=colormaps[j], origin="lower", vmin=-0.003, vmax=0.003
                )
            # Turn off axis ticks and labels for all subplots
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])

        # Label rows
        ax[i][0].set_ylabel(
            f"\\textbf{{{heliostat_name}}}", rotation=90, labelpad=10, va="center"
        )

    # Column titles with a smaller second line
    ax[0][0].set_title("\\textbf{Flux Image}\n{\\small (extracted with UTIS)}")
    ax[0][1].set_title("\\textbf{Generated Flux}\n{\\small (using ideal surface)}")
    ax[0][2].set_title("\\textbf{Generated Flux}\n{\\small (using deflectometry)}")
    ax[0][3].set_title("\\textbf{Surface}\n{\\small (measured by deflectometry)}")

    plt.savefig(plots_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overexposed flux comparison to {plots_path}")

    plt.close(fig)


if __name__ == "__main__":
    """
    Generate plots based on ray tracing results.

    This script loads the results from the ``ARTIST`` raytracing and generates a plot comparing the extracted image, the
    image generated with an ideal surface, the image generated with a fitted surface based on deflectometry data, and
    the measured deformations in the surface.

    Parameters
    ----------
    config : str
        Path to the configuration file.
    results_dir : str
        Path to directory where the results are saved.
    plots_dir : str
        Path to the directory where the plots are saved.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML configuration file.",
        default="./paint_plot_config.yaml",
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
    results_dir_default = config.get("results_dir", "./results")
    plots_dir_default = config.get("plots_dir", "./plots")

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

    results_path = pathlib.Path(args.results_dir) / "flux_prediction_results.pt"
    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_path}. Please run ``flux_prediction_raytracing.py``"
            f"or adjust the location of the results file and try again!"
        )

    plots_path = pathlib.Path(args.plots_dir) / "flux_comparison.pdf"
    if not plots_path.parent.is_dir():
        plots_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate and save plots.
    plot_flux_prediction(results_file=results_path, plots_path=plots_path)
