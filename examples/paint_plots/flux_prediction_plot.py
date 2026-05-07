"""
Generate plots based on ray tracing results.

This script loads the results from the ``ARTIST`` raytracing and generates a plot comparing the extracted image, the
image generated with an ideal surface, the image generated with a fitted surface based on deflectometry data, and
the measured deformations in the surface.

Command-Line Arguments
----------------------
config : str
    Path to the configuration file.
device : str
    Device to use for the computation.
results_dir : str
    Path to directory where the results are saved.
plots_dir : str
    Path to the directory where the plots are saved.
"""

import argparse
import pathlib
import warnings

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt

from artist.util.env import get_device


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


def plot_flux_prediction(
    results_file: pathlib.Path, plots_path: pathlib.Path, device: torch.device
) -> None:
    """
    Plot the flux prediction results, saving each plot type as a separate column-plot.

    Parameters
    ----------
    results_file : pathlib.Path
        Path to the results file.
    plots_path : pathlib.Path
        Base path to save the plots to. Suffixes depending on data column is appended.
    device : torch.device
        Device to use.
    """
    device = get_device(device)

    # Set plot style.
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"

    # Load results.
    results_dict: dict[str, dict[str, np.ndarray]] = torch.load(
        results_file,
        weights_only=False,
        map_location=device,
    )
    number_of_heliostats = len(results_dict)

    # Define configurations for each column type.
    colormaps: list[str] = ["gray", "hot", "hot", "plasma"]
    suffixes: list[str] = ["utis", "ideal", "deflectometry", "surface"]

    # Iterate over each column type to create separate figures.
    for j in range(4):
        fig, ax = plt.subplots(
            number_of_heliostats,
            1,
            figsize=(
                3,
                2.5 * number_of_heliostats,
            ),
            gridspec_kw={"hspace": 0.1},
        )

        # Extract axis if only one heliostat.
        if number_of_heliostats == 1:
            ax = [ax]

        for i, (heliostat_name, data) in enumerate(results_dict.items()):
            if j == 0:
                img = normalize(data["utis"])
            elif j == 1:
                img = normalize(data["ideal"])
            elif j == 2:
                img = normalize(data["deflectometry"])
            else:
                utis = normalize(data["utis"])
                img = data.get("surface", np.zeros_like(utis))

            # Generate plots.
            if j < 3:
                ax[i].imshow(img, cmap=colormaps[j])
            else:
                surface = img
                ax[i].scatter(
                    surface[0].cpu().numpy(),
                    surface[1].cpu().numpy(),
                    c=surface[2].cpu().numpy(),
                    cmap=colormaps[j],
                    s=11,
                    vmin=0.000,
                    vmax=0.007,
                )

            ax[i].set_xticks([])
            ax[i].set_yticks([])
            for spine in ax[i].spines.values():
                spine.set_visible(False)

        # Dynamically adjust the save path to include the suffix before the extension.
        out_path = plots_path.with_name(
            f"{plots_path.stem}_{suffixes[j]}{plots_path.suffix}"
        )

        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {suffixes[j]} comparison to {out_path}.")


if __name__ == "__main__":
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
            warnings.warn(f"Error parsing YAML file: {exc}")
    else:
        warnings.warn(
            f"Warning: Configuration file not found at {config_path}. Using defaults."
        )

    # Add remaining arguments to the parser with defaults loaded from the config.
    device_default = config.get("device", "cuda")
    results_dir_default = config.get("results_dir", "./examples/paint_plots/results")
    plots_dir_default = config.get("plots_dir", "./examples/paint_plots/plots")

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
    plot_flux_prediction(
        results_file=results_path, plots_path=plots_path, device=device
    )
