from pathlib import Path
from typing import Dict

from matplotlib import pyplot as plt
import numpy as np
import torch

from examples.paint_plots.helpers import join_safe, load_config


def plot_results_flux_comparision(
    result_file: str | Path, plot_save_path: str | Path
) -> None:
    """Plot and save a multi-row comparison of raw, UTIS, ideal, deflectometry, and surface images.

    For each heliostat contained in the results file, this function creates a row with:
    - Raw camera image
    - UTIS flux image
    - Ideal (raytraced) flux image
    - Deflectometry-based simulated flux image
    - Deflectometry-derived surface deviation map

    The resulting figure is saved as a PDF in the specified output directory.

    Parameters
    ----------
    result_file : str | Path
        Path to the torch .pt file containing a results_dict as produced by generate_flux_images.
        The dict is expected to be of the form:
        {
            "<heliostat_name>": {
                "image_raw": np.ndarray,
                "utis_image": np.ndarray,
                "flux_ideal": np.ndarray,
                "flux_deflectometry": np.ndarray,
                "surface": np.ndarray (optional)
            },
            ...
        }
    plot_save_path : str | Path
        Directory path where the output PDF will be written.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the result_file does not exist or the plot_save_path directory does not exist.
    KeyError
        If expected keys are missing for a heliostat entry (e.g., "image_raw", "utis_image", "flux_ideal", "flux_deflectometry").
    """
    # Load results
    if not Path(result_file).exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")
    if not Path(plot_save_path).exists():
        raise FileNotFoundError(f"Output directory does not exist: {plot_save_path}")

    results_dict: Dict[str, Dict[str, np.ndarray]] = torch.load(
        result_file, weights_only=False
    )
    num_hel = len(results_dict)

    # Create figure
    fig, ax = plt.subplots(
        num_hel,
        5,
        figsize=(15.5, 2.5 * num_hel),
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.2]},
    )
    if num_hel == 1:
        ax = [ax]

    # Define colormaps
    colormaps: list[str] = ["gray", "hot", "hot", "hot", "jet"]

    for i, (name, data) in enumerate(results_dict.items()):
        # Extract images
        raw = data["image_raw"]
        utis = data["utis_image"]
        flux_ideal = data["flux_ideal"]
        flux_def = data["flux_deflectometry"]
        surface = data.get("surface", np.zeros_like(raw))

        # Normalize intensities so each image has max=1
        def normalize(img: np.ndarray) -> np.ndarray:
            mx = float(img.max()) if img.size > 0 else 0.0
            return img / mx if mx > 0 else img

        raw = normalize(raw)
        utis = normalize(utis)
        flux_ideal = normalize(flux_ideal)
        flux_def = normalize(flux_def)

        images = [raw, utis, flux_ideal, flux_def, surface]

        for j, img in enumerate(images):
            if j < 4:
                ax[i][j].imshow(img, cmap=colormaps[j])
            else:
                # Surface deviation map
                ax[i][j].imshow(
                    img, cmap=colormaps[j], origin="lower", vmin=-0.003, vmax=0.003
                )
            ax[i][j].axis("off")

        # Label rows and adjust aspect
        ax[i][0].set_ylabel(name, rotation=90, labelpad=10, va="center")
        ax[i][4].set_aspect(1 / 1.2)

    # Column titles
    ax[0][0].set_title("Raw Image")
    ax[0][1].set_title("Flux - UTIS")
    ax[0][2].set_title("Flux - Ideal")
    ax[0][3].set_title("Flux - Deflectometry")
    ax[0][4].set_title("Surface - Deflectometry")

    fig.tight_layout()
    out_file = f"{plot_save_path}/02a_flux_comparison.pdf"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overexposed flux comparison to {out_file}")

    plt.close(fig)

plt.rcParams["font.family"] = "sans-serif"

config = load_config()
paint_plots_base_path = Path(config["base_path"])
results_path = join_safe(paint_plots_base_path, config["results_raytracing_dict_path"])
save_plot_path = join_safe(paint_plots_base_path, config["results_plot_path"])

plot_results_flux_comparision(
    result_file=str(results_path), plot_save_path=str(save_plot_path)
)