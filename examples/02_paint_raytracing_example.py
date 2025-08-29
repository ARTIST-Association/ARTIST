import json
import os
import pathlib
import sys
from pathlib import Path
from typing import Any, Dict

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.data_loader import paint_loader
from artist.data_loader.paint_loader import (
    extract_canting_and_translation_from_properties,
)
from artist.scenario.configuration_classes import (
    LightSourceConfig,
    LightSourceListConfig,
)
from artist.scenario.h5_scenario_generator import H5ScenarioGenerator
from artist.scenario.scenario import Scenario
from artist.scenario.surface_generator import SurfaceGenerator
from artist.util import config_dictionary, set_logger_config
from artist.util.environment_setup import get_device

sys.path.append(str(Path(__file__).resolve().parent))


MEASUREMENT_IDS = {"AA39": 149576, "AY26": 247613, "BC34": 82084}


def find_latest_deflectometry_file(name: str, paint_dir: str | Path) -> Path:
    """Find the latest deflectometry HDF5 file for a given heliostat.

    Parameters
    ----------
    name : str
        Heliostat name (e.g., "AA39").
    paint_dir : str | Path
        Base ''PAINT'' directory.

    Returns
    -------
    Path
        Path to the latest deflectometry file.

    Raises
    ------
    FileNotFoundError
        If no matching file is found.
    """
    search_path = pathlib.Path(paint_dir) / name / "Deflectometry"
    pattern = f"{name}-filled*.h5"
    files = sorted(search_path.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No deflectometry file found for {name} in {search_path}"
        )
    return files[-1]


def generate_paint_scenario(
    paint_dir: str | Path,
    scenario_path: str | Path,
    tower_file: str | Path,
    heliostat_names: list[str],
    device: torch.device | str = "cpu",
    use_deflectometry: bool = True,
) -> None:
    """Generate an HDF5 scenario from ''PAINT'' inputs.

    Parameters
    ----------
    paint_dir : str | Path
        Base ''PAINT'' repository path.
    scenario_path : str | Path
        Output scenario path without extension ('.h5' will be added).
    tower_file : str | Path
        Path to the tower measurements HDF5.
    heliostat_names : list[str]
        Heliostat identifiers to include.
    device : torch.device | str, optional
        Torch device for processing, by default "cpu".
    use_deflectometry : bool, optional
        If True, include deflectometry data for surface fitting, by default True.
    """
    # Normalize to Path for mypy correctness
    scenario_path = pathlib.Path(scenario_path)
    tower_file_path = pathlib.Path(tower_file)

    if not scenario_path.parent.is_dir():
        raise FileNotFoundError(
            f"The folder ``{scenario_path.parent}`` selected to save the scenario does not exist. "
            "Please create the folder or adjust the file path before running again!"
        )

    # Prepare heliostat files.
    # The function is a hacky workaround to catch a MyPy error.

    def create_heliostat_files_list(
        names, paint_dir, config_dict, use_deflectometry=False
    ):
        if use_deflectometry:
            return [
                (
                    name,
                    pathlib.Path(
                        f"{paint_dir}/{name}/{config_dict.paint_properties_folder_name}/{name}{config_dict.paint_properties_file_name_ending}"
                    ),
                    find_latest_deflectometry_file(name, paint_dir),
                )
                for name in names
            ]
        else:
            return [
                (
                    name,
                    pathlib.Path(
                        f"{paint_dir}/{name}/{config_dict.paint_properties_folder_name}/{name}{config_dict.paint_properties_file_name_ending}"
                    ),
                )
                for name in names
            ]

    heliostat_files_list = create_heliostat_files_list(
        heliostat_names, paint_dir, config_dictionary, use_deflectometry
    )
    paths_arg = heliostat_files_list

    # Include the power plant configuration.
    power_plant_config, target_area_list_config = (
        paint_loader.extract_paint_tower_measurements(
            tower_measurements_path=tower_file_path,  # pass Path
            device=device,
        )
    )

    # Include the light source configuration.
    light_source1_config = LightSourceConfig(
        light_source_key="sun_1",
        light_source_type=config_dictionary.sun_key,
        number_of_rays=10,
        distribution_type=config_dictionary.light_source_distribution_is_normal,
        mean=0.0,
        covariance=4.3681e-06,
    )

    # Create a list of light source configs - in this case only one.
    light_source_list = [light_source1_config]

    # Include the configuration for the list of light sources.
    light_source_list_config = LightSourceListConfig(
        light_source_list=light_source_list
    )

    number_of_nurbs_control_points = torch.tensor([20, 20], device=device)
    nurbs_fit_method = config_dictionary.fit_nurbs_from_normals
    nurbs_deflectometry_step_size = 100
    nurbs_fit_tolerance = 1e-10
    nurbs_fit_max_epoch = 400

    # Please leave the optimizable parameters empty, they will automatically be added for the surface fit.
    nurbs_fit_optimizer = torch.optim.Adam(
        [torch.empty(1, requires_grad=True)], lr=1e-3
    )
    nurbs_fit_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        nurbs_fit_optimizer,
        mode="min",
        factor=0.2,
        patience=50,
        threshold=1e-7,
        threshold_mode="abs",
    )

    heliostat_list_config, prototype_config = paint_loader.extract_paint_heliostats(
        paths=paths_arg,
        power_plant_position=power_plant_config.power_plant_position,
        number_of_nurbs_control_points=number_of_nurbs_control_points,
        deflectometry_step_size=nurbs_deflectometry_step_size,
        nurbs_fit_method=nurbs_fit_method,
        nurbs_fit_tolerance=nurbs_fit_tolerance,
        nurbs_fit_max_epoch=nurbs_fit_max_epoch,
        nurbs_fit_optimizer=nurbs_fit_optimizer,
        nurbs_fit_scheduler=nurbs_fit_scheduler,
        device=device,
    )

    # Generate the scenario given the defined parameters.
    scenario_generator = H5ScenarioGenerator(
        file_path=scenario_path,  # pass Path
        power_plant_config=power_plant_config,
        target_area_list_config=target_area_list_config,
        light_source_list_config=light_source_list_config,
        prototype_config=prototype_config,
        heliostat_list_config=heliostat_list_config,
    )
    scenario_generator.generate_scenario()


def load_image_as_tensor(
    name: str,
    paint_dir: str | Path,
    measurement_id: int,
    image_key: str,
) -> torch.Tensor:
    """Load a flux PNG as grayscale and return a float tensor in [0, 1].

    Parameters
    ----------
    name : str
        Heliostat name.
    paint_dir : str | Path
        Base ''PAINT'' directory.
    measurement_id : int
        Measurement identifier.
    image_key : str
        Image key suffix (e.g., "cropped", "flux").

    Returns
    -------
    torch.Tensor
        Grayscale image tensor with shape (H, W), dtype float32 in [0, 1].

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.
    """
    # Build the path.
    image_path = pathlib.Path(
        f"{paint_dir}/{name}/{config_dictionary.paint_calibration_folder_name}/{measurement_id}-{image_key}.png"
    )

    # Open image in grayscale ('L' mode).
    image = Image.open(image_path).convert("L")

    # Convert to tensor and normalize to [0, 1] float.
    tensor = torch.from_numpy(np.array(image)).float() / 255.0
    return tensor


def calculate_flux_deviation(
    flux_tensor_one: torch.Tensor,
    flux_tensor_two: torch.Tensor,
    normalize_by: str = "f1",
) -> torch.Tensor:
    """Calculate a relative flux deviation between two flux maps.

    The deviation is computed as the mean absolute difference after normalizing both
    images by the mean intensity of the selected reference (normalize_by).

    Parameters
    ----------
    flux_tensor_one : torch.Tensor
        First flux tensor (reference), shape (..., H, W).
    flux_tensor_two : torch.Tensor
        Second flux tensor (comparison), shape (..., H, W).
    normalize_by : {"f1", "f2"}, optional
        Select whether to normalize intensities by the mean of the first ("f1")
        or the second ("f2") tensor, by default "f1".

    Returns
    -------
    torch.Tensor
        Deviation per image (broadcast over leading batch dimensions), shape compatible
        with the batch dimensions of the inputs.

    Raises
    ------
    ValueError
        If input shapes do not match or normalize_by has an invalid value.
    """
    if flux_tensor_one.shape != flux_tensor_two.shape:
        raise ValueError(
            f"Shape mismatch: flux_tensor_one {flux_tensor_one.shape} vs flux_tensor_two {flux_tensor_two.shape}"
        )

    # Keep device/dtype and support batched images (..., H, W).
    eps = (
        torch.finfo(flux_tensor_one.dtype).eps
        if flux_tensor_one.is_floating_point()
        else 1e-8
    )
    # Reduce over last two dims (H, W).
    reduce_dims = tuple(range(flux_tensor_one.ndim - 2, flux_tensor_one.ndim))
    m1 = flux_tensor_one.mean(dim=reduce_dims, keepdim=True).clamp_min(eps)
    m2 = flux_tensor_two.mean(dim=reduce_dims, keepdim=True).clamp_min(eps)

    # Scale both to comparable range; honor normalize_by.
    if normalize_by == "f1":
        f1n = flux_tensor_one / m1
        f2n = flux_tensor_two / m1
    elif normalize_by == "f2":
        f1n = flux_tensor_one / m2
        f2n = flux_tensor_two / m2
    else:
        raise ValueError("normalize_by must be 'f1' or 'f2'.")

    abs_diff = (f1n - f2n).abs()
    sum_abs_diff = abs_diff.sum(dim=reduce_dims)
    # Normalization factor = mean of chosen reference (per-image), multiplied by pixel count.
    ref_mean = 1.0  # Since we normalized by the chosen mean above.
    num_px = flux_tensor_one.shape[-1] * flux_tensor_one.shape[-2]
    deviation = sum_abs_diff / (ref_mean * num_px)
    return deviation


def align_and_trace_rays(
    scenario: Scenario,
    aim_points: torch.Tensor,
    light_direction: torch.Tensor,
    active_heliostats_mask: torch.Tensor,
    target_area_mask: torch.Tensor,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Align heliostats and perform heliostat ray tracing.

    Parameters
    ----------
    scenario : Scenario
        Loaded scenario object.
    aim_points : torch.Tensor
        Aim points on the receiver for each heliostat; shape compatible with the
        scenario's alignment routine.
    light_direction : torch.Tensor
        Incoming light directions per heliostat; shape compatible with the scenario.
    active_heliostats_mask : torch.Tensor
        Mask indicating which heliostats are active.
    target_area_mask : torch.Tensor
        Target area indices for each active heliostat.
    device : torch.device | str, optional
        Device to use for computations, by default "cuda".

    Returns
    -------
    torch.Tensor
        Flux density image(s) on the receiver.
    """
    # Activate heliostats.
    scenario.heliostat_field.heliostat_groups[0].activate_heliostats(
        active_heliostats_mask=active_heliostats_mask
    )

    # Align all heliostats.
    scenario.heliostat_field.heliostat_groups[
        0
    ].align_surfaces_with_incident_ray_directions(
        aim_points=aim_points,
        incident_ray_directions=light_direction,
        active_heliostats_mask=active_heliostats_mask,
        device=device,
    )

    # Create a ray tracer.
    ray_tracer = HeliostatRayTracer(
        scenario=scenario, heliostat_group=scenario.heliostat_field.heliostat_groups[0]
    )

    # Perform heliostat-based ray tracing.
    return ray_tracer.trace_rays(
        incident_ray_directions=light_direction,
        active_heliostats_mask=active_heliostats_mask,
        target_area_mask=target_area_mask,
        device=device,
    )


def generate_flux_images(
    scenario_path: str | Path,
    heliostats: list[str],
    paint_dir: str | Path,
    result_file: str | Path,
    device: torch.device | str,
    result_key: str = "flux_deflectometry",
) -> None:
    """Generate flux images via alignment and ray tracing and save to a merged result file.

    Parameters
    ----------
    scenario_path : str | Path
        Base path (without .h5) of the scenario to load.
    heliostats : list[str]
        Heliostat names to process, order must match extracted calibration data.
    paint_dir : str | Path
        ''PAINT'' repository path.
    result_file : str | Path
        Path to a single merged .pt file to store results for all runs.
    device : torch.device | str
        Device to run computations on.
    result_key : str, optional
        Key under which to store the result (e.g., "flux_deflectometry", "flux_ideal"),
        by default "flux_deflectometry".
    """
    results_dict: Dict[str, Dict[str, np.ndarray]] = {}
    results_path = pathlib.Path(result_file)
    if results_path.exists():
        results_dict = torch.load(results_path, weights_only=False)

    scenario_h5_path = pathlib.Path(scenario_path).with_suffix(".h5")

    # Load scenario.
    with h5py.File(scenario_h5_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(scenario_file, device=device)

    scenario.light_sources.light_source_list[0].number_of_rays = 6000
    heliostat_properties_tuples: list[tuple[str, Path] | tuple[str, Path, Path]] = [
        (
            name,
            pathlib.Path(
                f"{paint_dir}/{name}/{config_dictionary.paint_properties_folder_name}/{name}{config_dictionary.paint_properties_file_name_ending}"
            ),
        )
        for name in heliostats
    ]

    # Extract facet translations and canting vectors (homogeneous 4D for downstream APIs).
    facet_transforms = extract_canting_and_translation_from_properties(
        heliostat_list=heliostat_properties_tuples,
        convert_to_4d=True,
        device=device,
    )
    facet_transforms_by_name = {
        heliostat_name: (facet_translations, facet_canting_vectors)
        for heliostat_name, facet_translations, facet_canting_vectors in facet_transforms
    }

    # Build properties list for calibration extraction.
    heliostat_data_mapping: list[tuple[str, list[Path]]] = []
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

    # Perform alignment and ray tracing to generate flux density images.
    flux = align_and_trace_rays(
        scenario=scenario,
        aim_points=focal_spots_calibration,
        light_direction=incident_ray_directions_calibration,
        active_heliostats_mask=heliostats_mask_calibration,
        target_area_mask=target_area_mask_calibration,
        device=device,
    )

    # Load raw and UTIS image, update results.
    for i, name in enumerate(heliostats):
        if name not in results_dict:
            results_dict[name] = {}

        # Populate once if not present.
        if "image_raw" not in results_dict[name]:
            raw_image = load_image_as_tensor(
                name, str(paint_dir), MEASUREMENT_IDS[name], "cropped"
            )
            results_dict[name]["image_raw"] = raw_image.cpu().detach().numpy()

        if "utis_image" not in results_dict[name]:
            utis_image = load_image_as_tensor(
                name, str(paint_dir), MEASUREMENT_IDS[name], "flux"
            )
            results_dict[name]["utis_image"] = utis_image.cpu().detach().numpy()

        # Store flux for the given scenario under the provided key.
        results_dict[name][result_key] = flux[i].cpu().detach().numpy()

        # For the deflectometry scenario, keep the surface plot content as before.
        if result_key == "flux_deflectometry" and "surface" not in results_dict[name]:
            # Surface points per facet (canted).
            facet_points_canted = (
                scenario.heliostat_field.heliostat_groups[0]
                .surface_points[i]
                .view(4, 2500, 4)
            )

            # Apply inverse canting and translation using properties-derived transforms.
            facet_translations, facet_canting_vectors = facet_transforms_by_name[name]
            facet_points_decanted = (
                SurfaceGenerator.perform_inverse_canting_and_translation(
                    canted_points=facet_points_canted,
                    translation=facet_translations,
                    canting=facet_canting_vectors,
                    device=device,
                )
            )

            # Convert to Z grid for plotting.
            facet_points_decanted = (
                facet_points_decanted[:, :, 2].view(4, 50, 50).cpu().detach().numpy()
            )
            surface_z_grid = np.block(
                [
                    [facet_points_decanted[2], facet_points_decanted[0]],
                    [facet_points_decanted[3], facet_points_decanted[1]],
                ]
            )
            results_dict[name]["surface"] = surface_z_grid

    torch.save(results_dict, results_path)


def plot_results_flux_comparision(
    result_file: str | Path, plot_save_path: str | Path
) -> None:
    """Plot and save a multi-row comparison of raw, UTIS, deflectometry, and surface images.

    For each heliostat contained in the results file, this function creates a row with:
    - Raw camera image
    - UTIS flux image
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
        If expected keys are missing for a heliostat entry (e.g., "image_raw", "utis_image", "flux_deflectometry").
    """
    # Load results
    if not pathlib.Path(result_file).exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")
    if not pathlib.Path(plot_save_path).exists():
        raise FileNotFoundError(f"Output directory does not exist: {plot_save_path}")

    results_dict: Dict[str, Dict[str, np.ndarray]] = torch.load(
        result_file, weights_only=False
    )
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
    colormaps: list[str] = ["gray", "hot", "hot", "jet"]

    for i, (name, data) in enumerate(results_dict.items()):
        # Extract images
        raw = data["image_raw"]
        utis = data["utis_image"]
        flux_def = data["flux_deflectometry"]
        surface = data.get("surface", np.zeros_like(raw))

        # Normalize intensities so each image has max=1
        def normalize(img: np.ndarray) -> np.ndarray:
            mx = float(img.max()) if img.size > 0 else 0.0
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


def simulate_overexposure(
    image: np.ndarray,
    exposure: float = 5.0,
    i_max: float = 1.0,
    thresh: float = 0.9,
    blur_ksize: int = 31,
    blur_sigma: float = 10,
) -> np.ndarray:
    """Simulate sensor overexposure and bloom effect on a linear image.

    Parameters
    ----------
    image : np.ndarray
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
    # Scale and hard clip.
    i_exp = image * exposure
    i_clip = np.clip(i_exp, 0, i_max)

    # Extract highlights for bloom.
    highlights = np.clip(i_exp - thresh, 0, None)

    # Apply Gaussian blur to highlights.
    bloom = cv2.GaussianBlur(highlights, (blur_ksize, blur_ksize), blur_sigma)

    # Composite clipped image + bloom, then clamp.
    return np.clip(i_clip + bloom, 0, i_max)


def load_config() -> Dict[str, Any]:
    """Load local example configuration from config.local.json.

    Returns
    -------
    dict[str, Any]
        Configuration dictionary read from config.local.json.

    Raises
    ------
    FileNotFoundError
        If no config.local.json file can be found next to this script.
    """
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


def main() -> None:
    """Run ray tracing example: generate flux images and plots from configuration.

    Loads configuration, ensures scenarios exist (deflectometry and ideal),
    generates/merges flux maps, and creates comparison plots.
    """
    config = load_config()

    # Map config -> args using new keys and base paths.
    paint_base = pathlib.Path(config["paint_repository_base_path"])
    examples_base = pathlib.Path(config["examples_base_path"])

    def join_safe(
        base: pathlib.Path, relative_path: str | pathlib.Path
    ) -> pathlib.Path:
        """Join base and a possibly absolute path by stripping leading separators."""
        relative_str = str(relative_path)
        return base / relative_str.lstrip("/\\")

    paint_dir = paint_base
    tower_file = join_safe(paint_base, config["paint_tower_file"])
    heliostats = config["heliostats"]
    scenario_base = join_safe(
        examples_base, config["examples_raytracing_scenario_path"]
    )
    result_file = join_safe(examples_base, config["examples_results_path"])
    save_plot_path = join_safe(examples_base, config["examples_save_plot_path"])

    device = get_device()
    set_logger_config()
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    plt.rcParams["font.family"] = "sans-serif"

    # Generate two scenarios: deflectometry and ideal (no deflectometry).
    scenario_deflec_base = pathlib.Path(str(scenario_base) + "_deflectometry")
    scenario_ideal_base = pathlib.Path(str(scenario_base) + "_ideal")

    for base, use_def in [(scenario_deflec_base, True), (scenario_ideal_base, False)]:
        h5_path = base.with_suffix(".h5")
        if h5_path.exists():
            print(
                f"Scenario found at {h5_path}... continue without generating scenario."
            )
        else:
            print(
                f"Scenario not found. Generating a new one at {h5_path} (use_deflectometry={use_def})..."
            )
            generate_paint_scenario(
                paint_dir=str(paint_dir),
                scenario_path=base,
                tower_file=str(tower_file),
                heliostat_names=heliostats,
                device=device,
                use_deflectometry=use_def,
            )

    # Generate and merge flux images for both scenarios into one results file.
    generate_flux_images(
        scenario_path=scenario_deflec_base,
        heliostats=heliostats,
        paint_dir=str(paint_dir),
        result_file=str(result_file),
        device=device,
        result_key="flux_deflectometry",
    )
    generate_flux_images(
        scenario_path=scenario_ideal_base,
        heliostats=heliostats,
        paint_dir=str(paint_dir),
        result_file=str(result_file),
        device=device,
        result_key="flux_ideal",
    )

    plot_results_flux_comparision(
        result_file=str(result_file), plot_save_path=str(save_plot_path)
    )


if __name__ == "__main__":
    main()
