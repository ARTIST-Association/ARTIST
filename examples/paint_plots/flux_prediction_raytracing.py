import argparse
import pathlib
import warnings
from typing import Dict, cast

import h5py
import numpy as np
import paint.util.paint_mappings as paint_mappings
import torch
import yaml
from PIL import Image

from artist.core import HeliostatRayTracer
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.data_parser.paint_scenario_parser import extract_paint_heliostat_properties
from artist.scenario import Scenario
from artist.util import set_logger_config
from artist.util.environment_setup import get_device

set_logger_config()


def perform_inverse_canting_and_translation(
    canted_points: torch.Tensor,
    translation: torch.Tensor,
    canting: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Invert the canting rotation and translation on a batch of facets.

    Parameters
    ----------
    canted_points : torch.Tensor
        Homogeneous points after the forward transform.
        Tensor of shape [number_of_facets, number_of_points, 4].
    translation : torch.Tensor
        Batch of facet translations.
        Tensor of shape [number_of_facets, 4].
    canting : torch.Tensor
        Batch of canting vectors (east, north).
        Tensor of shape [number_of_facets, 2, 4].
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        Original 3D points.
        Tensor of shape [number_of_facets, number_of_points, 3].
    """
    device = get_device(device=device)
    number_of_facets, number_of_points, _ = canted_points.shape

    # Build forward transform per facet (use only ENU 3D coordinates for rotation).
    forward_transform = torch.zeros((number_of_facets, 4, 4), device=device)

    east_unit_vector = torch.nn.functional.normalize(canting[:, 0, :3], dim=1)
    north_unit_vector = torch.nn.functional.normalize(canting[:, 1, :3], dim=1)
    up_unit_vector = torch.nn.functional.normalize(
        torch.linalg.cross(east_unit_vector, north_unit_vector, dim=1), dim=1
    )

    forward_transform[:, :3, 0] = east_unit_vector
    forward_transform[:, :3, 1] = north_unit_vector
    forward_transform[:, :3, 2] = up_unit_vector
    forward_transform[:, :3, 3] = translation[:, :3]
    forward_transform[:, 3, 3] = 1.0

    # Extract rotation and translation.
    rotation_matrix = forward_transform[:, :3, :3]
    translation_vector = forward_transform[:, :3, 3]

    # Compute inverse transform.
    rotation_matrix_inverse = rotation_matrix.transpose(1, 2)
    translation_inverse = -torch.bmm(
        rotation_matrix_inverse, translation_vector.unsqueeze(-1)
    ).squeeze(-1)

    inverse_transform = torch.zeros((number_of_facets, 4, 4), device=device)
    inverse_transform[:, :3, :3] = rotation_matrix_inverse
    inverse_transform[:, :3, 3] = translation_inverse
    inverse_transform[:, 3, 3] = 1.0

    # Apply inverse transform.
    restored_points = torch.bmm(canted_points, inverse_transform.transpose(1, 2))
    return restored_points[..., :3]


def extract_canting_and_translation_from_properties(
    heliostat_list: list[tuple[str, pathlib.Path]],
    convert_to_4d: bool = False,
    device: torch.device | None = None,
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """
    Extract facet translation and canting vectors per heliostat from ``PAINT`` properties files.

    Parameters
    ----------
    heliostat_list : list[tuple[str, pathlib.Path]]
        A list where each entry is a tuple containing the heliostat name and the path to the heliostat properties data.
    convert_to_4d : bool
        Indicating whether tensors should be converted to 4D format (default is False).
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    list[tuple[str, torch.Tensor, torch.Tensor]]
        A list containing a tuple for each heliostat including the heliostat name, the facet translations tensor of shape
        [number_of_facets, dimension] and the facet canting tensor of shape [number of facets, 2, dimension], where
        dimension is three or four depending which conversion is applied via the convert_to_4d parameter.
    """
    device = get_device(device=device)
    facet_transforms_per_heliostat: list[tuple[str, torch.Tensor, torch.Tensor]] = []

    for heliostat_name, properties_path in heliostat_list:
        try:
            (
                _,
                facet_translation_vectors,
                canting,
                _,
                _,
                _,
            ) = extract_paint_heliostat_properties(
                heliostat_properties_path=properties_path,
                power_plant_position=torch.tensor(
                    [
                        paint_mappings.POWER_PLANT_LAT,
                        paint_mappings.POWER_PLANT_LON,
                        paint_mappings.POWER_PLANT_ALT,
                    ]
                ),
                device=device,
            )

            if not convert_to_4d:
                facet_translation_vectors = facet_translation_vectors[:, :3]
                canting = canting[..., :3]

            facet_transforms_per_heliostat.append(
                (heliostat_name, facet_translation_vectors, canting)
            )

        except Exception as ex:
            warnings.warn(
                f"Failed to extract canting/translation for '{heliostat_name}' "
                f"from properties '{properties_path}': {ex}."
            )
            continue

    return facet_transforms_per_heliostat


def load_image_as_tensor(
    name: str,
    data_directory: pathlib.Path,
    measurement_id: int,
    image_key: str,
) -> torch.Tensor:
    """
    Load a flux PNG as a grayscale image and return it as a normalized tensor.

    Parameters
    ----------
    name : str
        Heliostat name.
    data_directory : Path
        Path to the data directory.
    measurement_id : int
        Measurement identifier.
    image_key : str
        Image key suffix (e.g., "cropped", "flux").

    Returns
    -------
    torch.Tensor
        Grayscale image tensor with shape (H, W), normalized to [0, 1].

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.
    """
    # Build the path.
    image_path = pathlib.Path(
        f"{data_directory}/{name}/{paint_mappings.SAVE_CALIBRATION}/{measurement_id}-{image_key}.png"
    )
    if not image_path.exists():
        raise FileNotFoundError(
            f"The image type {image_key} for heliostat {name} does not exist at {image_path}."
        )

    # Open image in grayscale ('L' mode).
    image = Image.open(image_path).convert("L")

    # Convert to tensor and normalize to [0, 1].
    tensor = torch.from_numpy(np.array(image)).float() / 255.0
    return tensor


def align_and_trace_rays(
    scenario: Scenario,
    aim_points: torch.Tensor,
    light_direction: torch.Tensor,
    active_heliostats_mask: torch.Tensor,
    target_area_mask: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Align heliostats and perform heliostat ray tracing.

    Parameters
    ----------
    scenario : Scenario
        Loaded scenario object.
    aim_points : torch.Tensor
        Aim points on the receiver for each heliostat.
    light_direction : torch.Tensor
        Incoming light directions per heliostat.
    active_heliostats_mask : torch.Tensor
        Mask indicating which heliostats are active.
    target_area_mask : torch.Tensor
        Target area indices for each active heliostat.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        Flux density image(s) on the receiver.
    """
    device = get_device(device=device)

    # Activate heliostats.
    scenario.heliostat_field.heliostat_groups[0].activate_heliostats(
        active_heliostats_mask=active_heliostats_mask,
        device=device,
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
        scenario=scenario,
        heliostat_group=scenario.heliostat_field.heliostat_groups[0],
    )

    # Perform heliostat-based ray tracing.
    return ray_tracer.trace_rays(
        incident_ray_directions=light_direction,
        active_heliostats_mask=active_heliostats_mask,
        target_area_mask=target_area_mask,
        device=device,
    )


def generate_flux_images(
    scenario_path: pathlib.Path,
    heliostats: dict[str, int],
    data_directory: pathlib.Path,
    results_file: pathlib.Path,
    result_key: str,
    device: torch.device | None,
) -> None:
    """
    Perform ray tracing in ``ARTIST`` and save the bitmaps to a results file.

    This function generates flux bitmaps through ray tracing in ``ARTIST`` and saves the results, i.e., the bitmaps to a
    unified results file for later plot generation.

    Parameters
    ----------
    scenario_path : pathlib.Path
        Path to the scenario being used.
    heliostats : dict[str, int]
        Mapping containing the heliostats present in the scenario and the calibration measurement to be used.
    data_directory : pathlib.Path
        Path to the data directory.
    result_file : pathlib.Path
        Path to the unified results file, saved as a torch checkpoint.
    result_key : str
        Key under which to store the result.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.
    """
    device = get_device(device)

    results_dict: Dict[str, Dict[str, np.ndarray | torch.Tensor]] = {}

    try:
        loaded = torch.load(results_file, weights_only=False)
        results_dict = cast(Dict[str, Dict[str, np.ndarray | torch.Tensor]], loaded)
    except FileNotFoundError:
        print(f"File not found: {results_file}. Initializing with an empty dictionary.")
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")

    # Load scenario.
    with h5py.File(str(scenario_path), mode="r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(scenario_file, device=device)

    # Hint: Lower the number of rays when running on cuda, to avoid raising "cuda out of memory errors".
    scenario.set_number_of_rays(number_of_rays=1000)

    heliostat_properties_tuples: list[tuple[str, pathlib.Path]] = [
        (
            heliostat_name,
            pathlib.Path(
                f"{data_directory}/{heliostat_name}/{paint_mappings.SAVE_PROPERTIES}/{heliostat_name}-{paint_mappings.HELIOSTAT_PROPERTIES_KEY}.json"
            ),
        )
        for heliostat_name in list(heliostats.keys())
    ]

    # Extract facet translations and canting vectors.
    facet_transforms = extract_canting_and_translation_from_properties(
        heliostat_list=heliostat_properties_tuples,
        convert_to_4d=True,
        device=device,
    )
    facet_transforms_by_name = {
        heliostat_name: (facet_translations, facet_canting_vectors)
        for heliostat_name, facet_translations, facet_canting_vectors in facet_transforms
    }

    # Build properties' list for calibration extraction.
    heliostat_data_mapping: list[tuple[str, list[pathlib.Path]]] = []
    for heliostat_name in heliostats.keys():
        heliostat_data_mapping.append(
            (
                heliostat_name,
                [
                    pathlib.Path(
                        f"{data_directory}/{heliostat_name}/{paint_mappings.SAVE_CALIBRATION}/{paint_mappings.CALIBRATION_PROPERTIES_NAME % heliostats[heliostat_name]}.json"
                    )
                ],
            )
        )

    # Load the calibration data.
    calibration_data_parser = PaintCalibrationDataParser()
    (
        focal_spots_calibration,
        incident_ray_directions_calibration,
        _,
        heliostats_mask_calibration,
        target_area_mask_calibration,
    ) = calibration_data_parser._parse_calibration_data(
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

    # Load UTIS image and update results.
    for i, heliostat_name in enumerate(list(heliostats.keys())):
        if heliostat_name not in results_dict:
            results_dict[heliostat_name] = {}

        if "utis" not in results_dict[heliostat_name]:
            utis_image = load_image_as_tensor(
                heliostat_name,
                data_directory,
                heliostats[heliostat_name],
                "flux",
            )
            results_dict[heliostat_name]["utis"] = utis_image.cpu().detach().numpy()

        # Store flux for the given scenario under the provided key.
        results_dict[heliostat_name][result_key] = flux[i].cpu().detach().numpy()

        # Save the surface normals per facet.
        if (
            result_key == "deflectometry"
            and "surface" not in results_dict[heliostat_name]
        ):
            # Surface points and normals per facet (canted).
            facet_points_canted = (
                scenario.heliostat_field.heliostat_groups[0]
                .surface_points[i]
                .view(4, 2500, 4)
            )
            facet_normals_canted = (
                scenario.heliostat_field.heliostat_groups[0]
                .surface_normals[i]
                .view(4, 2500, 4)
            )

            # Apply inverse canting and translation.
            facet_translations, facet_canting_vectors = facet_transforms_by_name[
                heliostat_name
            ]
            facet_points_decanted_tensor = perform_inverse_canting_and_translation(
                canted_points=facet_points_canted,
                translation=torch.zeros(4, 4),
                canting=facet_canting_vectors,
                device=device,
            )
            facet_normals_decanted_tensor = perform_inverse_canting_and_translation(
                canted_points=facet_normals_canted,
                translation=torch.zeros(4, 4),
                canting=facet_canting_vectors,
                device=device,
            )
            reference_direction = torch.tensor([0.0, 0.0, 1.0], device=device)

            facet_points_flat = facet_points_decanted_tensor.reshape(-1, 3)
            facet_normals_flat = facet_normals_decanted_tensor.reshape(-1, 3)

            x = facet_points_flat[:, 0]
            y = facet_points_flat[:, 1]
            cos_theta = facet_normals_flat @ reference_direction
            angles = torch.arccos(torch.clip(cos_theta, -1.0, 1.0))
            angles = torch.clip(angles, -0.1, 0.1)

            results_dict[heliostat_name]["surface"] = [x, y, angles]

    if not results_file.parent.exists():
        results_file.parent.mkdir(parents=True, exist_ok=True)

    torch.save(results_dict, results_file)

    print(f"Flux prediction results saved as {results_file}.")


if __name__ == "__main__":
    """
    Perform raytracing and save the results.

    This script executes the raytracing in ``ARTIST`` for the two previously generated scenarios. The resulting bitmaps
    representing flux images are saved for plotting later.

    Parameters
    ----------
    config : str
        Path to the configuration file.
    device : str
        Device to use for the computation.
    data_dir : str
        Path to the data directory.
    heliostats : dict[str, int]
        Heliostats and calibration measurements required for raytracing.
    results_dir : str
        Path to where the results will be saved.
    scenarios_dir : str
        Path to the directory containing the scenarios.
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
            warnings.warn(f"Error parsing YAML file: {exc}")
    else:
        warnings.warn(
            f"Warning: Configuration file not found at {config_path}. Using defaults."
        )

    # Add remaining arguments to the parser with defaults loaded from the config.
    data_dir_default = config.get("data_dir", "./paint_data")
    device_default = config.get("device", "cuda")
    heliostats_default = config.get(
        "heliostats_for_raytracing", {"AA39": 149576, "AY26": 247613, "BC34": 82084}
    )
    scenarios_dir_default = config.get("scenarios_dir", "./scenarios")
    results_dir_default = config.get("results_dir", "./results")

    parser.add_argument(
        "--device",
        type=str,
        help="Device to use.",
        default=device_default,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to downloaded paint data.",
        default=data_dir_default,
    )
    parser.add_argument(
        "--heliostats",
        type=str,
        help="Heliostats and calibration measurements required for the raytracing.",
        nargs="+",
        default=heliostats_default,
    )
    parser.add_argument(
        "--scenarios_dir",
        type=str,
        help="Path to directory containing the generated scenarios.",
        default=scenarios_dir_default,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Path to save the results.",
        default=results_dir_default,
    )

    # Re-parse the full set of arguments.
    args = parser.parse_args(args=unknown)

    device = get_device(torch.device(args.device))
    data_dir = pathlib.Path(args.data_dir)
    results_path = pathlib.Path(args.results_dir) / "flux_prediction_results.pt"

    deflectometry_scenario_file = (
        pathlib.Path(args.scenarios_dir) / "flux_prediction_deflectometry.h5"
    )
    ideal_scenario_file = pathlib.Path(args.scenarios_dir) / "flux_prediction_ideal.h5"

    # Generate and merge flux images for both scenarios into one results file.
    generate_flux_images(
        scenario_path=deflectometry_scenario_file,
        heliostats=args.heliostats,
        data_directory=data_dir,
        results_file=results_path,
        result_key="deflectometry",
        device=device,
    )
    generate_flux_images(
        scenario_path=ideal_scenario_file,
        heliostats=args.heliostats,
        data_directory=data_dir,
        results_file=results_path,
        result_key="ideal",
        device=device,
    )
