import argparse
import json
import pathlib
import warnings
from typing import cast

import h5py
import numpy as np
import paint.util.paint_mappings as paint_mappings
import torch
import yaml

from artist.core import loss_functions
from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.core.regularizers import IdealSurfaceRegularizer
from artist.core.surface_reconstructor import SurfaceReconstructor
from artist.data_parser.calibration_data_parser import CalibrationDataParser
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.data_parser.paint_scenario_parser import extract_paint_heliostat_properties
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, set_logger_config, utils
from artist.util.environment_setup import get_device, setup_distributed_environment
from artist.util.nurbs import NURBSSurfaces

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set up logger.
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
    number_of_facets, _, _ = canted_points.shape

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


def reconstruct_and_create_flux_image(
    data_directory: pathlib.Path,
    scenario_path: pathlib.Path,
    heliostat_data_mapping: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    validation_heliostat_data_mapping: list[
        tuple[str, list[pathlib.Path], list[pathlib.Path]]
    ],
    reconstruction_parameters: dict[str, float | int],
    results_file: pathlib.Path,
    result_key: str,
    device: torch.device | None,
) -> None:
    """
    Reconstruct the heliostat surface with ``ARTIST`` and save the bitmaps and surface to a results file.

    Parameters
    ----------
    data_directory : pathlib.Path
        Path to the data directory.
    scenario_path : pathlib.Path
        Path to the scenario being used.
    heliostat_data_mapping : list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
        Data mapping for the reconstruction.
    validation_heliostat_data_mapping : list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
        Data mapping for the plot.
    reconstruction_parameters : dict[str, float | int]
        Parameters for the reconstruction.
    results_file : pathlib.Path
        Path to the unified results file, saved as a torch checkpoint.
    result_key : str
        Key under which to store the result.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.
    """
    device = get_device(device)

    results_dict: dict[str, dict[str, np.ndarray | torch.Tensor]] = {}

    try:
        loaded = torch.load(results_file, weights_only=False)
        results_dict = cast(dict[str, torch.Tensor], loaded)
    except FileNotFoundError:
        print(f"File not found: {results_file}. Initializing with an empty dictionary.")
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")

    number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
        scenario_path=scenario_path
    )
    with setup_distributed_environment(
        number_of_heliostat_groups=number_of_heliostat_groups,
        device=device,
    ) as ddp_setup:
        device = ddp_setup[config_dictionary.device]

        number_of_surface_points_per_facet = torch.tensor(
            [
                reconstruction_parameters["number_of_surface_points"],
                reconstruction_parameters["number_of_surface_points"],
            ],
            device=device,
        )

        number_of_control_points_per_facet = torch.tensor(
            [
                reconstruction_parameters["number_of_control_points"],
                reconstruction_parameters["number_of_control_points"],
            ],
            device=device,
        )

        with h5py.File(scenario_path, "r") as scenario_file:
            scenario = Scenario.load_scenario_from_hdf5(
                scenario_file=scenario_file,
                number_of_surface_points_per_facet=number_of_surface_points_per_facet,
                change_number_of_control_points_per_facet=number_of_control_points_per_facet,
                device=device,
            )

        scenario.set_number_of_rays(
            number_of_rays=int(reconstruction_parameters["number_of_rays"])
        )

        for heliostat_group in scenario.heliostat_field.heliostat_groups:
            heliostat_group.nurbs_degrees = torch.tensor(
                [
                    reconstruction_parameters["nurbs_degree"],
                    reconstruction_parameters["nurbs_degree"],
                ],
                device=device,
            )

        data: dict[
            str,
            CalibrationDataParser
            | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
        ] = {
            config_dictionary.data_parser: PaintCalibrationDataParser(),
            config_dictionary.heliostat_data_mapping: heliostat_data_mapping,
        }

        heliostat_for_reconstruction_name = heliostat_data_mapping[0][0]
        heliostat_group_for_reconstruction = [
            group
            for group in scenario.heliostat_field.heliostat_groups
            if heliostat_for_reconstruction_name in group.names
        ][0]

        heliostat_properties_tuples: list[tuple[str, pathlib.Path]] = [
            (
                heliostat_for_reconstruction_name,
                pathlib.Path(
                    f"{data_directory}/{heliostat_for_reconstruction_name}/{paint_mappings.SAVE_PROPERTIES}/{heliostat_for_reconstruction_name}-{paint_mappings.HELIOSTAT_PROPERTIES_KEY}.json"
                ),
            )
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

        # Configure regularizers and their weights.
        ideal_surface_regularizer = IdealSurfaceRegularizer(
            weight=reconstruction_parameters["ideal_surface_loss_weight"],
            reduction_dimensions=(1, 2, 3),
        )

        regularizers = [
            ideal_surface_regularizer,
        ]

        scheduler = config_dictionary.reduce_on_plateau
        scheduler_parameters = {
            config_dictionary.min: 1e-7,
            config_dictionary.reduce_factor: reconstruction_parameters["reduce_factor"],
            config_dictionary.patience: reconstruction_parameters["patience"],
            config_dictionary.threshold: reconstruction_parameters["threshold"],
            config_dictionary.cooldown: 2,
        }

        optimization_configuration = {
            config_dictionary.initial_learning_rate: reconstruction_parameters[
                "initial_learning_rate"
            ],
            config_dictionary.tolerance: 0.00005,
            config_dictionary.max_epoch: 5000,
            config_dictionary.log_step: 0,
            config_dictionary.early_stopping_delta: 1e-4,
            config_dictionary.early_stopping_patience: 5000,
            config_dictionary.scheduler: scheduler,
            config_dictionary.scheduler_parameters: scheduler_parameters,
            config_dictionary.regularizers: regularizers,
        }

        surface_reconstructor = SurfaceReconstructor(
            ddp_setup=ddp_setup,
            scenario=scenario,
            data=data,
            optimization_configuration=optimization_configuration,
            number_of_surface_points=number_of_surface_points_per_facet,
            bitmap_resolution=torch.tensor([256, 256], device=device),
            device=device,
        )

        # Define loss.
        loss_definition = loss_functions.KLDivergenceLoss()

        _ = surface_reconstructor.reconstruct_surfaces(
            loss_definition=loss_definition,
            device=device,
        )

        evaluation_points = (
            utils.create_nurbs_evaluation_grid(
                number_of_evaluation_points=number_of_surface_points_per_facet,
                device=device,
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(
                1,
                heliostat_group_for_reconstruction.number_of_facets_per_heliostat,
                -1,
                -1,
            )
        )

        reconstructed_nurbs = NURBSSurfaces(
            degrees=heliostat_group_for_reconstruction.nurbs_degrees,
            control_points=heliostat_group_for_reconstruction.nurbs_control_points[
                0
            ].unsqueeze(0),
            device=device,
        )

        reconstructed_points, reconstructed_normals = (
            reconstructed_nurbs.calculate_surface_points_and_normals(
                evaluation_points=evaluation_points[0].unsqueeze(0),
                device=device,
            )
        )

        parser = PaintCalibrationDataParser(sample_limit=1)

        (
            validation_measured_flux_distributions,
            _,
            validation_incident_ray_directions,
            _,
            validation_active_heliostats_mask,
            validation_target_area_mask,
        ) = parser.parse_data_for_reconstruction(
            heliostat_data_mapping=validation_heliostat_data_mapping,
            heliostat_group=heliostat_group_for_reconstruction,
            scenario=scenario,
            device=device,
        )

        heliostat_group_for_reconstruction.activate_heliostats(
            active_heliostats_mask=validation_active_heliostats_mask,
            device=device,
        )

        heliostat_group_for_reconstruction.active_surface_points = (
            reconstructed_points.reshape(validation_active_heliostats_mask.sum(), -1, 4)
        )
        heliostat_group_for_reconstruction.active_surface_normals = (
            reconstructed_normals.reshape(
                validation_active_heliostats_mask.sum(), -1, 4
            )
        )

        heliostat_group_for_reconstruction.align_surfaces_with_incident_ray_directions(
            aim_points=scenario.target_areas.centers[validation_target_area_mask],
            incident_ray_directions=validation_incident_ray_directions,
            active_heliostats_mask=validation_active_heliostats_mask,
            device=device,
        )

        scenario.set_number_of_rays(number_of_rays=100)

        validation_ray_tracer = HeliostatRayTracer(
            scenario=scenario,
            heliostat_group=heliostat_group_for_reconstruction,
            world_size=ddp_setup["heliostat_group_world_size"],
            rank=ddp_setup["heliostat_group_rank"],
            batch_size=heliostat_group_for_reconstruction.number_of_active_heliostats,
            random_seed=ddp_setup["heliostat_group_rank"],
            bitmap_resolution=torch.tensor([256, 256], device=device),
        )

        validation_bitmaps_per_heliostat_reconstructed = (
            validation_ray_tracer.trace_rays(
                incident_ray_directions=validation_incident_ray_directions,
                active_heliostats_mask=validation_active_heliostats_mask,
                target_area_mask=validation_target_area_mask,
                device=device,
            )
        )

        kl_div_r = loss_definition(
            prediction=validation_bitmaps_per_heliostat_reconstructed,
            ground_truth=validation_measured_flux_distributions,
            target_area_mask=validation_target_area_mask,
            reduction_dimensions=(1, 2),
            device=device,
        )[0].item()

    # Apply inverse canting and translation.
    facet_translations, facet_canting_vectors = facet_transforms_by_name[
        heliostat_for_reconstruction_name
    ]
    reconstructed_normals_decanted = perform_inverse_canting_and_translation(
        canted_points=reconstructed_normals[0],
        translation=facet_translations,
        canting=facet_canting_vectors,
        device=device,
    )

    results = {
        "measured_flux": validation_measured_flux_distributions,
        "reconstructed_flux": validation_bitmaps_per_heliostat_reconstructed,
        "kl_div_reconstructed": kl_div_r,
        "points_reconstructed": reconstructed_points,
        "normals_reconstructed": reconstructed_normals_decanted.unsqueeze(0),
    }

    results_dict[result_key] = results

    if not results_file.parent.exists():
        results_file.parent.mkdir(parents=True, exist_ok=True)

    torch.save(results_dict, results_file)


def create_ideal_flux_image(
    scenario_path: pathlib.Path,
    validation_heliostat_data_mapping: list[
        tuple[str, list[pathlib.Path], list[pathlib.Path]]
    ],
    reconstruction_parameters: dict[str, float | int],
    results_file: pathlib.Path,
    result_key: str,
    device: torch.device | None,
) -> None:
    """
    Create the flux from the ideal heliostat surface.

    Parameters
    ----------
    scenario_path : pathlib.Path
        Path to the scenario being used.
    validation_heliostat_data_mapping : list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
        Data mapping for the plot.
    reconstruction_parameters : dict[str, float | int]
        Parameters for the reconstruction.
    results_file : pathlib.Path
        Path to the unified results file, saved as a torch checkpoint.
    result_key : str
        Key under which to store the result.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.
    """
    device = get_device(device)

    results_dict: dict[str, dict[str, np.ndarray | torch.Tensor]] = {}

    try:
        loaded = torch.load(results_file, weights_only=False)
        results_dict = cast(dict[str, torch.Tensor], loaded)
    except FileNotFoundError:
        print(f"File not found: {results_file}. Initializing with an empty dictionary.")
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")

    number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
        scenario_path=scenario_path
    )

    with setup_distributed_environment(
        number_of_heliostat_groups=number_of_heliostat_groups,
        device=device,
    ) as ddp_setup:
        device = ddp_setup[config_dictionary.device]

        number_of_surface_points_per_facet = torch.tensor(
            [
                reconstruction_parameters["number_of_surface_points"],
                reconstruction_parameters["number_of_surface_points"],
            ],
            device=device,
        )

        with h5py.File(scenario_path, "r") as scenario_file:
            scenario = Scenario.load_scenario_from_hdf5(
                scenario_file=scenario_file,
                number_of_surface_points_per_facet=number_of_surface_points_per_facet,
                device=device,
            )
        scenario.set_number_of_rays(number_of_rays=100)

        heliostat_for_reconstruction_name = validation_heliostat_data_mapping[0][0]
        heliostat_group_for_reconstruction = [
            group
            for group in scenario.heliostat_field.heliostat_groups
            if heliostat_for_reconstruction_name in group.names
        ][0]

        parser = PaintCalibrationDataParser(sample_limit=1)

        (
            validation_measured_flux_distributions,
            _,
            validation_incident_ray_directions,
            _,
            validation_active_heliostats_mask,
            validation_target_area_mask,
        ) = parser.parse_data_for_reconstruction(
            heliostat_data_mapping=validation_heliostat_data_mapping,
            heliostat_group=heliostat_group_for_reconstruction,
            scenario=scenario,
            device=device,
        )

        heliostat_group_for_reconstruction.activate_heliostats(
            active_heliostats_mask=validation_active_heliostats_mask,
            device=device,
        )
        heliostat_group_for_reconstruction.align_surfaces_with_incident_ray_directions(
            aim_points=scenario.target_areas.centers[validation_target_area_mask],
            incident_ray_directions=validation_incident_ray_directions,
            active_heliostats_mask=validation_active_heliostats_mask,
            device=device,
        )
        ray_tracer_ideal = HeliostatRayTracer(
            scenario=scenario,
            heliostat_group=heliostat_group_for_reconstruction,
            world_size=ddp_setup["heliostat_group_world_size"],
            rank=ddp_setup["heliostat_group_rank"],
            batch_size=heliostat_group_for_reconstruction.number_of_active_heliostats,
            random_seed=ddp_setup["heliostat_group_rank"],
            bitmap_resolution=torch.tensor([256, 256], device=device),
        )
        validation_bitmaps_per_heliostat_ideal = ray_tracer_ideal.trace_rays(
            incident_ray_directions=validation_incident_ray_directions,
            active_heliostats_mask=validation_active_heliostats_mask,
            target_area_mask=validation_target_area_mask,
            device=device,
        )

        loss_definition = loss_functions.KLDivergenceLoss()

        kl_div_ideal = loss_definition(
            prediction=validation_bitmaps_per_heliostat_ideal,
            ground_truth=validation_measured_flux_distributions,
            target_area_mask=validation_target_area_mask,
            reduction_dimensions=(1, 2),
            device=device,
        )[0].item()

    results = {
        "ideal_flux": validation_bitmaps_per_heliostat_ideal,
        "kl_div_ideal": kl_div_ideal,
    }
    results_dict[result_key] = results

    if not results_file.parent.exists():
        results_file.parent.mkdir(parents=True, exist_ok=True)

    torch.save(results_dict, results_file)


def create_deflectometry_surface(
    data_directory: pathlib.Path,
    scenario_path: pathlib.Path,
    validation_heliostat_data_mapping: list[
        tuple[str, list[pathlib.Path], list[pathlib.Path]]
    ],
    reconstruction_parameters: dict[str, float | int],
    results_file: pathlib.Path,
    result_key: str,
    device: torch.device | None,
) -> None:
    """
    Create the surface from the measured deflectometry as comparison.

    Parameters
    ----------
    data_directory : pathlib.Path
        Path to the data directory.
    scenario_path : pathlib.Path
        Path to the scenario being used.
    validation_heliostat_data_mapping : list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
        Data mapping for the plot.
    reconstruction_parameters : dict[str, float | int]
        Parameters for the reconstruction.
    results_file : pathlib.Path
        Path to the unified results file, saved as a torch checkpoint.
    result_key : str
        Key under which to store the result.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.
    """
    device = get_device(device)

    results_dict: dict[str, dict[str, np.ndarray | torch.Tensor]] = {}

    try:
        loaded = torch.load(results_file, weights_only=False)
        results_dict = cast(dict[str, torch.Tensor], loaded)
    except FileNotFoundError:
        print(f"File not found: {results_file}. Initializing with an empty dictionary.")
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
    number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
        scenario_path=scenario_path
    )

    with setup_distributed_environment(
        number_of_heliostat_groups=number_of_heliostat_groups,
        device=device,
    ) as ddp_setup:
        device = ddp_setup[config_dictionary.device]

        number_of_surface_points_per_facet = torch.tensor(
            [
                reconstruction_parameters["number_of_surface_points"],
                reconstruction_parameters["number_of_surface_points"],
            ],
            device=device,
        )

        with h5py.File(scenario_path, "r") as scenario_file:
            scenario = Scenario.load_scenario_from_hdf5(
                scenario_file=scenario_file,
                number_of_surface_points_per_facet=number_of_surface_points_per_facet,
                device=device,
            )

        heliostat_for_reconstruction_name = validation_heliostat_data_mapping[0][0]
        heliostat_group_for_reconstruction = [
            group
            for group in scenario.heliostat_field.heliostat_groups
            if heliostat_for_reconstruction_name in group.names
        ][0]

        heliostat_properties_tuples: list[tuple[str, pathlib.Path]] = [
            (
                heliostat_for_reconstruction_name,
                pathlib.Path(
                    f"{data_directory}/{heliostat_for_reconstruction_name}/{paint_mappings.SAVE_PROPERTIES}/{heliostat_for_reconstruction_name}-{paint_mappings.HELIOSTAT_PROPERTIES_KEY}.json"
                ),
            )
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

        evaluation_points = (
            utils.create_nurbs_evaluation_grid(
                number_of_evaluation_points=number_of_surface_points_per_facet,
                device=device,
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(
                1,
                heliostat_group_for_reconstruction.number_of_facets_per_heliostat,
                -1,
                -1,
            )
        )

        nurbs = NURBSSurfaces(
            degrees=heliostat_group_for_reconstruction.nurbs_degrees,
            control_points=heliostat_group_for_reconstruction.nurbs_control_points[
                0
            ].unsqueeze(0),
            device=device,
        )

        points_deflectometry, normals_deflectometry = (
            nurbs.calculate_surface_points_and_normals(
                evaluation_points=evaluation_points[0].unsqueeze(0),
                device=device,
            )
        )
        # Apply inverse canting and translation.
        facet_translations, facet_canting_vectors = facet_transforms_by_name[
            heliostat_for_reconstruction_name
        ]
        normals_decanted = perform_inverse_canting_and_translation(
            canted_points=normals_deflectometry[0],
            translation=facet_translations,
            canting=facet_canting_vectors,
            device=device,
        )

    results = {
        "points_deflectometry": points_deflectometry,
        "normals_deflectometry": normals_decanted.unsqueeze(0),
    }

    results_dict[result_key] = results

    if not results_file.parent.exists():
        results_file.parent.mkdir(parents=True, exist_ok=True)

    torch.save(results_dict, results_file)


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
    heliostat_for_reconstruction : dict[str, list[int]]
        The heliostat and its calibration numbers.
    results_dir : str
        Path to where the results will be saved.
    scenarios_dir : str
        Path to the directory containing the scenarios.
    reconstruction_parameters : dict[str, int | float]
        The reconstruction parameters.
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
    data_dir_default = config.get("data_dir", "./paint_data")
    device_default = config.get("device", "cuda")
    heliostat_for_reconstruction_default = config.get(
        "heliostat_for_reconstruction", {"AA39": [244862, 270398, 246213, 258959]}
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
        "--heliostat_for_reconstruction",
        type=str,
        help="The heliostat and its calibration numbers to be reconstructed.",
        nargs="+",
        default=heliostat_for_reconstruction_default,
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
    results_hpo_path = pathlib.Path(args.results_dir) / "hpo_results.json"
    results_path = (
        pathlib.Path(args.results_dir) / "surface_reconstruction_results.json"
    )

    deflectometry_scenario_file = (
        pathlib.Path(args.scenarios_dir) / "surface_comparison_deflectometry.h5"
    )
    ideal_scenario_file = (
        pathlib.Path(args.scenarios_dir) / "surface_reconstruction_ideal.h5"
    )

    viable_heliostats_data = pathlib.Path(args.results_dir) / "viable_heliostats.json"
    if not viable_heliostats_data.exists():
        raise FileNotFoundError(
            f"The viable heliostat list located at {viable_heliostats_data} could not be not found! Please run the ``viable_heliostat_list.py`` script to generate this list, or adjust the file path and try again."
        )

    # Load viable heliostats data.
    with open(viable_heliostats_data, "r") as f:
        viable_heliostats = json.load(f)

    heliostat_data_mapping: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]] = [
        (
            item["name"],
            [pathlib.Path(p) for p in item["calibrations"]],
            [pathlib.Path(p) for p in item["flux_images"]],
        )
        for item in viable_heliostats
    ]

    validation_heliostat_data_mapping = [
        (
            heliostat_data_mapping[0][0],
            [heliostat_data_mapping[0][1][0]],
            [heliostat_data_mapping[0][2][0]],
        )
    ]

    with open(results_hpo_path, "r") as file:
        reconstruction_parameters = json.load(file)

    # Generate and merge flux images and surfaces.
    reconstruct_and_create_flux_image(
        data_directory=data_dir,
        scenario_path=ideal_scenario_file,
        heliostat_data_mapping=heliostat_data_mapping,
        validation_heliostat_data_mapping=validation_heliostat_data_mapping,
        reconstruction_parameters=reconstruction_parameters,
        results_file=results_path,
        result_key="reconstructed",
        device=device,
    )
    create_ideal_flux_image(
        scenario_path=ideal_scenario_file,
        reconstruction_parameters=reconstruction_parameters,
        validation_heliostat_data_mapping=validation_heliostat_data_mapping,
        results_file=results_path,
        result_key="ideal",
        device=device,
    )

    create_deflectometry_surface(
        data_directory=data_dir,
        scenario_path=ideal_scenario_file,
        reconstruction_parameters=reconstruction_parameters,
        validation_heliostat_data_mapping=validation_heliostat_data_mapping,
        results_file=results_path,
        result_key="deflectometry",
        device=device,
    )
