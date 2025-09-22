import json
import logging
import pathlib
import random
from collections import Counter, defaultdict
from typing import Any, Callable

import h5py
import torch

from artist.scenario.configuration_classes import (
    ActuatorConfig,
    ActuatorListConfig,
    ActuatorParameters,
    ActuatorPrototypeConfig,
    HeliostatConfig,
    HeliostatListConfig,
    KinematicConfig,
    KinematicDeviations,
    KinematicPrototypeConfig,
    PowerPlantConfig,
    PrototypeConfig,
    SurfaceConfig,
    SurfacePrototypeConfig,
    TargetAreaConfig,
    TargetAreaListConfig,
)
from artist.scenario.surface_generator import SurfaceGenerator
from artist.util import config_dictionary, utils
from artist.util.environment_setup import get_device

log = logging.getLogger(__name__)
"""A logger for the paint data loader."""


def extract_paint_calibration_properties_data(
    heliostat_calibration_mapping: list[tuple[str, list[pathlib.Path]]],
    power_plant_position: torch.Tensor,
    heliostat_names: list[str],
    target_area_names: list[str],
    limit_number_of_measurements: int | None = None,
    centroid_extraction_method: str = config_dictionary.paint_utis,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract calibration data from ``PAINT`` calibration files.

    Parameters
    ----------
    heliostat_calibration_mapping : list[tuple[str, list[pathlib.Path]]]
        The mapping of heliostats and their calibration data files.
    power_plant_position : torch.Tensor
        The power plant position.
        Tensor of shape [3].
    heliostat_names : list[str]
        All possible heliostat names.
    target_area_names : list[str]
        All possible target area names.
    limit_number_of_measurements : int | None
        Limits the number of measurements loaded if an int is provided (default is None).
    centroid_extraction_method : str
        The method by which the focal spot centroid was extracted (default is the centroid extracted by ``UTIS``).
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The calibration focal spots.
        Tensor of shape [number_of_calibration_data_points, 4].
    torch.Tensor
        The incident ray directions.
        Tensor of shape [number_of_calibration_data_points, 4].
    torch.Tensor
        The motor positions.
        Tensor of shape [number_of_calibration_data_points, 2].
    torch.Tensor
        A mask with active heliostats and their replications.
        Tensor of shape [number_of_heliostats].
    torch.Tensor
        The target area mapping for the heliostats.
        Tensor of shape [number_of_active_heliostats].
    """
    device = get_device(device=device)

    log.info(
        "Beginning extraction of calibration properties data from ```PAINT``` file."
    )

    if centroid_extraction_method not in [
        config_dictionary.paint_utis,
        config_dictionary.paint_helios,
    ]:
        raise ValueError(
            f"The centroid extraction method {centroid_extraction_method} is not supported. Currently we"
            f"only support centroid extraction from {config_dictionary.paint_helios} or {config_dictionary.paint_utis}."
        )
    target_indices = {name: index for index, name in enumerate(target_area_names)}

    # Gather calibration data
    replication_counter: Counter[str] = Counter()
    calibration_data_per_heliostat = defaultdict(list)

    for heliostat_name, paths in heliostat_calibration_mapping:
        number_of_measurements = min(
            len(paths), limit_number_of_measurements or len(paths)
        )
        for path in paths[:number_of_measurements]:
            with open(path, "r") as f:
                calibration_data_dict = json.load(f)
            replication_counter[heliostat_name] += 1

            calibration_data_per_heliostat[heliostat_name].append(
                [
                    target_indices[
                        calibration_data_dict[
                            config_dictionary.paint_calibration_target
                        ]
                    ],
                    calibration_data_dict[config_dictionary.paint_focal_spot][
                        centroid_extraction_method
                    ],
                    calibration_data_dict[config_dictionary.paint_light_source_azimuth],
                    calibration_data_dict[
                        config_dictionary.paint_light_source_elevation
                    ],
                    [
                        calibration_data_dict[config_dictionary.paint_motor_positions][
                            config_dictionary.paint_first_axis
                        ],
                        calibration_data_dict[config_dictionary.paint_motor_positions][
                            config_dictionary.paint_second_axis
                        ],
                    ],
                ]
            )

    total_samples = sum(replication_counter[name] for name in heliostat_names)
    calibration_replications = torch.tensor(
        [replication_counter[name] for name in heliostat_names], device=device
    )

    target_area_mapping = torch.empty(total_samples, device=device, dtype=torch.long)
    focal_spots_global = torch.empty((total_samples, 3), device=device)
    azimuths = torch.empty(total_samples, device=device)
    elevations = torch.empty(total_samples, device=device)
    motor_positions = torch.empty((total_samples, 2), device=device)

    index = 0
    for name in heliostat_names:
        for (
            target_index,
            focal_spot,
            azimuth,
            elevation,
            motor_pos,
        ) in calibration_data_per_heliostat.get(name, []):
            target_area_mapping[index] = target_index
            focal_spots_global[index] = torch.tensor(focal_spot, device=device)
            azimuths[index] = azimuth
            elevations[index] = elevation
            motor_positions[index] = torch.tensor(motor_pos, device=device)
            index += 1

    focal_spots_enu = convert_wgs84_coordinates_to_local_enu(
        focal_spots_global, power_plant_position, device=device
    )
    focal_spots = utils.convert_3d_points_to_4d_format(focal_spots_enu, device=device)

    light_source_positions_enu = azimuth_elevation_to_enu(
        azimuths, elevations, degree=True, device=device
    )
    light_source_positions = utils.convert_3d_points_to_4d_format(
        light_source_positions_enu, device=device
    )
    incident_ray_directions = (
        torch.tensor([0.0, 0.0, 0.0, 1.0], device=device) - light_source_positions
    )

    log.info("Loading calibration properties data complete.")

    return (
        focal_spots,
        incident_ray_directions,
        motor_positions,
        calibration_replications,
        target_area_mapping,
    )


def extract_paint_tower_measurements(
    tower_measurements_path: pathlib.Path,
    device: torch.device | None = None,
) -> tuple[PowerPlantConfig, TargetAreaListConfig]:
    """
    Extract tower data from a ``PAINT`` tower measurements file for scenario generation.

    Parameters
    ----------
    tower_measurements_path : pathlib.Path
        The path to the tower measurement file.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    PowerPlantConfig
        The configuration of the power plant.
    TargetAreaListConfig
        The configuration of the tower target areas.
    """
    device = get_device(device=device)

    log.info("Beginning extraction of tower data from ```PAINT``` file.")

    with open(tower_measurements_path, "r") as file:
        tower_dict = json.load(file)

    power_plant_position = torch.tensor(
        tower_dict[config_dictionary.paint_power_plant_properties][
            config_dictionary.paint_coordinates
        ],
        dtype=torch.float64,
        device=device,
    )

    target_area_config_list = []

    for target_area in list(tower_dict.keys())[1:]:
        prefix = (
            "receiver_outer_"
            if target_area == config_dictionary.target_area_receiver
            else ""
        )

        target_area_corners = [
            f"{prefix}{config_dictionary.paint_upper_left}",
            f"{prefix}{config_dictionary.paint_lower_left}",
            f"{prefix}{config_dictionary.paint_upper_right}",
            f"{prefix}{config_dictionary.paint_lower_right}",
        ]

        target_area_corner_points_wgs84 = torch.tensor(
            [
                tower_dict[target_area][config_dictionary.paint_coordinates][corner]
                for corner in target_area_corners
            ],
            dtype=torch.float64,
            device=device,
        )

        corner_points_enu = convert_wgs84_coordinates_to_local_enu(
            target_area_corner_points_wgs84, power_plant_position, device=device
        )

        upper_left, lower_left, upper_right, lower_right = corner_points_enu
        plane_e, plane_u = corner_points_to_plane(
            upper_left, upper_right, lower_left, lower_right
        )

        center_lat_lon = torch.tensor(
            [
                tower_dict[target_area][config_dictionary.paint_coordinates][
                    config_dictionary.paint_center
                ]
            ],
            dtype=torch.float64,
            device=device,
        )
        center_enu = convert_wgs84_coordinates_to_local_enu(
            center_lat_lon, power_plant_position, device=device
        )
        center = utils.convert_3d_points_to_4d_format(center_enu[0], device=device)

        normal_vector = utils.convert_3d_directions_to_4d_format(
            torch.tensor(
                [tower_dict[target_area][config_dictionary.paint_normal_vector]],
                device=device,
            ),
            device=device,
        )

        tower_area_config = TargetAreaConfig(
            target_area_key=target_area,
            geometry=tower_dict[target_area][
                config_dictionary.paint_target_area_geometry
            ],
            center=center,
            normal_vector=normal_vector,
            plane_e=plane_e,
            plane_u=plane_u,
        )

        target_area_config_list.append(tower_area_config)

    # Create the power plant configuration.
    power_plant_config = PowerPlantConfig(power_plant_position=power_plant_position)

    # Create the tower area configurations.
    target_area_list_config = TargetAreaListConfig(target_area_config_list)

    log.info("Loading tower data` data complete.")

    return power_plant_config, target_area_list_config


def extract_paint_heliostat_properties(
    heliostat_properties_path: pathlib.Path,
    power_plant_position: torch.Tensor,
    device: torch.device | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    KinematicDeviations,
    torch.Tensor,
    list[tuple[str, bool, list[float], ActuatorParameters]],
]:
    """
    Extract heliostat properties from paint.

    Parameters
    ----------
    heliostat_properties_path : pathlib.Path
        The path to the heliostat properties file.
    power_plant_position : torch.Tensor
        Tensor of shape [3].
        The power plant position.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The heliostat position.
        Tensor of shape [4].
    torch.Tensor
        The facet translation vectors.
        Tensor of shape [number_of_facets, 4].
    torch.Tensor
        The facet canting vectors in east and north direction.
        Tensor of shape [number_of_facets, 2, 4].
    KinematicDeviations
        The kinematic deviation parameters.
    torch.Tensor
        The initial orientation.
        Tensor of shape [4].
    list[tuple[str, bool, list[float], ActuatorParameters]]
        The actuator parameter list.
    """
    device = get_device(device=device)

    with open(heliostat_properties_path, "r") as file:
        heliostat_dict = json.load(file)

    log.info("Beginning extraction of heliostat properties data from ```PAINT``` file.")

    heliostat_position_3d = convert_wgs84_coordinates_to_local_enu(
        torch.tensor(
            [heliostat_dict[config_dictionary.paint_heliostat_position]],
            dtype=torch.float64,
            device=device,
        ),
        power_plant_position,
        device=device,
    )
    heliostat_position = utils.convert_3d_points_to_4d_format(
        heliostat_position_3d[0], device=device
    )

    number_of_facets = heliostat_dict[config_dictionary.paint_facet_properties][
        config_dictionary.paint_number_of_facets
    ]

    facet_translation_vectors = torch.empty(number_of_facets, 3, device=device)
    canting = torch.empty(number_of_facets, 2, 3, device=device)

    for facet in range(number_of_facets):
        facet_translation_vectors[facet, :] = torch.tensor(
            heliostat_dict[config_dictionary.paint_facet_properties][
                config_dictionary.paint_facets
            ][facet][config_dictionary.paint_translation_vector],
            device=device,
        )
        canting[facet, 0] = torch.tensor(
            heliostat_dict[config_dictionary.paint_facet_properties][
                config_dictionary.paint_facets
            ][facet][config_dictionary.paint_canting_e],
            device=device,
        )
        canting[facet, 1] = torch.tensor(
            heliostat_dict[config_dictionary.paint_facet_properties][
                config_dictionary.paint_facets
            ][facet][config_dictionary.paint_canting_n],
            device=device,
        )

    # Convert to 4D format.
    facet_translation_vectors = utils.convert_3d_directions_to_4d_format(
        facet_translation_vectors, device=device
    )
    canting = utils.convert_3d_directions_to_4d_format(canting, device=device)

    kinematic_deviations = KinematicDeviations(
        first_joint_translation_e=torch.tensor(
            heliostat_dict[config_dictionary.paint_kinematic][
                config_dictionary.paint_first_joint_translation_e
            ],
            device=device,
        ),
        first_joint_translation_n=torch.tensor(
            heliostat_dict[config_dictionary.paint_kinematic][
                config_dictionary.paint_first_joint_translation_n
            ],
            device=device,
        ),
        first_joint_translation_u=torch.tensor(
            heliostat_dict[config_dictionary.paint_kinematic][
                config_dictionary.paint_first_joint_translation_u
            ],
            device=device,
        ),
        first_joint_tilt_e=torch.tensor(0.0, device=device),
        first_joint_tilt_n=torch.tensor(0.0, device=device),
        first_joint_tilt_u=torch.tensor(0.0, device=device),
        second_joint_translation_e=torch.tensor(
            heliostat_dict[config_dictionary.paint_kinematic][
                config_dictionary.paint_second_joint_translation_e
            ],
            device=device,
        ),
        second_joint_translation_n=torch.tensor(
            heliostat_dict[config_dictionary.paint_kinematic][
                config_dictionary.paint_second_joint_translation_n
            ],
            device=device,
        ),
        second_joint_translation_u=torch.tensor(
            heliostat_dict[config_dictionary.paint_kinematic][
                config_dictionary.paint_second_joint_translation_u
            ],
            device=device,
        ),
        second_joint_tilt_e=torch.tensor(0.0, device=device),
        second_joint_tilt_n=torch.tensor(0.0, device=device),
        second_joint_tilt_u=torch.tensor(0.0, device=device),
        concentrator_translation_e=torch.tensor(
            heliostat_dict[config_dictionary.paint_kinematic][
                config_dictionary.paint_concentrator_translation_e
            ],
            device=device,
        ),
        concentrator_translation_n=torch.tensor(
            heliostat_dict[config_dictionary.paint_kinematic][
                config_dictionary.paint_concentrator_translation_n
            ],
            device=device,
        ),
        concentrator_translation_u=torch.tensor(
            heliostat_dict[config_dictionary.paint_kinematic][
                config_dictionary.paint_concentrator_translation_u
            ],
            device=device,
        ),
        concentrator_tilt_e=torch.tensor(0.0, device=device),
        concentrator_tilt_n=torch.tensor(0.0, device=device),
        concentrator_tilt_u=torch.tensor(0.0, device=device),
    )

    # Include the initial orientation for the kinematic.
    initial_orientation = utils.convert_3d_directions_to_4d_format(
        torch.tensor(
            heliostat_dict[config_dictionary.paint_initial_orientation],
            device=device,
        ),
        device=device,
    )

    paint_actuators = list(
        heliostat_dict[config_dictionary.paint_kinematic][
            config_dictionary.paint_actuators
        ]
    )
    actuator_parameters_list = []

    for paint_actuator in paint_actuators:
        parameters = ActuatorParameters(
            increment=torch.tensor(
                paint_actuator[config_dictionary.paint_increment], device=device
            ),
            initial_stroke_length=torch.tensor(
                paint_actuator[config_dictionary.paint_initial_stroke_length],
                device=device,
            ),
            offset=torch.tensor(
                paint_actuator[config_dictionary.paint_offset], device=device
            ),
            pivot_radius=torch.tensor(
                paint_actuator[config_dictionary.paint_pivot_radius],
                device=device,
            ),
            initial_angle=torch.tensor(
                paint_actuator[config_dictionary.paint_initial_angle],
                device=device,
            ),
        )
        actuator_type = paint_actuator[config_dictionary.paint_actuator_type]
        clockwise_axis_movement = paint_actuator[
            config_dictionary.paint_clockwise_axis_movement
        ]
        min_max_motor_positions = [
            paint_actuator[config_dictionary.paint_min_increment],
            paint_actuator[config_dictionary.paint_max_increment],
        ]
        actuator_parameters_list.append(
            (
                actuator_type,
                clockwise_axis_movement,
                min_max_motor_positions,
                parameters,
            )
        )
    log.info("Loading heliostat properties data complete.")

    return (
        heliostat_position,
        facet_translation_vectors,
        canting,
        kinematic_deviations,
        initial_orientation,
        actuator_parameters_list,
    )


def extract_paint_deflectometry_data(
    heliostat_deflectometry_path: pathlib.Path,
    number_of_facets: int,
    device: torch.device | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Extract paint deflectometry data.

    Parameters
    ----------
    heliostat_deflectometry_path : pathlib.Path
        The heliostat deflectometry file path.
    number_of_facets : int
        The number of facets.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    list[torch.Tensor]
        The surface points per facet.
    list[torch.Tensor]
        The surface normals per facet.
    """
    device = get_device(device=device)

    log.info("Beginning extraction of deflectometry data from ```PAINT``` file.")

    with h5py.File(heliostat_deflectometry_path, "r") as file:
        surface_points_with_facets_list = []
        surface_normals_with_facets_list = []
        for f in range(number_of_facets):
            number_of_points = len(
                file[f"{config_dictionary.paint_facet}{f + 1}"][
                    config_dictionary.paint_surface_points
                ]
            )
            single_facet_surface_points = torch.empty(
                number_of_points, 3, device=device
            )
            single_facet_surface_normals = torch.empty(
                number_of_points, 3, device=device
            )

            points_data = torch.tensor(
                file[f"{config_dictionary.paint_facet}{f + 1}"][
                    config_dictionary.paint_surface_points
                ][()],
                device=device,
            )
            normals_data = torch.tensor(
                file[f"{config_dictionary.paint_facet}{f + 1}"][
                    config_dictionary.paint_surface_normals
                ][()],
                device=device,
            )

            for i, point_data in enumerate(points_data):
                single_facet_surface_points[i, :] = point_data
            for i, normal_data in enumerate(normals_data):
                single_facet_surface_normals[i, :] = normal_data
            surface_points_with_facets_list.append(single_facet_surface_points)
            surface_normals_with_facets_list.append(single_facet_surface_normals)

    log.info("Loading deflectometry data complete.")

    return surface_points_with_facets_list, surface_normals_with_facets_list


def _ideal_surface_generator(
    file_tuple: tuple[str, pathlib.Path],
    facet_translation_vectors: torch.Tensor,
    canting: torch.Tensor,
    number_of_nurbs_control_points: torch.Tensor,
    device: torch.device | None,
    **kwargs: Any,
) -> SurfaceConfig:
    r"""
    Generate a surface configuration for an ideal heliostat.

    This is a helper function designed to be passed as a callable to the `_process_heliostats_from_paths` function,
    handling the specific logic for generating an ideal surface.

    Parameters
    ----------
    file_tuple : tuple[str, pathlib.Path]
        A tuple containing the heliostat name and path to the properties file, not used in this function but required
        for API compatibility.
    facet_translation_vectors : torch.Tensor
        The translation vectors for each facet.
        Tensor of shape [number_of_facets, 4].
    canting : torch.Tensor
        The canting vectors for each facet.
        Tensor of shape [number_of_facets, 2, 4].
    number_of_nurbs_control_points : torch.Tensor
        The number of NURBS control points.
        Tensor of shape [2].
    device : torch.device | None
        The device to use.
    \*\*kwargs : Any
        Additional keyword arguments, not used by this function but accepted for API compatibility.

    Returns
    -------
    SurfaceConfig
        The generated ideal surface configuration object.
    """
    device = get_device(device=device)

    surface_generator = SurfaceGenerator(
        number_of_control_points=number_of_nurbs_control_points.to(device),
        device=device,
    )
    return surface_generator.generate_ideal_surface_config(
        facet_translation_vectors=facet_translation_vectors,
        canting=canting,
        device=device,
    )


def _fitted_surface_generator(
    file_tuple: tuple[str, pathlib.Path, pathlib.Path],
    facet_translation_vectors: torch.Tensor,
    canting: torch.Tensor,
    number_of_nurbs_control_points: torch.Tensor,
    device: torch.device | None,
    **kwargs: Any,
) -> SurfaceConfig:
    r"""
    Generate a surface configuration for a fitted heliostat.

    This is a helper function designed to be passed as a callable to the `_process_heliostats_from_paths` function,
    handling the specific logic for generating a fitted surface based on deflectometry data.

    Parameters
    ----------
    file_tuple : tuple[str, pathlib.Path, pathlib.Path]
        A tuple containing the heliostat name, path to the properties file, and path to deflecometry data file.
    facet_translation_vectors : torch.Tensor
        The translation vectors for each facet.
        Tensor of shape [number_of_facets, 4].
    canting : torch.Tensor
        The canting vectors for each facet.
        Tensor of shape [number_of_facets, 2, 4].
    number_of_nurbs_control_points : torch.Tensor
        The number of NURBS control points.
        Tensor of shape [2].
    device : torch.device | None
        The device to use.
    \*\*kwargs : Any
        Additional keyword arguments used for the fitting process, including:
        - `nurbs_fit_optimizer`: The PyTorch optimizer for the NURBS fit.
        - `nurbs_fit_scheduler`: The PyTorch learning rate scheduler for the fit.
        - `deflectometry_step_size`: Step size to reduce data points for efficiency.
        - `nurbs_fit_method`: The fitting method to use.
        - `nurbs_fit_tolerance`: The tolerance for the fitting convergence.
        - `nurbs_fit_max_epoch`: The maximum number of epochs for the fit.

    Returns
    -------
    SurfaceConfig
        The generated fitted surface configuration object.
    """
    device = get_device(device=device)

    surface_generator = SurfaceGenerator(
        number_of_control_points=number_of_nurbs_control_points.to(device),
        device=device,
    )
    (surface_points_with_facets_list, surface_normals_with_facets_list) = (
        extract_paint_deflectometry_data(
            heliostat_deflectometry_path=pathlib.Path(file_tuple[2]),
            number_of_facets=facet_translation_vectors.shape[0],
            device=device,
        )
    )
    return surface_generator.generate_fitted_surface_config(
        heliostat_name=str(file_tuple[0]),
        facet_translation_vectors=facet_translation_vectors,
        canting=canting,
        surface_points_with_facets_list=surface_points_with_facets_list,
        surface_normals_with_facets_list=surface_normals_with_facets_list,
        optimizer=kwargs.get("nurbs_fit_optimizer"),
        scheduler=kwargs.get("nurbs_fit_scheduler"),
        deflectometry_step_size=kwargs.get("deflectometry_step_size", 100),
        fit_method=kwargs.get(
            "nurbs_fit_method", config_dictionary.fit_nurbs_from_normals
        ),
        tolerance=kwargs.get("nurbs_fit_tolerance", 1e-10),
        max_epoch=kwargs.get("nurbs_fit_max_epoch", 400),
        device=device,
    )


def _process_heliostats_from_paths(
    paths: (
        list[tuple[str, pathlib.Path]] | list[tuple[str, pathlib.Path, pathlib.Path]]
    ),
    power_plant_position: torch.Tensor,
    number_of_nurbs_control_points: torch.Tensor,
    surface_config_generator: Callable,
    device: torch.device | None = None,
    **kwargs: Any,
) -> tuple[HeliostatListConfig, PrototypeConfig]:
    r"""
    Process heliostat properties from file paths.

    Parameters
    ----------
    paths : list[tuple[str, pathlib.Path]] | list[tuple[str, pathlib.Path, pathlib.Path]]
        The list of heliostat paths, where each element's structure depends on whether the surface is ideal or fitted,
        i.e. for ideal surfaces only the heliostat name and path to the properties file is required, whilst for the
        fitted surface the path to the deflectometry file is also required.
    power_plant_position : torch.Tensor
        The position of the power plant in latitude, longitude and elevation.
    number_of_nurbs_control_points : torch.Tensor
        The number of NURBS control points.
    surface_config_generator : Callable
        A function that generates the surface configuration, either ideal or fitted.
    device : torch.device | None
        The device to use.
    \*\*kwargs : Any
        Any additional arguments to pass to the surface config generator function.

    Returns
    -------
    HeliostatListConfig
        The configuration of all heliostats in the scenario.
    PrototypeConfig
        The configuration for a heliostat prototype.
    """
    device = get_device(device=device)

    prototype_surface = None
    prototype_kinematic = None
    prototype_actuator_list = None
    heliostat_config_list = []

    for heliostat_index, file_tuple in enumerate(paths):
        # Extract common heliostat properties.
        (
            heliostat_position,
            facet_translation_vectors,
            canting,
            kinematic_deviations,
            initial_orientation,
            actuator_parameters_list,
        ) = extract_paint_heliostat_properties(
            heliostat_properties_path=pathlib.Path(file_tuple[1]),
            power_plant_position=power_plant_position,
            device=device,
        )

        # Generate the surface configuration using the callable generator function.
        surface_config = surface_config_generator(
            file_tuple=file_tuple,
            facet_translation_vectors=facet_translation_vectors,
            canting=canting,
            number_of_nurbs_control_points=number_of_nurbs_control_points,
            device=device,
            **kwargs,
        )
        prototype_surface = surface_config

        kinematic_config = KinematicConfig(
            type=config_dictionary.rigid_body_key,
            initial_orientation=initial_orientation,
            deviations=kinematic_deviations,
        )
        prototype_kinematic = kinematic_config

        actuator_list = []
        for actuator_index, actuator_parameters_tuple in enumerate(
            actuator_parameters_list
        ):
            actuator = ActuatorConfig(
                key=f"{config_dictionary.heliostat_actuator_key}_{actuator_index}",
                type=actuator_parameters_tuple[0],
                clockwise_axis_movement=actuator_parameters_tuple[1],
                min_max_motor_positions=actuator_parameters_tuple[2],
                parameters=actuator_parameters_tuple[3],
            )
            actuator_list.append(actuator)
        actuators_list_config = ActuatorListConfig(actuator_list=actuator_list)
        prototype_actuator_list = actuator_list

        # Create the heliostat configuration with the generated surface config.
        heliostat_config = HeliostatConfig(
            name=str(file_tuple[0]),
            id=heliostat_index,
            position=heliostat_position,
            surface=surface_config,
            kinematic=kinematic_config,
            actuators=actuators_list_config,
        )
        heliostat_config_list.append(heliostat_config)

        # Create the prototype configuration.
        surface_prototype_config = SurfacePrototypeConfig(
            facet_list=prototype_surface.facet_list
        )
        kinematic_prototype_config = KinematicPrototypeConfig(
            type=prototype_kinematic.type,
            initial_orientation=prototype_kinematic.initial_orientation,
            deviations=prototype_kinematic.deviations,
        )
        actuator_prototype_config = ActuatorPrototypeConfig(
            actuator_list=prototype_actuator_list
        )

    prototype_config = PrototypeConfig(
        surface_prototype=surface_prototype_config,
        kinematic_prototype=kinematic_prototype_config,
        actuators_prototype=actuator_prototype_config,
    )
    heliostats_list_config = HeliostatListConfig(heliostat_list=heliostat_config_list)

    return heliostats_list_config, prototype_config


def extract_paint_heliostats_ideal_surface(
    paths: list[tuple[str, pathlib.Path]],
    power_plant_position: torch.Tensor,
    number_of_nurbs_control_points: torch.Tensor = torch.tensor([10, 10]),
    device: torch.device | None = None,
) -> tuple[HeliostatListConfig, PrototypeConfig]:
    """
    Extract heliostat data with ideal surfaces from ``PAINT`` heliostat properties files.

    Parameters
    ----------
    paths : list[tuple[str, pathlib.Path]]
        Name of the heliostat and path to the heliostat properties file
    power_plant_position : torch.Tensor
        The position of the power plant in latitude, longitude and elevation.
        Tensor of shape [3].
    number_of_nurbs_control_points : torch.Tensor
        The number of NURBS control points in both dimensions (default is torch.tensor([10,10])).
        Tensor of shape [2].
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    HeliostatListConfig
        The configuration of all heliostats in the scenario.
    PrototypeConfig
        The configuration for a heliostat prototype.
    """
    device = get_device(device=device)

    return _process_heliostats_from_paths(
        paths=paths,
        power_plant_position=power_plant_position,
        number_of_nurbs_control_points=number_of_nurbs_control_points,
        surface_config_generator=_ideal_surface_generator,
        device=device,
    )


def extract_paint_heliostats_fitted_surface(
    paths: list[tuple[str, pathlib.Path, pathlib.Path]],
    power_plant_position: torch.Tensor,
    nurbs_fit_optimizer: torch.optim.Optimizer,
    nurbs_fit_scheduler: torch.optim.lr_scheduler.LRScheduler,
    number_of_nurbs_control_points: torch.Tensor = torch.tensor([10, 10]),
    deflectometry_step_size: int = 100,
    nurbs_fit_method: str = config_dictionary.fit_nurbs_from_normals,
    nurbs_fit_tolerance: float = 1e-10,
    nurbs_fit_max_epoch: int = 400,
    device: torch.device | None = None,
) -> tuple[HeliostatListConfig, PrototypeConfig]:
    """
    Extract heliostat data with fitted surfaces from ``PAINT`` heliostat properties and deflectometry files.

    Parameters
    ----------
    paths : list[tuple[str, pathlib.Path, pathlib.Path]]
        Name of the heliostat and a pair of heliostat properties and deflectometry file paths.
    power_plant_position : torch.Tensor
        The position of the power plant in latitude, longitude and elevation.
        Tensor of shape [3].
    nurbs_fit_optimizer : torch.optim.Optimizer
        The NURBS fit optimizer.
    nurbs_fit_scheduler : torch.optim.lr_scheduler.LRScheduler
        The NURBS fit learning rate scheduler.
    number_of_nurbs_control_points : torch.Tensor
        The number of NURBS control points in both dimensions (default is torch.tensor([10,10])).
        Tensor of shape [2].
    deflectometry_step_size : int
        The step size used to reduce the number of deflectometry points and normals for compute efficiency (default is 100).
    nurbs_fit_method : str
        The method used to fit the NURBS, either from deflectometry points or normals (default is config_dictionary.fit_nurbs_from_normals).
    nurbs_fit_tolerance : float
        The tolerance value used for fitting NURBS surfaces to deflectometry (default is 1e-10).
    nurbs_fit_max_epoch : int
        The maximum number of epochs for the NURBS fit (default is 400).
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    HeliostatListConfig
        The configuration of all heliostats in the scenario.
    PrototypeConfig
        The configuration for a heliostat prototype.
    """
    device = get_device(device=device)

    return _process_heliostats_from_paths(
        paths=paths,
        power_plant_position=power_plant_position,
        number_of_nurbs_control_points=number_of_nurbs_control_points,
        surface_config_generator=_fitted_surface_generator,
        device=device,
        nurbs_fit_optimizer=nurbs_fit_optimizer,
        nurbs_fit_scheduler=nurbs_fit_scheduler,
        deflectometry_step_size=deflectometry_step_size,
        nurbs_fit_method=nurbs_fit_method,
        nurbs_fit_tolerance=nurbs_fit_tolerance,
        nurbs_fit_max_epoch=nurbs_fit_max_epoch,
    )


def azimuth_elevation_to_enu(
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    slant_range: float = 1.0,
    degree: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Transform coordinates from azimuth and elevation to east, north, up.

    This method assumes a south-oriented azimuth-elevation coordinate system, where 0° points toward the south.

    Parameters
    ----------
    azimuth : torch.Tensor
        Azimuth, 0° points toward the south (degrees).
    elevation : torch.Tensor
        Elevation angle above horizon, neglecting aberrations (degrees).
    slant_range : float
        Slant range in meters (default is 1.0).
    degree : bool
        Whether input is given in degrees (default is True).
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The east, north, up (ENU) coordinates.
    """
    device = get_device(device=device)

    if degree:
        elevation = torch.deg2rad(elevation)
        azimuth = torch.deg2rad(azimuth)

    azimuth[azimuth < 0] += 2 * torch.pi

    r = slant_range * torch.cos(elevation)

    enu = torch.zeros((azimuth.shape[0], 3), device=device)

    enu[:, 0] = r * torch.sin(azimuth)
    enu[:, 1] = -r * torch.cos(azimuth)
    enu[:, 2] = slant_range * torch.sin(elevation)

    return enu


def convert_wgs84_coordinates_to_local_enu(
    coordinates_to_transform: torch.Tensor,
    reference_point: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Transform coordinates from latitude, longitude and altitude (WGS84) to local east, north, up (ENU).

    This function calculates the north and east offsets in meters of a coordinate from the reference point.
    It converts the latitude and longitude to radians, calculates the radius of curvature values,
    and then computes the offsets based on the differences between the coordinate and the reference point.
    Finally, it returns a tensor containing these offsets along with the altitude difference.

    Parameters
    ----------
    coordinates_to_transform : torch.Tensor
        The coordinates in latitude, longitude, altitude that are to be transformed.
        Tensor of shape [number_of_coordinates, 3].
    reference_point : torch.Tensor
        The center of origin of the ENU coordinate system in WGS84 coordinates.
        Tensor of shape [3].
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The east offsets in meters, norths offset in meters, and the altitude differences from the reference point.
        Tensor of shape [number_of_coordinates, 3].
    """
    device = get_device(device=device)

    transformed_coordinates = torch.zeros_like(
        coordinates_to_transform, dtype=torch.float32, device=device
    )

    wgs84_a = 6378137.0  # Major axis in meters
    wgs84_b = 6356752.314245  # Minor axis in meters
    wgs84_e2 = (wgs84_a**2 - wgs84_b**2) / wgs84_a**2  # Eccentricity squared

    # Convert latitude and longitude to radians.
    latitudes = torch.deg2rad(coordinates_to_transform[:, 0])
    longitudes = torch.deg2rad(coordinates_to_transform[:, 1])
    latitude_reference_point = torch.deg2rad(reference_point[0])
    longitude_reference_point = torch.deg2rad(reference_point[1])

    # Calculate meridional radius of curvature for the first latitude.
    sin_lat1 = torch.sin(latitudes)
    rn1 = wgs84_a / torch.sqrt(1 - wgs84_e2 * sin_lat1**2)

    # Calculate transverse radius of curvature for the first latitude.
    rm1 = (wgs84_a * (1 - wgs84_e2)) / ((1 - wgs84_e2 * sin_lat1**2) ** 1.5)

    # Calculate delta latitude and delta longitude in radians.
    dlat_rad = latitude_reference_point - latitudes
    dlon_rad = longitude_reference_point - longitudes

    # Calculate north and east offsets in meters.
    transformed_coordinates[:, 0] = -(dlon_rad * rn1 * torch.cos(latitudes))
    transformed_coordinates[:, 1] = -(dlat_rad * rm1)
    transformed_coordinates[:, 2] = coordinates_to_transform[:, 2] - reference_point[2]

    return transformed_coordinates


def corner_points_to_plane(
    upper_left: torch.Tensor,
    upper_right: torch.Tensor,
    lower_left: torch.Tensor,
    lower_right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Span a plane from corner points.

    Parameters
    ----------
    upper_left : torch.Tensor
        The upper left corner coordinate.
        Tensor of shape [3].
    upper_right : torch.Tensor
        The upper right corner coordinate.
        Tensor of shape [3].
    lower_left : torch.Tensor
        The lower left corner coordinate.
        Tensor of shape [3].
    lower_right : torch.Tensor
        The lower right corner coordinate.
        Tensor of shape [3].

    Returns
    -------
    torch.Tensor
        The plane measurement in east direction.
    torch.Tensor
        The plane measurement in up direction.
    """
    plane_e = (
        torch.abs(upper_right[0] - upper_left[0])
        + torch.abs(lower_right[0] - lower_left[0])
    ) / 2
    plane_u = (
        torch.abs(upper_left[2] - lower_left[2])
        + torch.abs(upper_right[2] - lower_right[2])
    ) / 2
    return plane_e, plane_u


def build_heliostat_data_mapping(
    base_path: str,
    heliostat_names: list[str],
    number_of_measurements: int,
    image_variant: str,
    randomize: bool = True,
    seed: int = 42,
) -> list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]:
    """
    Build a mapping of heliostat names to their calibration property and image files loaded from ``PAINT``.

    It assuemes that the data directory has the same structure  and file names the ``PAINT`` data base.
    This method loads property and image files from the specified variant, and returns a structured mapping.
    If fewer measurements are available than requested, a warning is logged and the available
    subset is used. Optionally, the selection can be randomized using a fixed seed.

    Parameters
    ----------
    base_path : str
        Path to the root directory containing heliostat calibration data.
    heliostat_names : list[str]
        List of heliostat names to include in the mapping.
    number_of_measurements : int
        Number of valid calibration samples to retrieve per heliostat.
    image_variant : str
        Image variant to use. Must match the expected filename suffix (``flux``, ``flux-centered``, ``cropped``, or ``raw``).
    randomize : bool
        Whether to shuffle the measurement files before selection (default is True).
    seed : int
        Random seed for reproducibility (default is 42).

    Returns
    -------
    list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
        A list of tuples for each heliostat, where each tuple contains:
        - the heliostat name,
        - a list of selected property file paths,
        - a list of corresponding image file paths.
    """
    base = pathlib.Path(base_path)
    heliostat_map = []

    for name in heliostat_names:
        calibration_dir = base / name / config_dictionary.paint_calibration_folder_name
        if not calibration_dir.exists():
            log.warning(f"Calibration directory for {name} not found.")
            continue

        property_files = list(
            calibration_dir.glob(
                config_dictionary.paint_calibration_properties_file_name_ending
            )
        )

        if randomize:
            random.Random(seed).shuffle(property_files)
        else:
            property_files.sort()

        properties, images = [], []

        for property_file in property_files:
            id_str = property_file.stem.split("-")[0]
            image_file = calibration_dir / f"{id_str}-{image_variant}.png"

            if image_file.exists():
                properties.append(property_file)
                images.append(image_file)

                if len(properties) == number_of_measurements:
                    break

        if len(properties) < number_of_measurements:
            log.warning(
                f"{name} has only {len(properties)} valid measurements (needed {number_of_measurements})."
            )

        if properties and images:
            heliostat_map.append((name, properties, images))

    return heliostat_map
