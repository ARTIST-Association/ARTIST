import json
import logging
import pathlib
import random
from typing import Any, Callable, List, cast

import h5py
import paint.util.paint_mappings as paint_mappings
import torch

from artist.scenario.configuration_classes import (
    ActuatorConfig,
    ActuatorListConfig,
    ActuatorParameters,
    ActuatorPrototypeConfig,
    HeliostatConfig,
    HeliostatListConfig,
    KinematicsConfig,
    KinematicsDeviations,
    KinematicsPrototypeConfig,
    PowerPlantConfig,
    PrototypeConfig,
    SurfaceConfig,
    SurfacePrototypeConfig,
    TargetAreaConfig,
    TargetAreaListConfig,
)
from artist.scenario.surface_generator import SurfaceGenerator
from artist.util import config_dictionary, index_mapping, utils
from artist.util.environment_setup import get_device

log = logging.getLogger(__name__)
"""A logger for the paint data loader."""


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

    log.info("Beginning extraction of tower data from PAINT file.")

    with open(tower_measurements_path, "r") as file:
        tower_dict = json.load(file)

    power_plant_position = torch.tensor(
        tower_dict[paint_mappings.POWER_PLANT_KEY][
            paint_mappings.TOWER_COORDINATES_KEY
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
            f"{prefix}{paint_mappings.UPPER_LEFT}",
            f"{prefix}{paint_mappings.LOWER_LEFT}",
            f"{prefix}{paint_mappings.UPPER_RIGHT}",
            f"{prefix}{paint_mappings.LOWER_RIGHT}",
        ]

        target_area_corner_points_wgs84 = torch.tensor(
            [
                tower_dict[target_area][paint_mappings.TOWER_COORDINATES_KEY][corner]
                for corner in target_area_corners
            ],
            dtype=torch.float64,
            device=device,
        )

        corner_points_enu = utils.convert_wgs84_coordinates_to_local_enu(
            target_area_corner_points_wgs84, power_plant_position, device=device
        )

        upper_left, lower_left, upper_right, lower_right = corner_points_enu
        plane_e, plane_u = corner_points_to_plane(
            upper_left, upper_right, lower_left, lower_right
        )

        center_lat_lon = torch.tensor(
            [
                tower_dict[target_area][paint_mappings.TOWER_COORDINATES_KEY][
                    paint_mappings.CENTER
                ]
            ],
            dtype=torch.float64,
            device=device,
        )
        center_enu = utils.convert_wgs84_coordinates_to_local_enu(
            center_lat_lon, power_plant_position, device=device
        )
        center = utils.convert_3d_points_to_4d_format(center_enu[0], device=device)

        normal_vector = utils.convert_3d_directions_to_4d_format(
            torch.tensor(
                [tower_dict[target_area][paint_mappings.TOWER_NORMAL_VECTOR_KEY]],
                device=device,
            ),
            device=device,
        )

        tower_area_config = TargetAreaConfig(
            target_area_key=target_area,
            geometry=tower_dict[target_area][paint_mappings.TOWER_TYPE_KEY],
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

    log.info("Loading tower data complete.")

    return power_plant_config, target_area_list_config


def extract_paint_heliostat_properties(
    heliostat_properties_path: pathlib.Path,
    power_plant_position: torch.Tensor,
    device: torch.device | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    KinematicsDeviations,
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
    KinematicsDeviations
        The kinematics deviation parameters.
    torch.Tensor
        The initial orientation.
        Tensor of shape [4].
    list[tuple[str, bool, list[float], ActuatorParameters]]
        The actuator parameter list.
    """
    device = get_device(device=device)

    with open(heliostat_properties_path, "r") as file:
        heliostat_dict = json.load(file)

    log.info("Beginning extraction of heliostat properties data from PAINT file.")

    heliostat_position_3d = utils.convert_wgs84_coordinates_to_local_enu(
        torch.tensor(
            [heliostat_dict[paint_mappings.HELIOSTAT_POSITION_KEY]],
            dtype=torch.float64,
            device=device,
        ),
        power_plant_position,
        device=device,
    )
    heliostat_position = utils.convert_3d_points_to_4d_format(
        heliostat_position_3d[0], device=device
    )

    number_of_facets = heliostat_dict[paint_mappings.FACET_PROPERTIES_KEY][
        paint_mappings.NUM_FACETS
    ]

    facet_translation_vectors = torch.empty(number_of_facets, 3, device=device)
    canting = torch.empty(number_of_facets, 2, 3, device=device)

    for facet in range(number_of_facets):
        facet_translation_vectors[facet, :] = torch.tensor(
            heliostat_dict[paint_mappings.FACET_PROPERTIES_KEY][
                paint_mappings.FACETS_LIST
            ][facet][paint_mappings.TRANSLATION_VECTOR],
            device=device,
        )
        canting[facet, index_mapping.facet_canting_e] = torch.tensor(
            heliostat_dict[paint_mappings.FACET_PROPERTIES_KEY][
                paint_mappings.FACETS_LIST
            ][facet][paint_mappings.CANTING_E],
            device=device,
        )
        canting[facet, index_mapping.facet_canting_n] = torch.tensor(
            heliostat_dict[paint_mappings.FACET_PROPERTIES_KEY][
                paint_mappings.FACETS_LIST
            ][facet][paint_mappings.CANTING_N],
            device=device,
        )

    # Convert to 4D format.
    facet_translation_vectors = utils.convert_3d_directions_to_4d_format(
        facet_translation_vectors, device=device
    )
    canting = utils.convert_3d_directions_to_4d_format(canting, device=device)

    kinematics_deviations = KinematicsDeviations(
        first_joint_translation_e=torch.tensor(
            heliostat_dict[paint_mappings.KINEMATIC_PROPERTIES_KEY][
                paint_mappings.FIRST_JOINT_TRANSLATION_E_KEY
            ],
            device=device,
        ),
        first_joint_translation_n=torch.tensor(
            heliostat_dict[paint_mappings.KINEMATIC_PROPERTIES_KEY][
                paint_mappings.FIRST_JOINT_TRANSLATION_N_KEY
            ],
            device=device,
        ),
        first_joint_translation_u=torch.tensor(
            heliostat_dict[paint_mappings.KINEMATIC_PROPERTIES_KEY][
                paint_mappings.FIRST_JOINT_TRANSLATION_U_KEY
            ],
            device=device,
        ),
        first_joint_tilt_n=torch.tensor(0.0, device=device),
        first_joint_tilt_u=torch.tensor(0.0, device=device),
        second_joint_translation_e=torch.tensor(
            heliostat_dict[paint_mappings.KINEMATIC_PROPERTIES_KEY][
                paint_mappings.SECOND_JOINT_TRANSLATION_E_KEY
            ],
            device=device,
        ),
        second_joint_translation_n=torch.tensor(
            heliostat_dict[paint_mappings.KINEMATIC_PROPERTIES_KEY][
                paint_mappings.SECOND_JOINT_TRANSLATION_N_KEY
            ],
            device=device,
        ),
        second_joint_translation_u=torch.tensor(
            heliostat_dict[paint_mappings.KINEMATIC_PROPERTIES_KEY][
                paint_mappings.SECOND_JOINT_TRANSLATION_U_KEY
            ],
            device=device,
        ),
        second_joint_tilt_e=torch.tensor(0.0, device=device),
        second_joint_tilt_n=torch.tensor(0.0, device=device),
        concentrator_translation_e=torch.tensor(
            heliostat_dict[paint_mappings.KINEMATIC_PROPERTIES_KEY][
                paint_mappings.CONCENTRATOR_TRANSLATION_E_KEY
            ],
            device=device,
        ),
        concentrator_translation_n=torch.tensor(
            heliostat_dict[paint_mappings.KINEMATIC_PROPERTIES_KEY][
                paint_mappings.CONCENTRATOR_TRANSLATION_N_KEY
            ],
            device=device,
        ),
        concentrator_translation_u=torch.tensor(
            heliostat_dict[paint_mappings.KINEMATIC_PROPERTIES_KEY][
                paint_mappings.CONCENTRATOR_TRANSLATION_U_KEY
            ],
            device=device,
        ),
    )

    # Include the initial orientation for the kinematics.
    initial_orientation = utils.convert_3d_directions_to_4d_format(
        torch.tensor(
            heliostat_dict[paint_mappings.INITIAL_ORIENTATION_KEY],
            device=device,
        ),
        device=device,
    )

    paint_actuators = list(
        heliostat_dict[paint_mappings.KINEMATIC_PROPERTIES_KEY][
            paint_mappings.ACTUATOR_KEY
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
        kinematics_deviations,
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

    log.info("Beginning extraction of deflectometry data from PAINT file.")

    with h5py.File(heliostat_deflectometry_path, "r") as file:
        surface_points_with_facets_list = []
        surface_normals_with_facets_list = []
        for f in range(number_of_facets):
            number_of_points = len(
                file[f"{paint_mappings.FACET_KEY}{f + 1}"][
                    paint_mappings.SURFACE_POINT_KEY
                ]
            )
            single_facet_surface_points = torch.empty(
                number_of_points, 3, device=device
            )
            single_facet_surface_normals = torch.empty(
                number_of_points, 3, device=device
            )

            points_data = torch.tensor(
                file[f"{paint_mappings.FACET_KEY}{f + 1}"][
                    paint_mappings.SURFACE_POINT_KEY
                ][()],
                device=device,
            )
            normals_data = torch.tensor(
                file[f"{paint_mappings.FACET_KEY}{f + 1}"][
                    paint_mappings.SURFACE_NORMAL_KEY
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
        A tuple containing the heliostat name, path to the properties file, and path to deflectometry data file.
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
        i.e., for ideal surfaces only the heliostat name and path to the properties file is required, whilst for the
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
    prototype_kinematics = None
    prototype_actuator_list = None
    heliostat_config_list = []

    for heliostat_index, file_tuple in enumerate(paths):
        # Extract common heliostat properties.
        (
            heliostat_position,
            facet_translation_vectors,
            canting,
            kinematics_deviations,
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

        kinematics_config = KinematicsConfig(
            type=config_dictionary.rigid_body_key,
            initial_orientation=initial_orientation,
            deviations=kinematics_deviations,
        )
        prototype_kinematics = kinematics_config

        actuator_list = []
        for actuator_index, actuator_parameters_tuple in enumerate(
            actuator_parameters_list
        ):
            actuator = ActuatorConfig(
                key=f"{config_dictionary.heliostat_actuator_key}_{actuator_index}",
                type=str(actuator_parameters_tuple[index_mapping.paint_actuator_type]),
                clockwise_axis_movement=bool(
                    actuator_parameters_tuple[
                        index_mapping.paint_actuator_clockwise_axis_movement
                    ]
                ),
                min_max_motor_positions=cast(
                    List[float],
                    actuator_parameters_tuple[
                        index_mapping.paint_actuator_min_max_motor_positions
                    ],
                ),
                parameters=cast(
                    ActuatorParameters,
                    actuator_parameters_tuple[index_mapping.paint_actuator_parameters],
                ),
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
            kinematics=kinematics_config,
            actuators=actuators_list_config,
        )
        heliostat_config_list.append(heliostat_config)

        # Create the prototype configuration.
        surface_prototype_config = SurfacePrototypeConfig(
            facet_list=prototype_surface.facet_list
        )
        kinematics_prototype_config = KinematicsPrototypeConfig(
            type=prototype_kinematics.type,
            initial_orientation=prototype_kinematics.initial_orientation,
            deviations=prototype_kinematics.deviations,
        )
        actuator_prototype_config = ActuatorPrototypeConfig(
            actuator_list=prototype_actuator_list
        )

    prototype_config = PrototypeConfig(
        surface_prototype=surface_prototype_config,
        kinematics_prototype=kinematics_prototype_config,
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
        If None, ``ARTIST`` will automatically select the most appropriate
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
        If None, ``ARTIST`` will automatically select the most appropriate
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


def extract_paint_heliostats_mixed_surface(
    paths: (
        list[tuple[str, pathlib.Path]] | list[tuple[str, pathlib.Path, pathlib.Path]]
    ),
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
    Extract heliostat data with a mix of ideal and fitted surfaces from PAINT files.

    This function processes a list of heliostat file paths. If a deflectometry path is provided for a heliostat, a
    fitted surface is generated. Otherwise, an ideal surface is used.

    Parameters
    ----------
    paths : list[tuple[str, pathlib.Path, pathlib.Path]] | list[tuple[str, pathlib.Path, pathlib.Path, pathlib.Path]]
        A list where each tuple contains the heliostat name, path to the properties file, and an optional path to the
        deflectometry file.
    power_plant_position : torch.Tensor
        The position of the power plant in latitude, longitude, and elevation.
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
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    HeliostatListConfig
        The configuration of all heliostats in the scenario.
    PrototypeConfig
        The configuration for a heliostat prototype. This is always based on an ideal surface.
    """
    device = get_device(device=device)

    fitted_paths = []
    ideal_paths = []

    for path_tuple in paths:
        # Check if the third element (deflectometry path) is a valid path.
        if len(path_tuple) == 3 and path_tuple[2] is not None:
            fitted_paths.append(path_tuple)
        else:
            ideal_paths.append(path_tuple)

    # Process ideal heliostats.
    ideal_heliostats_config, prototype_config = _process_heliostats_from_paths(
        paths=ideal_paths,
        power_plant_position=power_plant_position,
        number_of_nurbs_control_points=number_of_nurbs_control_points,
        surface_config_generator=_ideal_surface_generator,
        device=device,
    )

    # Process fitted heliostats.
    fitted_heliostats_config, _ = _process_heliostats_from_paths(
        paths=fitted_paths,
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

    # Combine the lists.
    combined_heliostat_list = (
        ideal_heliostats_config.heliostat_list + fitted_heliostats_config.heliostat_list
    )
    combined_heliostat_list_config = HeliostatListConfig(
        heliostat_list=combined_heliostat_list
    )

    return combined_heliostat_list_config, prototype_config


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

    It assumes that the data directory has the same structure and file names the ``PAINT`` database.
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
        calibration_dir = base / name / paint_mappings.SAVE_CALIBRATION
        if not calibration_dir.exists():
            log.warning(f"Calibration directory for {name} not found.")
            continue

        property_files = list(
            calibration_dir.glob(f"*{paint_mappings.CALIBRATION_PROPERTIES_IDENTIFIER}")
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
