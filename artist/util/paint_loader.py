import json
import pathlib
from typing import Union

import torch

from artist.util import config_dictionary, utils
from artist.util.configuration_classes import (
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
from artist.util.surface_converter import SurfaceConverter


def extract_paint_calibration_data(
    heliostat_calibration_mapping: list[tuple[str, list[pathlib.Path]]],
    power_plant_position: torch.Tensor,
    heliostat_names: list[str],
    target_area_names: list[str],
    device: Union[torch.device, str] = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract calibration data from ``PAINT`` calibration files.

    Parameters
    ----------
    heliostat_calibration_mapping : list[tuple[str, list[pathlib.Path]]]
        The mapping of heliostats and their calibration data files.
    power_plant_position : torch.Tensor
        The power plant position.
    heliostat_names : list[str]
        All possible heliostat names.
    target_area_names : list[str]
        All possible target area names.
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Returns
    -------
    torch.Tensor
        The calibration flux density centers.
    torch.Tensor
        The sun positions.
    torch.Tensor
        The motor positions.
    torch.Tensor
        The heliostat indices in order.
    torch.Tensor
        The target area indices in order.
    """
    device = torch.device(device)

    total_number_of_calibrations = sum(
        len(paths) for _, paths in heliostat_calibration_mapping
    )

    calibration_heliostat_names = []
    calibration_target_names = []
    centers_calibration_images = torch.zeros(
        (total_number_of_calibrations, 4), device=device
    )
    sun_positions_enu = torch.zeros((total_number_of_calibrations, 4), device=device)
    all_motor_positions = torch.zeros((total_number_of_calibrations, 2), device=device)

    index = 0
    for mapping in heliostat_calibration_mapping:
        for path in mapping[1]:
            calibration_heliostat_names.append(mapping[0])
            with open(path, "r") as file:
                calibration_dict = json.load(file)
                calibration_target_names.append(
                    calibration_dict[config_dictionary.paint_calibration_target]
                )
                center_calibration_image = convert_wgs84_coordinates_to_local_enu(
                    torch.tensor(
                        calibration_dict[config_dictionary.paint_focal_spot][
                            config_dictionary.paint_utis
                        ],
                        dtype=torch.float64,
                        device=device,
                    ),
                    power_plant_position,
                    device=device,
                )
                centers_calibration_images[index] = utils.convert_3d_point_to_4d_format(
                    center_calibration_image, device=device
                )
                sun_azimuth = torch.tensor(
                    calibration_dict[config_dictionary.paint_sun_azimuth], device=device
                )
                sun_elevation = torch.tensor(
                    calibration_dict[config_dictionary.paint_sun_elevation],
                    device=device,
                )
                sun_positions_enu[index] = utils.convert_3d_point_to_4d_format(
                    azimuth_elevation_to_enu(
                        sun_azimuth, sun_elevation, degree=True, device=device
                    ),
                    device=device,
                )
                all_motor_positions[index] = torch.tensor(
                    [
                        calibration_dict[config_dictionary.paint_motor_positions][
                            config_dictionary.paint_first_axis
                        ],
                        calibration_dict[config_dictionary.paint_motor_positions][
                            config_dictionary.paint_second_axis
                        ],
                    ],
                    device=device,
                )
            index += 1

    heliostat_index_map = {name: index for index, name in enumerate(heliostat_names)}
    heliostat_indices = torch.tensor(
        [heliostat_index_map[name] for name in calibration_heliostat_names],
        device=device,
    )
    target_area_index_map = {
        target_area_name: index
        for index, target_area_name in enumerate(target_area_names)
    }
    target_area_indices = torch.tensor(
        [
            target_area_index_map[target_area_name]
            for target_area_name in calibration_target_names
        ],
        device=device,
    )

    return (
        centers_calibration_images,
        sun_positions_enu,
        all_motor_positions,
        heliostat_indices,
        target_area_indices,
    )


def extract_paint_tower_measurements(
    tower_measurements_path: pathlib.Path,
    device: Union[torch.device, str] = "cuda",
) -> tuple[PowerPlantConfig, TargetAreaListConfig]:
    """
    Extract tower data from a ``PAINT`` tower measurements file for scenario generation.

    Parameters
    ----------
    tower_measurements_path : pathlib.Path
        The path to the tower measurement file.
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Returns
    -------
    PowerPlantConfig
        The configuration of the power plant.
    TargetAreaListConfig
        The configuration of the tower target areas.
    """
    with open(tower_measurements_path, "r") as file:
        tower_dict = json.load(file)
        power_plant_position = torch.tensor(
            tower_dict[config_dictionary.paint_power_plant_properties][
                config_dictionary.paint_coordinates
            ],
            dtype=torch.float64,
            device=device,
        )

        target_areas = list(tower_dict.keys())[1:]
        target_area_config_list = []

        for target_area in target_areas:
            geometry = tower_dict[target_area][
                config_dictionary.paint_target_area_geometry
            ]

            center_lat_lon = torch.tensor(
                tower_dict[target_area][config_dictionary.paint_coordinates][
                    config_dictionary.paint_center
                ],
                dtype=torch.float64,
                device=device,
            )
            center_3d = convert_wgs84_coordinates_to_local_enu(
                center_lat_lon, power_plant_position, device=device
            )
            center = utils.convert_3d_point_to_4d_format(center_3d, device=device)
            normal_vector = utils.convert_3d_direction_to_4d_format(
                torch.tensor(
                    tower_dict[target_area][config_dictionary.paint_normal_vector],
                    device=device,
                ),
                device=device,
            )

            prefix = ""
            if target_area == config_dictionary.target_area_receiver:
                prefix = "receiver_outer_"
            upper_left = convert_wgs84_coordinates_to_local_enu(
                torch.tensor(
                    tower_dict[target_area][config_dictionary.paint_coordinates][
                        f"{prefix}{config_dictionary.paint_upper_left}"
                    ],
                    dtype=torch.float64,
                    device=device,
                ),
                power_plant_position,
                device=device,
            )
            lower_left = convert_wgs84_coordinates_to_local_enu(
                torch.tensor(
                    tower_dict[target_area][config_dictionary.paint_coordinates][
                        f"{prefix}{config_dictionary.paint_lower_left}"
                    ],
                    dtype=torch.float64,
                    device=device,
                ),
                power_plant_position,
                device=device,
            )
            upper_right = convert_wgs84_coordinates_to_local_enu(
                torch.tensor(
                    tower_dict[target_area][config_dictionary.paint_coordinates][
                        f"{prefix}{config_dictionary.paint_upper_right}"
                    ],
                    dtype=torch.float64,
                    device=device,
                ),
                power_plant_position,
                device=device,
            )
            lower_right = convert_wgs84_coordinates_to_local_enu(
                torch.tensor(
                    tower_dict[target_area][config_dictionary.paint_coordinates][
                        f"{prefix}{config_dictionary.paint_lower_right}"
                    ],
                    dtype=torch.float64,
                    device=device,
                ),
                power_plant_position,
                device=device,
            )
            plane_e, plane_u = corner_points_to_plane(
                upper_left, upper_right, lower_left, lower_right
            )

            tower_area_config = TargetAreaConfig(
                target_area_key=target_area,
                geometry=geometry,
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

    return power_plant_config, target_area_list_config


def extract_paint_heliostats(
    heliostat_and_deflectometry_paths: list[tuple[str, pathlib.Path, pathlib.Path]],
    power_plant_position: torch.Tensor,
    aim_point: torch.Tensor,
    max_epochs_for_surface_training: int = 400,
    device: Union[torch.device, str] = "cuda",
) -> tuple[HeliostatListConfig, PrototypeConfig]:
    """
    Extract heliostat data from ``PAINT`` heliostat properties and deflectometry files.

    Parameters
    ----------
    heliostat_and_deflectometry_paths : tuple[str, pathlib.Path, pathlib.Path]
        Name of the heliostat and a pair of heliostat properties and deflectometry file paths.
    power_plant_position : torch.Tensor
        The position of the power plant in latitude, longitude and elevation.
    aim_point : torch.Tensor
        The default aim point for the heliostats (Should ideally be on a receiver).
    max_epochs_for_surface_training : int
        The maximum amount of epochs for fitting the NURBS (default is 400).
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Returns
    -------
    HeliostatListConfig
        The configuration of all heliostats in the scenario.
    PrototypeConfig
        The configuration for a heliostat prototype.
    """
    prototype_facet_list = None
    prototype_kinematic = None
    prototype_actuator_list = None

    heliostat_config_list = []
    for id, file_tuple in enumerate(heliostat_and_deflectometry_paths):
        with open(file_tuple[1], "r") as file:
            heliostat_dict = json.load(file)
            heliostat_position_3d = convert_wgs84_coordinates_to_local_enu(
                torch.tensor(
                    heliostat_dict[config_dictionary.paint_heliostat_position],
                    dtype=torch.float64,
                    device=device,
                ),
                power_plant_position,
                device=device,
            )
            heliostat_position = utils.convert_3d_point_to_4d_format(
                heliostat_position_3d, device=device
            )

            # Generate surface configuration from data.
            surface_converter = SurfaceConverter(
                step_size=100,
                max_epoch=max_epochs_for_surface_training,
            )

            facet_list = surface_converter.generate_surface_config_from_paint(
                deflectometry_file_path=file_tuple[2],
                heliostat_file_path=file_tuple[1],
                device=device,
            )

            surface_config = SurfaceConfig(facet_list=facet_list)
            prototype_facet_list = facet_list

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
            initial_orientation = utils.convert_3d_direction_to_4d_format(
                torch.tensor(
                    heliostat_dict[config_dictionary.paint_initial_orientation],
                    device=device,
                ),
                device=device,
            )

            # Include the kinematic prototype configuration.
            kinematic_config = KinematicConfig(
                type=config_dictionary.rigid_body_key,
                initial_orientation=initial_orientation,
                deviations=kinematic_deviations,
            )
            prototype_kinematic = kinematic_config

            paint_actuators = list(
                heliostat_dict[config_dictionary.paint_kinematic][
                    config_dictionary.paint_actuators
                ]
            )
            actuator_list = []

            for i, paint_actuator in enumerate(paint_actuators):
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
                actuator = ActuatorConfig(
                    key=f"{config_dictionary.heliostat_actuator_key}_{i}",
                    type=paint_actuator[config_dictionary.paint_actuator_type],
                    clockwise_axis_movement=paint_actuator[
                        config_dictionary.paint_clockwise_axis_movement
                    ],
                    parameters=parameters,
                )
                actuator_list.append(actuator)
            prototype_actuator_list = actuator_list
            actuators_list_config = ActuatorListConfig(actuator_list=actuator_list)

        heliostat_config = HeliostatConfig(
            name=file_tuple[0],
            id=id,
            position=heliostat_position,
            aim_point=aim_point,
            surface=surface_config,
            kinematic=kinematic_config,
            actuators=actuators_list_config,
        )

        heliostat_config_list.append(heliostat_config)

        # Include the configuration for a prototype (Will be extracted from the first heliostat in the list).
        surface_prototype_config = SurfacePrototypeConfig(
            facet_list=prototype_facet_list
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

    # Create the configuration for all heliostats.
    heliostats_list_config = HeliostatListConfig(heliostat_list=heliostat_config_list)

    return heliostats_list_config, prototype_config


def azimuth_elevation_to_enu(
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    slant_range: float = 1.0,
    degree: bool = True,
    device: Union[torch.device, str] = "cuda",
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
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Returns
    -------
    torch.Tensor
        The east, north, up (ENU) coordinates.
    """
    if degree:
        elevation = torch.deg2rad(elevation)
        azimuth = torch.deg2rad(azimuth)

    if azimuth < 0.0:
        azimuth += torch.pi * 2

    r = slant_range * torch.cos(elevation)

    enu = torch.zeros(3, device=device)

    enu[0] = r * torch.sin(azimuth)
    enu[1] = -r * torch.cos(azimuth)
    enu[2] = slant_range * torch.sin(elevation)

    return enu


def convert_wgs84_coordinates_to_local_enu(
    coordinates_to_transform: torch.Tensor,
    reference_point: torch.Tensor,
    device: Union[torch.device, str] = "cuda",
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
    reference_point : torch.Tensor
        The center of origin of the ENU coordinate system in WGS84 coordinates.
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Returns
    -------
    torch.Tensor
        The east offset in meters, north offset in meters, and the altitude difference from the reference point.
    """
    device = torch.device(device)
    wgs84_a = 6378137.0  # Major axis in meters
    wgs84_b = 6356752.314245  # Minor axis in meters
    wgs84_e2 = (wgs84_a**2 - wgs84_b**2) / wgs84_a**2  # Eccentricity squared

    # Convert latitude and longitude to radians.
    lat_rad = torch.deg2rad(coordinates_to_transform[0])
    lon_rad = torch.deg2rad(coordinates_to_transform[1])
    alt = coordinates_to_transform[2] - reference_point[2]
    lat_tower_rad = torch.deg2rad(reference_point[0])
    lon_tower_rad = torch.deg2rad(reference_point[1])

    # Calculate meridional radius of curvature for the first latitude.
    sin_lat1 = torch.sin(lat_rad)
    rn1 = wgs84_a / torch.sqrt(1 - wgs84_e2 * sin_lat1**2)

    # Calculate transverse radius of curvature for the first latitude.
    rm1 = (wgs84_a * (1 - wgs84_e2)) / ((1 - wgs84_e2 * sin_lat1**2) ** 1.5)

    # Calculate delta latitude and delta longitude in radians.
    dlat_rad = lat_tower_rad - lat_rad
    dlon_rad = lon_tower_rad - lon_rad

    # Calculate north and east offsets in meters.
    north_offset_m = dlat_rad * rm1
    east_offset_m = dlon_rad * rn1 * torch.cos(lat_rad)

    return torch.tensor(
        [-east_offset_m, -north_offset_m, alt], dtype=torch.float32, device=device
    )


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
    upper_right : torch.Tensor
        The upper right corner coordinate.
    lower_left : torch.Tensor
        The lower left corner coordinate.
    lower_right : torch.Tensor
        The lower right corner coordinate.

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
