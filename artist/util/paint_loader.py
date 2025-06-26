import json
import pathlib
from collections import Counter, defaultdict

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
from artist.util.environment_setup import get_device
from artist.util.surface_converter import SurfaceConverter


def extract_paint_calibration_data(
    heliostat_calibration_mapping: list[tuple[str, list[pathlib.Path]]],
    power_plant_position: torch.Tensor,
    heliostat_names: list[str],
    target_area_names: list[str],
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
    heliostat_names : list[str]
        All possible heliostat names.
    target_area_names : list[str]
        All possible target area names.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA, MPS, or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The calibration focal spots.
    torch.Tensor
        The light source positions.
    torch.Tensor
        The motor positions.
    torch.Tensor
        A mask with active heliostats and their replications.
    torch.Tensor
        The target area mapping for the heliostats.
    """
    device = get_device(device=device)

    target_indices = {name: index for index, name in enumerate(target_area_names)}

    # Gather calibration data
    replication_counter: Counter[str] = Counter()
    calibration_data_per_heliostat = defaultdict(list)

    for heliostat_name, paths in heliostat_calibration_mapping:
        for path in paths:
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
                        config_dictionary.paint_utis
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
    focal_spots = utils.convert_3d_point_to_4d_format(focal_spots_enu, device=device)

    light_source_positions_enu = azimuth_elevation_to_enu(
        azimuths, elevations, degree=True, device=device
    )
    light_source_positions = utils.convert_3d_point_to_4d_format(
        light_source_positions_enu, device=device
    )
    incident_ray_directions = (
        torch.tensor([0.0, 0.0, 0.0, 1.0], device=device) - light_source_positions
    )

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
        If None, ARTIST will automatically select the most appropriate
        device (CUDA, MPS, or CPU) based on availability and OS.

    Returns
    -------
    PowerPlantConfig
        The configuration of the power plant.
    TargetAreaListConfig
        The configuration of the tower target areas.
    """
    device = get_device(device=device)

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
        center = utils.convert_3d_point_to_4d_format(center_enu[0], device=device)

        normal_vector = utils.convert_3d_direction_to_4d_format(
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

    return power_plant_config, target_area_list_config


def extract_paint_heliostats(
    heliostat_and_deflectometry_paths: list[tuple[str, pathlib.Path, pathlib.Path]],
    power_plant_position: torch.Tensor,
    aim_point: torch.Tensor,
    max_epochs_for_surface_training: int = 400,
    device: torch.device | None = None,
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
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA, MPS, or CPU) based on availability and OS.

    Returns
    -------
    HeliostatListConfig
        The configuration of all heliostats in the scenario.
    PrototypeConfig
        The configuration for a heliostat prototype.
    """
    device = get_device(device=device)

    prototype_facet_list = None
    prototype_kinematic = None
    prototype_actuator_list = None

    heliostat_config_list = []
    for id, file_tuple in enumerate(heliostat_and_deflectometry_paths):
        with open(file_tuple[1], "r") as file:
            heliostat_dict = json.load(file)
            heliostat_position_3d = convert_wgs84_coordinates_to_local_enu(
                torch.tensor(
                    [heliostat_dict[config_dictionary.paint_heliostat_position]],
                    dtype=torch.float64,
                    device=device,
                ),
                power_plant_position,
                device=device,
            )
            heliostat_position = utils.convert_3d_point_to_4d_format(
                heliostat_position_3d[0], device=device
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
        If None, ARTIST will automatically select the most appropriate
        device (CUDA, MPS, or CPU) based on availability and OS.

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
    device: Optional[torch.device] = None,
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
    device : Optional[torch.device]
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA, MPS, or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The east offsets in meters, norths offset in meters, and the altitude differences from the reference point.
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
    latitude_refernce_point = torch.deg2rad(reference_point[0])
    longitude_reference_point = torch.deg2rad(reference_point[1])

    # Calculate meridional radius of curvature for the first latitude.
    sin_lat1 = torch.sin(latitudes)
    rn1 = wgs84_a / torch.sqrt(1 - wgs84_e2 * sin_lat1**2)

    # Calculate transverse radius of curvature for the first latitude.
    rm1 = (wgs84_a * (1 - wgs84_e2)) / ((1 - wgs84_e2 * sin_lat1**2) ** 1.5)

    # Calculate delta latitude and delta longitude in radians.
    dlat_rad = latitude_refernce_point - latitudes
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
