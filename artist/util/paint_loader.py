import json
import pathlib
from typing import Union

import torch

from artist.util import config_dictionary, utils
from artist.util.configuration_classes import (
    ActuatorConfig,
    ActuatorListConfig,
    ActuatorParameters,
    KinematicConfig,
    KinematicDeviations,
)


def extract_paint_calibration_properties(
    calibration_properties_path: pathlib.Path,
) -> str:
    """
    Extract calibration data from ```PAINT`` calibration file for scenario generation.

    Parameters
    ----------
    calibration_properties_path : pathlib.Path
        The path to the calibration file.

    Returns
    -------
    str
        The name of the calibration traget name.
    """
    with open(calibration_properties_path, "r") as file:
        calibration_dict = json.load(file)
        calibration_target_name = calibration_dict[
            config_dictionary.paint_calibration_traget
        ]

    return calibration_target_name


def extract_paint_tower_measurements(
    tower_measurements_path: pathlib.Path,
    target_name: str,
    device: Union[torch.device, str] = "cuda",
) -> tuple[torch.Tensor, str, torch.Tensor, torch.Tensor, float, float]:
    """
    Extract tower data from ```PAINT`` tower measurement file for scenario generation.

    Parameters
    ----------
    tower_measurements_path : pathlib.Path
        The path to the tower measurement file.
    target_name : str
        The name of the target plane (receiver or calibration target).
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Returns
    -------
    torch.Tensor
        The position of the power plant in latitude, longitude, elevation.
    str
        The type of the target (i.e. plane).
    torch.Tensor
        The coordinates of the target center in ENU.
    torch.Tensor
        The normal vector of the target plane.
    float
        The dimension of the target plane in east direction (width).
    float
        The dimension of the target plane in up dimension (height).
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
        target_type = tower_dict[target_name][config_dictionary.paint_receiver_type]
        target_center_lat_lon = torch.tensor(
            tower_dict[target_name][config_dictionary.paint_coordinates][
                config_dictionary.paint_center
            ],
            dtype=torch.float64,
            device=device,
        )
        target_center_3d = utils.convert_wgs84_coordinates_to_local_enu(
            target_center_lat_lon, power_plant_position, device=device
        )
        target_center = utils.convert_3d_points_to_4d_format(
            target_center_3d, device=device
        )
        normal_vector = utils.convert_3d_direction_to_4d_format(
            torch.tensor(
                tower_dict[target_name][config_dictionary.paint_normal_vector],
                device=device,
            ),
            device=device,
        )
        upper_left = utils.convert_wgs84_coordinates_to_local_enu(
            torch.tensor(
                tower_dict[target_name][config_dictionary.paint_coordinates][
                    config_dictionary.paint_upper_left
                ],
                dtype=torch.float64,
                device=device,
            ),
            power_plant_position,
            device=device,
        )
        lower_left = utils.convert_wgs84_coordinates_to_local_enu(
            torch.tensor(
                tower_dict[target_name][config_dictionary.paint_coordinates][
                    config_dictionary.paint_lower_left
                ],
                dtype=torch.float64,
                device=device,
            ),
            power_plant_position,
            device=device,
        )
        upper_right = utils.convert_wgs84_coordinates_to_local_enu(
            torch.tensor(
                tower_dict[target_name][config_dictionary.paint_coordinates][
                    config_dictionary.paint_upper_right
                ],
                dtype=torch.float64,
                device=device,
            ),
            power_plant_position,
            device=device,
        )
        lower_right = utils.convert_wgs84_coordinates_to_local_enu(
            torch.tensor(
                tower_dict[target_name][config_dictionary.paint_coordinates][
                    config_dictionary.paint_lower_right
                ],
                dtype=torch.float64,
                device=device,
            ),
            power_plant_position,
            device=device,
        )
        plane_e, plane_u = utils.corner_points_to_plane(
            upper_left, upper_right, lower_left, lower_right
        )

    return (
        power_plant_position,
        target_type,
        target_center,
        normal_vector,
        plane_e,
        plane_u,
    )


def extract_paint_heliostat_properties(
    heliostat_properties_path: pathlib.Path,
    power_plant_position: torch.Tensor,
    device: Union[torch.device, str] = "cuda",
) -> tuple[torch.Tensor, KinematicConfig, ActuatorListConfig]:
    """
    Extract heliostat data from ```PAINT`` heliostat file for scenario generation.

    Parameters
    ----------
    heliostat_properties_path : pathlib.Path
        The path to the heliostat file.
    power_plant_position : str
        The position of the power plant in latitude, longitude and elevation.
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Returns
    -------
    torch.Tensor
        The position of the heliostat in ENU.
    KinematicConfig
        The kinematic configuration including type, initial orientation and deviations.
    ActuatorListConfig
        Configuration for multiple actuators with individual parameters.
    """
    with open(heliostat_properties_path, "r") as file:
        heliostat_dict = json.load(file)
        heliostat_position_3d = utils.convert_wgs84_coordinates_to_local_enu(
            torch.tensor(
                heliostat_dict[config_dictionary.paint_heliostat_position],
                dtype=torch.float64,
                device=device,
            ),
            power_plant_position,
            device=device,
        )
        heliostat_position = utils.convert_3d_points_to_4d_format(
            heliostat_position_3d, device=device
        )

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
                paint_actuator[config_dictionary.paint_pivot_radius], device=device
            ),
            initial_angle=torch.tensor(
                paint_actuator[config_dictionary.paint_initial_angle], device=device
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

    actuators_list_config = ActuatorListConfig(actuator_list=actuator_list)

    return heliostat_position, kinematic_config, actuators_list_config
