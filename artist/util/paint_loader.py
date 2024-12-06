import json
import pathlib

import torch

from artist.util import config_dictionary, utils
from artist.util.configuration_classes import (
    ActuatorConfig,
    ActuatorListConfig,
    ActuatorParameters,
    KinematicDeviations,
)


def read_paint_calibration_properties(calibration_properties_path: pathlib.Path):
    with open(calibration_properties_path, "r") as file:
        calibration_dict = json.load(file)
        calibration_target_name = calibration_dict[
            config_dictionary.paint_calibration_traget
        ]

    return calibration_target_name


def read_paint_tower_measurements(
    tower_measurements_path: pathlib.Path, target_name: str, device: torch.device
):
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


def read_paint_heliostat_properties(
    heliostat_properties_path: pathlib.Path,
    power_plant_position: torch.Tensor,
    device: torch.device,
):
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

    actuators = list(
        heliostat_dict[config_dictionary.paint_kinematic][
            config_dictionary.paint_actuators
        ][()]
    )

    for actuator in actuators:
        actuator1_parameters = ActuatorParameters(
            increment=torch.tensor(
                heliostat_dict[config_dictionary.paint_kinematic][
                    f"{config_dictionary.paint_increment}_{index}"
                ],
                device=device,
            ),
            initial_stroke_length=torch.tensor(
                heliostat_dict[config_dictionary.paint_kinematic][
                    f"{config_dictionary.paint_initial_stroke_length}_{index}"
                ],
                device=device,
            ),
            offset=torch.tensor(
                heliostat_dict[config_dictionary.paint_kinematic][
                    f"{config_dictionary.paint_offset}_{index}"
                ],
                device=device,
            ),
            pivot_radius=torch.tensor(
                heliostat_dict[config_dictionary.paint_kinematic][
                    f"{config_dictionary.paint_pivot_radius}_{index}"
                ],
                device=device,
            ),
            initial_angle=torch.tensor(
                heliostat_dict[config_dictionary.paint_kinematic][
                    f"{config_dictionary.paint_initial_angle}_{index}"
                ],
                device=device,
            ),
        )
    # Include an actuator 1.
    actuator1 = ActuatorConfig(
        actuator=f"{config_dictionary.actuator}_{index}",
        actuator_type=heliostat_dict[config_dictionary.paint_heliostat_kinematic][
            f"{config_dictionary.paint_actuator_type}_{index}"
        ].lower(),
        actuator_clockwise=heliostat_dict[config_dictionary.paint_heliostat_kinematic][
            f"{config_dictionary.paint_clockwise}_{index}"
        ],
        actuator_parameters=actuator1_parameters,
    )

    index = 2
    actuator2_parameters = ActuatorParameters(
        increment=torch.tensor(
            heliostat_dict[config_dictionary.paint_heliostat_kinematic][
                f"{config_dictionary.paint_increment}_{index}"
            ],
            device=device,
        ),
        initial_stroke_length=torch.tensor(
            heliostat_dict[config_dictionary.paint_heliostat_kinematic][
                f"{config_dictionary.paint_initial_stroke_length}_{index}"
            ],
            device=device,
        ),
        offset=torch.tensor(
            heliostat_dict[config_dictionary.paint_heliostat_kinematic][
                f"{config_dictionary.paint_offset}_{index}"
            ],
            device=device,
        ),
        pivot_radius=torch.tensor(
            heliostat_dict[config_dictionary.paint_heliostat_kinematic][
                f"{config_dictionary.paint_pivot_radius}_{index}"
            ],
            device=device,
        ),
        initial_angle=torch.tensor(
            heliostat_dict[config_dictionary.paint_heliostat_kinematic][
                f"{config_dictionary.paint_initial_angle}_{index}"
            ],
            device=device,
        ),
    )
    # Include an actuator 1.
    actuator2 = ActuatorConfig(
        actuator=f"{config_dictionary.actuator}_{index}",
        actuator_type=heliostat_dict[config_dictionary.paint_heliostat_kinematic][
            f"{config_dictionary.paint_actuator_type}_{index}"
        ].lower(),
        actuator_clockwise=heliostat_dict[config_dictionary.paint_heliostat_kinematic][
            f"{config_dictionary.paint_clockwise}_{index}"
        ],
        actuator_parameters=actuator2_parameters,
    )

    # Create a list of actuators.
    actuator_list = [actuator1, actuator2]

    # Include the actuator prototype config.
    actuators_list_config = ActuatorListConfig(actuator_list=actuator_list)

    return heliostat_position, kinematic_deviations, actuators_list_config
