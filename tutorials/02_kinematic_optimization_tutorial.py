import pathlib

import h5py
import torch

from artist.util import paint_loader, set_logger_config
from artist.util.kinematic_optimizer import KinematicOptimizer
from artist.util.scenario import Scenario

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set up logger
set_logger_config()

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path("please/insert/the/path/to/the/scenario/here/scenario.h5")

# Also specify the heliostats to be calibrated and the paths to your calibration-properties.json files.
# Please follow the following style: list[tuple[str, list[pathlib.Path]]]
heliostat_calibration_mapping = [
    (
        "heliostat_name_1",
        [
            pathlib.Path(
                "please/insert/the/path/to/the/calibration/data/here/calibration-properties.json"
            ),
            # pathlib.Path(
            #     "please/insert/the/path/to/the/calibration/data/here/calibration-properties.json"
            # ),
        ],
    ),
    # (
    #     "heliostat_name_2",
    #     [
    #         pathlib.Path(
    #             "please/insert/the/path/to/the/calibration/data/here/calibration-properties.json"
    #         ),
    #         pathlib.Path(
    #             "please/insert/the/path/to/the/calibration/data/here/calibration-properties.json"
    #         ),
    #     ],
    # ),
]

# Load the scenario.
with h5py.File(scenario_path, "r") as scenario_file:
    scenario = Scenario.load_scenario_from_hdf5(
        scenario_file=scenario_file, device=device
    )

for heliostat_group_index, heliostat_group in enumerate(
    scenario.heliostat_field.heliostat_groups
):
    # Load the calibration data.
    (
        centers_calibration_images,
        sun_positions,
        calibration_motor_positions,
        heliostat_indices,
        target_area_indices,
    ) = paint_loader.extract_paint_calibration_data(
        heliostat_calibration_mapping=[
            (heliostat_name, paths)
            for heliostat_name, paths in heliostat_calibration_mapping
            if heliostat_name in heliostat_group.names
        ],
        heliostat_names=heliostat_group.names,
        target_area_names=scenario.target_areas.names,
        power_plant_position=scenario.power_plant_position,
        device=device,
    )

    # Create calibration group
    heliostat_group_class = type(heliostat_group)
    calibration_group = heliostat_group_class(
        names=[heliostat_group.names[i] for i in heliostat_indices.tolist()],
        positions=heliostat_group.positions[heliostat_indices],
        aim_points=scenario.target_areas.centers[target_area_indices],
        surface_points=heliostat_group.surface_points[heliostat_indices],
        surface_normals=heliostat_group.surface_normals[heliostat_indices],
        initial_orientations=heliostat_group.initial_orientations[heliostat_indices],
        kinematic_deviation_parameters=heliostat_group.kinematic_deviation_parameters[
            heliostat_indices
        ],
        actuator_parameters=heliostat_group.actuator_parameters[heliostat_indices],
        device=device,
    )

    # The incident ray direction needs to be normed.
    incident_ray_directions = (
        torch.tensor([0.0, 0.0, 0.0, 1.0], device=device) - sun_positions
    )

    # Select the kinematic parameters to be optimized and calibrated.
    optimizable_parameters = [
        calibration_group.kinematic_deviation_parameters.requires_grad_(),
        calibration_group.actuator_parameters.requires_grad_(),
    ]

    # Set up optimizer and scheduler.
    tolerance = 0.0005
    max_epoch = 1000
    initial_learning_rate = 0.0001

    use_ray_tracing = False
    if use_ray_tracing:
        calibration_motor_positions = None
        tolerance = 0.035
        max_epoch = 10000
        initial_learning_rate = 0.002

    optimizer = torch.optim.Adam(optimizable_parameters, lr=initial_learning_rate)

    # Create the kinematic optimizer.
    kinematic_optimizer = KinematicOptimizer(
        scenario=scenario,
        calibration_group=calibration_group,
        optimizer=optimizer,
    )

    # Calibrate the kinematic.
    calibrated_kinematic_deviation_parameters, calibrated_actuator_parameters = (
        kinematic_optimizer.optimize(
            tolerance=tolerance,
            max_epoch=max_epoch,
            centers_calibration_images=centers_calibration_images,
            incident_ray_directions=incident_ray_directions,
            target_area_indices=target_area_indices,
            motor_positions=calibration_motor_positions,
            num_log=max_epoch,
            device=device,
        )
    )
