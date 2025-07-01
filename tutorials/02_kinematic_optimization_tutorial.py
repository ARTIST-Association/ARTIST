import pathlib

import h5py
import torch

from artist.core.kinematic_optimizer import KinematicOptimizer
from artist.data_loader import paint_loader
from artist.scenario.scenario import Scenario
from artist.util import set_logger_config
from artist.util.environment_setup import get_device

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set up logger
set_logger_config()

# Set the device
device = get_device()

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path(
    "/workVERLEIHNIX/mb/ARTIST/tutorials/data/scenarios/test_scenario_paint_four_heliostats.h5"
)

# Also specify the heliostats to be calibrated and the paths to your calibration-properties.json files.
# Please follow the following style: list[tuple[str, list[pathlib.Path]]]
heliostat_calibration_mapping = [
    (
        "AA39",
        [
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/218385-calibration-properties.json"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/246955-calibration-properties.json"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/275564-calibration-properties.json"
            ),
            # pathlib.Path(
            #     "please/insert/the/path/to/the/paint/data/here/calibration-properties.json"
            # ),
            # ....
        ],
    ),
    (
        "AA31",
        [
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA31/126372-calibration-properties.json"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA31/219988-calibration-properties.json"
            ),
        ],
    ),
    # ...
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
        focal_spots_calibration,
        incident_ray_directions_calibration,
        motor_positions_calibration,
        heliostats_mask_calibration,
        target_area_mask_calibration,
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

    # Select the kinematic parameters to be optimized and calibrated.
    optimizable_parameters = [
        heliostat_group.kinematic.deviation_parameters.requires_grad_(),
        heliostat_group.kinematic.actuators.actuator_parameters.requires_grad_(),
    ]

    # Set up optimizer and scheduler.
    tolerance = 0.0005
    max_epoch = 1000
    initial_learning_rate = 0.0001

    use_ray_tracing = False
    if use_ray_tracing:
        motor_positions_calibration = None
        tolerance = 0.035
        max_epoch = 1000
        initial_learning_rate = 0.0005

    optimizer = torch.optim.Adam(optimizable_parameters, lr=initial_learning_rate)

    # Create the kinematic optimizer.
    kinematic_optimizer = KinematicOptimizer(
        scenario=scenario,
        heliostat_group=heliostat_group,
        optimizer=optimizer,
    )

    # Calibrate the kinematic.
    kinematic_optimizer.optimize(
        focal_spots_calibration=focal_spots_calibration,
        incident_ray_directions=incident_ray_directions_calibration,
        active_heliostats_mask=heliostats_mask_calibration,
        target_area_mask_calibration=target_area_mask_calibration,
        motor_positions_calibration=motor_positions_calibration,
        tolerance=tolerance,
        max_epoch=max_epoch,
        num_log=max_epoch,
        device=device,
    )
