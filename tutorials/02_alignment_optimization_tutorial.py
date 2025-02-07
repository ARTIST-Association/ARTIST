import pathlib

import h5py
import torch

from artist.scenario import Scenario
from artist.util import paint_loader, set_logger_config, utils
from artist.util.alignment_optimizer import AlignmentOptimizer

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set up logger
set_logger_config()

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path("please/insert/the/path/to/the/scenario/here/name.h5")

# Also specify the path to your calibration-properties.json file.
calibration_properties_path = pathlib.Path(
    "please/insert/the/path/to/the/calibration/properties/here/calibration-properties.json"
)

# Load the scenario.
with h5py.File(scenario_path, "r") as scenario_file:
    example_scenario = Scenario.load_scenario_from_hdf5(
        scenario_file=scenario_file, device=device
    )

# Load the calibration data.
(
    calibration_target_name,
    center_calibration_image,
    incident_ray_direction,
    motor_positions,
) = paint_loader.extract_paint_calibration_data(
    calibration_properties_path=calibration_properties_path,
    power_plant_position=example_scenario.power_plant_position,
    device=device,
)

# Get optimizable parameters. This will select all 28 kinematic parameters.
parameters = utils.get_rigid_body_kinematic_parameters_from_scenario(
    kinematic=example_scenario.heliostats.heliostat_list[0].kinematic
)

# Set up optimizer and scheduler parameters.
tolerance = 1e-7
max_epoch = 150
initial_learning_rate = 0.01
learning_rate_factor = 0.1
learning_rate_patience = 20
learning_rate_threshold = 0.1

use_raytracing = False
if use_raytracing:
    motor_positions = None
    tolerance = 1e-7
    max_epoch = 27
    initial_learning_rate = 0.0002
    learning_rate_factor = 0.1
    learning_rate_patience = 18
    learning_rate_threshold = 0.1

optimizer = torch.optim.Adam(parameters, lr=initial_learning_rate)

# Set up learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=learning_rate_factor,
    patience=learning_rate_patience,
    threshold=learning_rate_threshold,
    threshold_mode="abs",
)

# Create alignment optimizer.
alignment_optimizer = AlignmentOptimizer(
    scenario=example_scenario,
    optimizer=optimizer,
    scheduler=scheduler,
)

optimized_parameters, optimized_scenario = alignment_optimizer.optimize(
    tolerance=tolerance,
    max_epoch=max_epoch,
    center_calibration_image=center_calibration_image,
    incident_ray_direction=incident_ray_direction,
    motor_positions=motor_positions,
    device=device,
)
