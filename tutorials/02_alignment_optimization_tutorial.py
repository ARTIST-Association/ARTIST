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

# Also specify the path to your calibration-properties.json file.
calibration_properties_paths = [
    pathlib.Path(
        "please/insert/the/path/to/the/calibration/data/here/calibration-properties.json"
    ),
    # pathlib.Path(
    #     "please/insert/the/path/to/the/calibration/data/here/calibration-properties.json"
    # )
]

# Load the scenario.
with h5py.File(scenario_path, "r") as scenario_file:
    scenario = Scenario.load_scenario_from_hdf5(
        scenario_file=scenario_file, device=device
    )

# Load the calibration data.
(
    calibration_target_names,
    center_calibration_images,
    sun_positions,
    all_calibration_motor_positions,
) = paint_loader.extract_paint_calibration_data(
    calibration_properties_paths=calibration_properties_paths,
    power_plant_position=scenario.power_plant_position,
    device=device,
)

# The incident ray direction needs to be normed.
incident_ray_directions = (
    torch.tensor([0.0, 0.0, 0.0, 1.0], device=device) - sun_positions
)

# Create a calibration scenario from the original scenario.
# It contains a single helisotat, chosen by its index.
calibration_scenario = scenario.create_calibration_scenario(
    heliostat_index=2, device=device
)

# Select the kinematic parameters to be optimzed and calibrated.
optimizable_parameters = [
    calibration_scenario.heliostat_field.all_kinematic_deviation_parameters.requires_grad_(),
    calibration_scenario.heliostat_field.all_actuator_parameters.requires_grad_(),
]

# Set up optimizer and scheduler.
tolerance = 1e-7
max_epoch = 150
initial_learning_rate = 0.01
learning_rate_factor = 0.1
learning_rate_patience = 20
learning_rate_threshold = 0.1

use_raytracing = False
if use_raytracing:
    all_calibration_motor_positions = None
    tolerance = 1e-7
    max_epoch = 27
    initial_learning_rate = 0.0002
    learning_rate_factor = 0.1
    learning_rate_patience = 18
    learning_rate_threshold = 0.1

optimizer = torch.optim.Adam(optimizable_parameters, lr=initial_learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=learning_rate_factor,
    patience=learning_rate_patience,
    threshold=learning_rate_threshold,
    threshold_mode="abs",
)

# Create the kinematic optimizer.
kinematic_optimizer = KinematicOptimizer(
    scenario=calibration_scenario,
    optimizer=optimizer,
    scheduler=scheduler,
)

# Calibrate the kinematic.
kinematic_optimizer.optimize(
    tolerance=tolerance,
    max_epoch=max_epoch,
    center_calibration_images=center_calibration_images,
    incident_ray_directions=incident_ray_directions,
    calibration_target_names=calibration_target_names,
    motor_positions=all_calibration_motor_positions,
    num_log=max_epoch,
    device=device,
)
