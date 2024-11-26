import pathlib

import h5py
import torch

from artist import ARTIST_ROOT
from artist.scenario import Scenario
from artist.util import set_logger_config, utils
from artist.util.alignment_optimizer import AlignmentOptimizer

# If you have already generated the tutorial scenario yourself, you can leave this boolean as False. If not, set it to
# true and a pre-generated scenario file will be used for this tutorial!
use_pre_generated_scenario = True
scenario_path = "please/insert/the/path/to/the/scenario/here/name.h5"
if use_pre_generated_scenario:
    scenario_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tutorials/data/test_scenario_alignment_optimization.h5"
    )

# Set up logger
set_logger_config()

# Set the device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Load the scenario.
with h5py.File(scenario_path, "r") as scenario_file:
    example_scenario = Scenario.load_scenario_from_hdf5(
        scenario_file=scenario_file, device=device
    )

# Get optimizable parameters. (This will choose all 28 kinematic parameters)
parameters = utils.get_rigid_body_kinematic_parameters_from_scenario(
    scenario=example_scenario
)

# Set up optimizer
optimizer = torch.optim.Adam(parameters, lr=0.001)

# Set up learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.1,
    patience=20,
    threshold=0.1,
    threshold_mode="abs",
)

# Choose calibration data
calibration_properties_path = (
    pathlib.Path(ARTIST_ROOT) / "tutorials/data/test_calibration_properties.json"
)

# Load the calibration data
center_calibration_image, incident_ray_direction, motor_positions = (
    utils.get_calibration_properties(
        calibration_properties_path=calibration_properties_path, device=device
    )
)

# Create alignment optimizer
alignment_optimizer = AlignmentOptimizer(
    scenario=example_scenario,
    optimizer=optimizer,
    scheduler=scheduler,
)

# Optimize kinematic parameters
# In this example motor positions are provided.
# Without motor positions the optimizer would use raytracing.
optimized_parameters, optimized_scenario = alignment_optimizer.optimize(
    tolerance=1e-7,
    max_epoch=150,
    center_calibration_image=center_calibration_image,
    incident_ray_direction=incident_ray_direction,
    motor_positions=motor_positions,
    device=device,
)
