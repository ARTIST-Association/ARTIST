import pathlib

from artist.raytracing.heliostat_tracing import HeliostatRayTracer
import h5py
import torch

from artist import ARTIST_ROOT
from artist.scenario import Scenario
from artist.util import paint_loader, set_logger_config, utils
from artist.util.alignment_optimizer import AlignmentOptimizer

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set up logger
set_logger_config()

# Set the device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# The distributed environment is setup and destroyed using a Generator object.
environment_generator = utils.setup_distributed_environment(device=device)

is_distributed, rank, world_size = next(environment_generator)

if device.type == "cuda":
    torch.cuda.set_device(rank % torch.cuda.device_count())

# If you have already generated the tutorial scenario yourself, you can leave this boolean as False. If not, set it to
# true and a pre-generated scenario file will be used for this tutorial!
use_pre_generated_scenario = True
scenario_path = (
    pathlib.Path(ARTIST_ROOT) / "please/insert/the/path/to/the/scenario/here/name.h5"
)
if use_pre_generated_scenario:
    scenario_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tutorials/data/test_scenario_paint_single_heliostat.h5"
    )

# Load the scenario.
with h5py.File(scenario_path, "r") as scenario_file:
    example_scenario = Scenario.load_scenario_from_hdf5(
        scenario_file=scenario_file, device=device
    )

# Set calibration data
calibration_properties_path = (
    pathlib.Path(ARTIST_ROOT) / "tutorials/data/calibration-properties.json"
)

# Load the calibration data.
calibration_target_name, center_calibration_image, incident_ray_direction, motor_positions = (
    paint_loader.extract_paint_calibration_data(
        calibration_properties_path=calibration_properties_path,
        power_plant_position=example_scenario.power_plant_position,
        device=device,
    )
)

# Get optimizable parameters. This will select all 28 kinematic parameters.
parameters = utils.get_rigid_body_kinematic_parameters_from_scenario(
    kinematic=example_scenario.heliostats.heliostat_list[0].kinematic
)

# Set up optimizer and scheduler parameters
tolerance = 1e-7
max_epoch = 150
initial_learning_rate = 0.01
learning_rate_factor = 0.1
learning_rate_patience = 20
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

# Create distributed alignment optimizer
alignment_optimizer = AlignmentOptimizer(
    scenario=example_scenario,
    optimizer=optimizer,
    scheduler=scheduler,
    world_size=world_size,
    rank=rank,
    batch_size=1000,
    is_distributed=is_distributed,
)

optimized_parameters, optimized_scenario = alignment_optimizer.optimize(
    tolerance=tolerance,
    max_epoch=max_epoch,
    center_calibration_image=center_calibration_image,
    calibration_target_name=calibration_target_name,
    incident_ray_direction=incident_ray_direction,
    motor_positions=motor_positions,
    device=device,
)


# Create raytracer
raytracer = HeliostatRayTracer(
    scenario=optimized_scenario,
    world_size=world_size,
    rank=rank,
    batch_size=100,
)

# Perform heliostat-based raytracing.
final_bitmap = raytracer.trace_rays(
    incident_ray_direction=incident_ray_direction, device=device
)

if is_distributed:
    final_bitmap = torch.distributed.all_reduce(
        final_bitmap, op=torch.distributed.ReduceOp.SUM
    )

final_bitmap = raytracer.normalize_bitmap(final_bitmap)

# Make sure the code after the yield statement in the environment Generator
# is called, to clean up the distributed process group.
try:
    next(environment_generator)
except StopIteration:
    pass