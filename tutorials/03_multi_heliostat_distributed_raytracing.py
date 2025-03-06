import pathlib
import time

import h5py
import torch

from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util.scenario import Scenario
from artist.util import paint_loader, set_logger_config, utils

from matplotlib import pyplot as plt

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set up logger
set_logger_config()

# Set the device
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path(
    "/workVERLEIHNIX/mb/ARTIST/tutorials/data/scenarios/test_scenario_paint_multi_heliostat_10000_4.h5"
)

# Also specify the path to your calibration-properties.json file, used only to retrieve realisitc sun position.
calibration_properties_path = pathlib.Path(
    "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/calibration-properties.json"
)

# The distributed environment is setup and destroyed using a Generator object.
environment_generator = utils.setup_distributed_environment(device=device)

device, is_distributed, rank, world_size = next(environment_generator)

# Load the scenario.
with h5py.File(scenario_path) as scenario_file:
    scenario = Scenario.load_scenario_from_hdf5(
        scenario_file=scenario_file, device=device
    )

# Load the incident_ray_direction from the calibration data.
(
    _,
    _,
    sun_position,
    _,
) = paint_loader.extract_paint_calibration_data(
    calibration_properties_path=calibration_properties_path,
    power_plant_position=scenario.power_plant_position,
    device=device,
)

# Incident ray direction needs to be normed
incident_ray_direction = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device) - sun_position
#incident_ray_direction = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)

scenario.heliostat_field.number_of_heliostats = scenario.heliostat_field.number_of_heliostats * 500
scenario.heliostat_field.all_heliostat_positions = scenario.heliostat_field.all_heliostat_positions.repeat(500, 1)
scenario.heliostat_field.all_aim_points = scenario.heliostat_field.all_aim_points.repeat(500, 1)
scenario.heliostat_field.all_surface_points = scenario.heliostat_field.all_surface_points.repeat(500, 1, 1)
scenario.heliostat_field.all_surface_normals = scenario.heliostat_field.all_surface_normals.repeat(500, 1, 1)
scenario.heliostat_field.all_initial_orientations = scenario.heliostat_field.all_initial_orientations.repeat(500, 1)
scenario.heliostat_field.all_kinematic_deviation_parameters = scenario.heliostat_field.all_kinematic_deviation_parameters.repeat(500, 1)
scenario.heliostat_field.all_actuator_parameters = scenario.heliostat_field.all_actuator_parameters.repeat(500, 1, 1)
scenario.heliostat_field.all_aligned_heliostats = scenario.heliostat_field.all_aligned_heliostats.repeat(500)
scenario.heliostat_field.all_preferred_reflection_directions = scenario.heliostat_field.all_preferred_reflection_directions.repeat(500, 1)
scenario.heliostat_field.all_current_aligned_surface_points = scenario.heliostat_field.all_current_aligned_surface_points.repeat(500, 1, 1)
scenario.heliostat_field.all_current_aligned_surface_normals = scenario.heliostat_field.all_current_aligned_surface_normals.repeat(500, 1, 1)

start_time = time.time()
# Align all heliostats
scenario.heliostat_field.align_surfaces_with_incident_ray_direction(
    incident_ray_direction=incident_ray_direction,
    device=device
)

# Create raytracer
raytracer = HeliostatRayTracer(
    scenario=scenario, world_size=world_size, rank=rank, batch_size=500, random_seed=rank
)

# Perform heliostat-based raytracing.
final_bitmap = raytracer.trace_rays(
    incident_ray_direction=incident_ray_direction,
    target_area=scenario.get_target_area("receiver"),
    device=device
)

plt.imshow(final_bitmap.cpu().detach(), cmap="inferno")
plt.title(f"Flux Density Distribution from rank (heliostat): {rank}")
plt.savefig(f"AC_new_rank_{rank}_{device.type}.png")

if is_distributed:
    torch.distributed.all_reduce(final_bitmap, op=torch.distributed.ReduceOp.SUM)

end_time = time.time()
print(f"Alignment and raytracing, heliostats: 2000, {device}, time: {end_time-start_time}")

#final_bitmap = raytracer.normalize_bitmap(final_bitmap, aimpoint_area)

plt.imshow(final_bitmap.cpu().detach(), cmap="inferno")
plt.title("Total Flux Density Distribution")
plt.savefig(f"AC_new_final_single_device_mode_{device.type}.png")

# Make sure the code after the yield statement in the environment Generator
# is called, to clean up the distributed process group.
try:
    next(environment_generator)
except StopIteration:
    pass
