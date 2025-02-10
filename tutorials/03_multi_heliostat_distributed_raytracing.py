import pathlib
import time

import h5py
import torch

from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.scenario import Scenario
from artist.util import paint_loader, set_logger_config, utils

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set up logger
set_logger_config()

# Set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    incident_ray_direction,
    _,
) = paint_loader.extract_paint_calibration_data(
    calibration_properties_path=calibration_properties_path,
    power_plant_position=scenario.power_plant_position,
    device=device,
)

# Align all heliostats
for i in range(2000):
    scenario.heliostats.heliostat_list[
        0
    ].set_aligned_surface_with_incident_ray_direction(
        incident_ray_direction=incident_ray_direction, device=device
    )


start_time = time.time()

# Create raytracer
raytracer = HeliostatRayTracer(
    scenario=scenario, world_size=world_size, rank=rank, batch_size=1, random_seed=rank
)

# Perform heliostat-based raytracing.
final_bitmap = raytracer.trace_rays(
    incident_ray_direction=incident_ray_direction, device=device
)

if is_distributed:
    torch.distributed.all_reduce(final_bitmap, op=torch.distributed.ReduceOp.SUM)

final_bitmap = raytracer.normalize_bitmap(final_bitmap)

end_time = time.time()
elapsed = end_time - start_time

print(elapsed)
