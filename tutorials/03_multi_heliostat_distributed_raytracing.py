import pathlib
import time

import h5py
import torch

from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.scenario import Scenario
from artist.util import paint_loader, set_logger_config, utils
from tutorials.new_scenario import NewScenario

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

incident_ray_direction = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device) - sun_position

# Align all heliostats
for i in range(len(scenario.heliostats.heliostat_list)):
    scenario.heliostats.heliostat_list[
        i
    ].set_aligned_surface_with_incident_ray_direction(
        incident_ray_direction=incident_ray_direction, device=device
    )

aligned_scenario = NewScenario(scenario=scenario, device=device)
aimpoint_area = next(
    (
        area
        for area in aligned_scenario.target_areas.target_area_list
        if area.name == "receiver"
    ),
    None,
)

start_time = time.time()
# Create raytracer
raytracer = HeliostatRayTracer(
    scenario=aligned_scenario, world_size=world_size, rank=rank, batch_size=1, random_seed=rank
)

# Perform heliostat-based raytracing.
final_bitmap = raytracer.trace_rays(
    incident_ray_direction=incident_ray_direction,
    target_area=aimpoint_area,
    device=device
)

plt.imshow(final_bitmap.cpu().detach(), cmap="inferno")
plt.title(f"Flux Density Distribution from rank (heliostat): {rank}")
plt.savefig(f"rank_{rank}_{device}.png")

if is_distributed:
    torch.distributed.all_reduce(final_bitmap, op=torch.distributed.ReduceOp.SUM)

#final_bitmap = raytracer.normalize_bitmap(final_bitmap, aimpoint_area)

end_time = time.time()
print(end_time-start_time)

plt.imshow(final_bitmap.cpu().detach(), cmap="inferno")
plt.title("Total Flux Density Distribution")
plt.savefig(f"final_single_device_mode_{device}.png")

