import pathlib

import h5py
import torch
from matplotlib import pyplot as plt

from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util import set_logger_config, utils
from artist.util.scenario import Scenario

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set up logger
set_logger_config()

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path("/workVERLEIHNIX/mb/ARTIST/tutorials/data/scenarios/test_scenario_paint_four_heliostats.h5")

# The distributed environment is setup and destroyed using a Generator object.
environment_generator = utils.setup_global_distributed_environment(device=device)

device, is_distributed, rank, world_size = next(environment_generator)

# Load the scenario.
with h5py.File(scenario_path) as scenario_file:
    scenario = Scenario.load_scenario_from_hdf5(
        scenario_file=scenario_file, device=device
    )

# The incident ray direction needs to be normed.
incident_ray_direction = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)

heliostat_selection = ["AA39", "AA35"]

bitmaps = []
for group_index, group in enumerate(scenario.heliostat_field.heliostat_groups):

    if not heliostat_selection:
        heliostat_indices = list(range(len(group.names)))
    else:
        heliostat_indices = [i for i, name in enumerate(group.names) if name in heliostat_selection]

    target_area_index = scenario.target_areas.names.index("receiver")
    group.aim_points = scenario.target_areas.centers[target_area_index]

    # Align all heliostats.
    group.align_surfaces_with_incident_ray_direction(
        incident_ray_direction=incident_ray_direction, 
        heliostat_indices=heliostat_indices,
        device=device
    )

    # Create a ray tracer.
    ray_tracer = HeliostatRayTracer(
        scenario=scenario,
        heliostat_group=group,
        heliostat_indices=heliostat_indices,
        light_source=scenario.light_sources.light_source_list[0],
        world_size=world_size, 
        rank=rank, 
        batch_size=4, 
        random_seed=rank
    )

    # Perform heliostat-based ray tracing.
    group_bitmap = ray_tracer.trace_rays(
        incident_ray_direction=incident_ray_direction,
        target_area_indices=[target_area_index, target_area_index],
        device=device,
    )

    if is_distributed:
        torch.distributed.all_reduce(group_bitmap, op=torch.distributed.ReduceOp.SUM)

    bitmaps.append(group_bitmap)



# Make sure the code after the yield statement in the environment Generator
# is called, to clean up the distributed process group.
try:
    next(environment_generator)
except StopIteration:
    pass


# plt.imshow(final_bitmap.cpu().detach(), cmap="inferno")
# plt.title(f"Flux Density Distribution from rank: {rank}")
# plt.savefig(f"distributed_flux_rank_{rank}.png")

# plt.imshow(final_bitmap.cpu().detach(), cmap="inferno")
# plt.title("Total Flux Density Distribution")
# plt.savefig("distributed_final_flux.png")