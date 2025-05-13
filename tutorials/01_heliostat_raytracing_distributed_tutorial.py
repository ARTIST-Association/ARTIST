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

active_heliostats = ["AA39", "AA31", "AB38"]

heliostat_target_mapping = [
    ("AA39", "receiver"),
    #("AA39", "solar_tower_juelich_upper"),
    ("AA31", "solar_tower_juelich_lower"),
    ("AB38", "multi_focus_tower")
]

final_flux_distributions = torch.zeros((
    scenario.heliostat_field.number_of_heliostat_groups,
    scenario.target_areas.number_of_target_areas,
    256,
    256,), device=device
)

#TODO mapping
for heliostat_group_index, heliostat_group in enumerate(scenario.heliostat_field.heliostat_groups):

    if not active_heliostats:
        active_heliostats_indices = list(range(heliostat_group.number_of_heliostats))
    else:
        active_heliostats_indices = [i for i, name in enumerate(heliostat_group.names) if name in active_heliostats]

    target_area_indices = torch.tensor(
        [scenario.target_areas.names.index("multi_focus_tower"), 
         scenario.target_areas.names.index("receiver"), 
         scenario.target_areas.names.index("solar_tower_juelich_upper")
        ], device=device)

    heliostat_group.kinematic.aim_points[active_heliostats_indices] = scenario.target_areas.centers[target_area_indices]

    # Align all heliostats.
    heliostat_group.align_surfaces_with_incident_ray_direction(
        incident_ray_direction=incident_ray_direction, 
        active_heliostats_indices=active_heliostats_indices,
        device=device
    )

    # Create a ray tracer.
    ray_tracer = HeliostatRayTracer(
        scenario=scenario,
        heliostat_group_index=heliostat_group_index,
        light_source=scenario.light_sources.light_source_list[0],
        world_size=world_size, 
        rank=rank, 
        batch_size=4, 
        random_seed=rank
    )

    # Perform heliostat-based ray tracing.
    group_bitmaps = ray_tracer.trace_rays(
        incident_ray_direction=incident_ray_direction,
        active_heliostats_indices=active_heliostats_indices,
        target_area_indices=target_area_indices,
        device=device,
    )

    if is_distributed:
        torch.distributed.all_reduce(group_bitmaps, op=torch.distributed.ReduceOp.SUM)

    final_flux_distributions[heliostat_group_index] = group_bitmaps


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