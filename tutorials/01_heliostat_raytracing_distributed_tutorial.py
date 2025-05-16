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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Specify the path to your scenario.h5 file.
# scenario_path = pathlib.Path("/workVERLEIHNIX/mb/ARTIST/tutorials/data/scenarios/test_scenario_paint_four_heliostats.h5")
scenario_path = pathlib.Path("tutorials/data/scenarios/test_scenario_paint_four_heliostats.h5")

# The distributed environment is setup and destroyed using a Generator object.
environment_generator = utils.setup_global_distributed_environment(device=device)

device, is_distributed, rank, world_size = next(environment_generator)

# Load the scenario.
with h5py.File(scenario_path) as scenario_file:
    scenario = Scenario.load_scenario_from_hdf5(
        scenario_file=scenario_file, device=device
    )

# The incident ray direction needs to be normed.
heliostat_target_sun_mapping_string = [
    ("AB38", "multi_focus_tower", torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)),
    ("AA31", "multi_focus_tower", torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)),
    ("AA35", "solar_tower_juelich_lower", torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)),
    ("AA39", "solar_tower_juelich_lower", torch.tensor([-1.0, 0.0, 0.0, 0.0], device=device)),
]
#heliostat_target_sun_mapping_string = None

final_flux_distributions = torch.zeros((
    scenario.heliostat_field.number_of_heliostat_groups,
    scenario.target_areas.number_of_target_areas,
    256,
    256,), device=device
)

for heliostat_group_index, heliostat_group in enumerate(scenario.heliostat_field.heliostat_groups):

    all_incident_ray_directions, incident_ray_direction_indices, active_heliostats_indices, target_area_indices = scenario.index_mapping(
        string_mapping=heliostat_target_sun_mapping_string,
        heliostat_group_index=heliostat_group_index,
        device=device
    )

    heliostat_group.kinematic.aim_points[active_heliostats_indices] = scenario.target_areas.centers[target_area_indices]

    # Align all heliostats.
    heliostat_group.align_surfaces_with_incident_ray_directions(
        incident_ray_directions=all_incident_ray_directions[incident_ray_direction_indices], 
        active_heliostats_indices=active_heliostats_indices,
        device=device
    )

    # Create a ray tracer.
    ray_tracer = HeliostatRayTracer(
        scenario=scenario,
        heliostat_group=heliostat_group,
        world_size=world_size, 
        rank=rank, 
        batch_size=4, 
        random_seed=rank
    )

    # Perform heliostat-based ray tracing.
    group_bitmaps = ray_tracer.trace_rays(
        incident_ray_directions=all_incident_ray_directions[incident_ray_direction_indices],
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


plt.imshow(final_flux_distributions[0, 0].cpu().detach(), cmap="inferno")
plt.savefig("test1.png")

# plt.imshow(final_flux_distributions[0, 1].cpu().detach(), cmap="inferno")
# plt.savefig("receiver_AB38.png")

plt.imshow(final_flux_distributions[0, 2].cpu().detach(), cmap="inferno")
plt.savefig("test2.png")

# plt.imshow(final_flux_distributions[0, 3].cpu().detach(), cmap="inferno")
# plt.savefig("solar_tower_upper_AA31_mit_AB38_mft.png")


# plt.imshow(final_bitmap.cpu().detach(), cmap="inferno")
# plt.title("Total Flux Density Distribution")
# plt.savefig("distributed_final_flux.png")