import pathlib

import h5py
import torch

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
scenario_path = pathlib.Path("please/insert/the/path/to/the/scenario/here/scenario.h5")

# The distributed environment is setup and destroyed using a Generator object.
environment_generator = utils.setup_global_distributed_environment(device=device)

device, is_distributed, rank, world_size = next(environment_generator)

# Load the scenario.
with h5py.File(scenario_path) as scenario_file:
    scenario = Scenario.load_scenario_from_hdf5(
        scenario_file=scenario_file, device=device
    )

# Specify a mapping of active heliostats, their targets, and the incident ray directions.
# If no mapping is provided, the default activates all heliostats, the target is the receiver for all heliostats
# and the light source is in the south for all heliostats.
heliostat_target_light_source_mapping_string = None
# if you want to customize the mapping, choose the following style: list[tuple[str, str, torch.Tensor]]
# heliostat_target_light_source_mapping_string = [
#     ("heliostat_name_1", "target_name_3", incident_ray_direction_tensor_1),
#     ("heliostat_name_2", "target_name_1", incident_ray_direction_tensor_1),
#     ("heliostat_name_3", "target_name_2", incident_ray_direction_tensor_2),
#     ("heliostat_name_4", "target_name_1", incident_ray_direction_tensor_3),
# ]


bitmap_resolution_e, bitmap_resolution_u = 256, 256
final_flux_distributions = torch.zeros(
    (
        scenario.heliostat_field.number_of_heliostat_groups,
        scenario.target_areas.number_of_target_areas,
        bitmap_resolution_e,
        bitmap_resolution_u,
    ),
    device=device,
)

for heliostat_group_index, heliostat_group in enumerate(
    scenario.heliostat_field.heliostat_groups
):
    (
        incident_ray_directions,
        active_heliostats_indices,
        target_area_indices,
    ) = scenario.index_mapping(
        string_mapping=heliostat_target_light_source_mapping_string,
        heliostat_group_index=heliostat_group_index,
        device=device,
    )

    heliostat_group.kinematic.aim_points[active_heliostats_indices] = (
        scenario.target_areas.centers[target_area_indices]
    )

    # Align all heliostats.
    heliostat_group.align_surfaces_with_incident_ray_directions(
        incident_ray_directions=incident_ray_directions,
        active_heliostats_indices=active_heliostats_indices,
        device=device,
    )

    # Create a ray tracer.
    ray_tracer = HeliostatRayTracer(
        scenario=scenario,
        heliostat_group=heliostat_group,
        world_size=world_size,
        rank=rank,
        batch_size=4,
        random_seed=rank,
        bitmap_resolution_e=bitmap_resolution_e,
        bitmap_resolution_u=bitmap_resolution_u,
    )

    # Perform heliostat-based ray tracing.
    group_bitmaps = ray_tracer.trace_rays(
        incident_ray_directions=incident_ray_directions,
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
