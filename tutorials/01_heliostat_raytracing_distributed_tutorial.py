import pathlib

import h5py
import torch

from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util import set_logger_config
from artist.util.environment_setup import (
    setup_global_distributed_environment,
)
from artist.util.scenario import Scenario

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set up logger
set_logger_config()

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path("please/insert/the/path/to/the/scenario/here/scenario.h5")

# The distributed environment is setup and destroyed using a Generator object.
environment_generator = setup_global_distributed_environment()

device, is_distributed, rank, world_size = next(environment_generator)

# Load the scenario.
with h5py.File(scenario_path) as scenario_file:
    scenario = Scenario.load_scenario_from_hdf5(
        scenario_file=scenario_file, device=device
    )

incident_ray_direction = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)

heliostat_target_light_source_mapping = None

# If you want to customize the mapping, choose the following style: list[tuple[str, str, torch.Tensor]]
# heliostat_target_light_source_mapping = [
#     ("AA39", "receiver", incident_ray_direction),
#     ("AA35", "solar_tower_juelich_upper", incident_ray_direction),
# ]

bitmap_resolution_e, bitmap_resolution_u = 256, 256
flux_distributions = torch.zeros(
    (
        scenario.target_areas.number_of_target_areas,
        bitmap_resolution_e,
        bitmap_resolution_u,
    ),
    device=device,
)

for heliostat_group_index, heliostat_group in enumerate(
    scenario.heliostat_field.heliostat_groups
):
    # If no mapping from heliostats to target areas to incident ray direction is provided, the scenario.index_mapping() method
    # activates all heliostats. It is possible to then provide a default target area index and a default incident ray direction
    # if those are not specified either all heliostats are assigned to the first target area found in the scenario with an
    # incident ray direction "north" (meaning the light source position is directly in the south) for all heliostats.
    (
        active_heliostats_mask,
        target_area_mask,
        incident_ray_directions,
    ) = scenario.index_mapping(
        heliostat_group=heliostat_group,
        string_mapping=heliostat_target_light_source_mapping,
        device=device,
    )

    # Align heliostats.
    heliostat_group.align_surfaces_with_incident_ray_directions(
        aim_points=scenario.target_areas.centers[target_area_mask],
        incident_ray_directions=incident_ray_directions,
        active_heliostats_mask=active_heliostats_mask,
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
    group_bitmaps_per_heliostat = ray_tracer.trace_rays(
        incident_ray_directions=incident_ray_directions,
        active_heliostats_mask=active_heliostats_mask,
        target_area_mask=target_area_mask,
        device=device,
    )

    group_bitmaps_per_target = ray_tracer.get_bitmaps_per_target(
        bitmaps_per_heliostat=group_bitmaps_per_heliostat,
        target_area_mask=target_area_mask,
        device=device,
    )

    if is_distributed:
        torch.distributed.all_reduce(
            flux_distributions, op=torch.distributed.ReduceOp.SUM
        )

    flux_distributions = flux_distributions + group_bitmaps_per_target

# Make sure the code after the yield statement in the environment Generator
# is called, to clean up the distributed process group.
try:
    next(environment_generator)
except StopIteration:
    pass
