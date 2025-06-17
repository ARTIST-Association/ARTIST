import pathlib

import h5py
import torch

from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util import set_logger_config
from artist.util.environment_setup import (
    setup_distributed_environment,
)
from artist.util.scenario import Scenario

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set up logger
set_logger_config()

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path(
    "tutorials/data/scenarios/test_scenario_paint_four_heliostats.h5"
)

heliostat_group_assignments = {0: [0, 1], 1: [2, 3]}

with setup_distributed_environment(
    heliostat_group_assignments=heliostat_group_assignments
) as (
    device,
    is_distributed,
    rank,
    heliostat_group_rank,
    world_size,
    heliostat_group_world_size,
    group_id,
    subgroup,
):
    # Load the scenario.
    with h5py.File(scenario_path) as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file, device=device
        )

    # Artificially add another heliostat group
    scenario.heliostat_field.heliostat_groups.append(
        scenario.heliostat_field.heliostat_groups[0]
    )

    incident_ray_direction = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)

    heliostat_target_light_source_mapping = None

    # If you want to customize the mapping, choose the following style: list[tuple[str, str, torch.Tensor]]
    # heliostat_target_light_source_mapping = [
    #     ("AA39", "receiver", incident_ray_direction),
    #     ("AA35", "solar_tower_juelich_upper", incident_ray_direction),
    # ]

    heliostat_group = scenario.heliostat_field.heliostat_groups[group_id]

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
        world_size=heliostat_group_world_size,
        rank=heliostat_group_rank,
        batch_size=4,
        random_seed=rank,
    )

    # Perform heliostat-based ray tracing.
    bitmaps_per_heliostat = ray_tracer.trace_rays(
        incident_ray_directions=incident_ray_directions,
        active_heliostats_mask=active_heliostats_mask,
        target_area_mask=target_area_mask,
        device=device,
    )

    bitmaps_per_target = ray_tracer.get_bitmaps_per_target(
        bitmaps_per_heliostat=bitmaps_per_heliostat,
        target_area_mask=target_area_mask,
        device=device,
    )

    if is_distributed:
        torch.distributed.all_reduce(
            bitmaps_per_target, op=torch.distributed.ReduceOp.SUM, group=subgroup
        )

        torch.distributed.all_reduce(
            bitmaps_per_target, op=torch.distributed.ReduceOp.SUM
        )
