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
scenario_path = pathlib.Path(
    "/workVERLEIHNIX/mb/ARTIST/tutorials/data/scenarios/test_scenario_paint_four_heliostats.h5"
)

# The distributed environment is setup and destroyed using a Generator object.
environment_generator = utils.setup_global_distributed_environment(device=device)

device, is_distributed, rank, world_size = next(environment_generator)

# Load the scenario.
with h5py.File(scenario_path) as scenario_file:
    scenario = Scenario.load_scenario_from_hdf5(
        scenario_file=scenario_file, device=device
    )

incident_ray_direction = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)

# heliostat_target_light_source_mapping = None

# If you want to customize the mapping, choose the following style: list[tuple[str, str, torch.Tensor]]
heliostat_target_light_source_mapping = [
    ("AA39", "receiver", incident_ray_direction),
    ("AA35", "solar_tower_juelich_upper", incident_ray_direction),
]
# If no mapping is provided, the default activates all heliostats, the selected target is the first target
# found in the scenario for all heliostats, and the incident ray direction specified above will be set for
# all heliostats.

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
    incident_ray_directions = incident_ray_direction.expand(
        heliostat_group.number_of_heliostats
    )
    active_heliostats_indices = None
    target_area_indices = None
    if heliostat_target_light_source_mapping:
        (
            incident_ray_directions,
            active_heliostats_indices,
            target_area_indices,
        ) = scenario.index_mapping(
            string_mapping=heliostat_target_light_source_mapping,
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
