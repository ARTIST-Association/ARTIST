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
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path(
    "tutorials/data/scenarios/test_scenario_paint_four_heliostats.h5"
)

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

# Set the aim point for the heliostats.
aim_point = scenario.get_target_area("receiver").center

# TODO NUR FÜR DEN AKTUELLEN RANK DIE INFOS RAUSGEBEN
# TODO DADURCH FOR LOOP ELIMINIERIEN
heliostat_group_to_rank_mapping = (
    utils.set_up_single_heliostat_group_distributed_environment(
        rank=rank,
        world_size=world_size,
        number_of_heliostat_groups=len(scenario.heliostat_field.heliostat_groups),
    )
)


for (
    heliostat_group_index,
    heliostat_group_ranks,
    group_environment,
) in heliostat_group_to_rank_mapping:
    if rank in heliostat_group_ranks:
        scenario.heliostat_field.heliostat_groups[
            heliostat_group_index
        ].aim_points = aim_point.unsqueeze(0).expand(
            heliostat_group.aim_points.shape[0], -1
        )
        heliostat_group.align_surfaces_with_incident_ray_direction(
            incident_ray_direction=incident_ray_direction, device=device
        )

        ray_tracer = HeliostatRayTracer(
            scenario=scenario,
            world_size=world_size,
            rank=rank,
            batch_size=2,
            random_seed=rank,
        )

        final_bitmap = ray_tracer.trace_rays(
            incident_ray_direction=incident_ray_direction,
            target_area=scenario.get_target_area("receiver"),
            device=device,
        )

        if is_distributed:
            torch.distributed.all_reduce(
                final_bitmap, op=torch.distributed.ReduceOp.SUM, group=heliostat_group
            )


plt.imshow(final_bitmap.cpu().detach(), cmap="inferno")
plt.title(f"Flux Density Distribution from rank: {rank}")
plt.savefig(f"distributed_flux_rank_{rank}.png")

if is_distributed:
    torch.distributed.all_reduce(final_bitmap, op=torch.distributed.ReduceOp.SUM)

# Make sure the code after the yield statement in the environment Generator
# is called, to clean up the distributed process group.
try:
    next(environment_generator)
except StopIteration:
    pass

plt.imshow(final_bitmap.cpu().detach(), cmap="inferno")
plt.title("Total Flux Density Distribution")
plt.savefig("distributed_final_flux.png")
