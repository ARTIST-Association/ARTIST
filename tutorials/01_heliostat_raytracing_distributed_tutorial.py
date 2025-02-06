import pathlib

import h5py
import torch
from matplotlib import pyplot as plt

from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.scenario import Scenario
from artist.util import set_logger_config, utils

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set up logger
set_logger_config()

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path("please/insert/the/path/to/the/scenario/here/name.h5")

# The distributed environment is setup and destroyed using a Generator object.
environment_generator = utils.setup_distributed_environment(device=device)

device, is_distributed, rank, world_size = next(environment_generator)

# Load the scenario.
with h5py.File(scenario_path) as scenario_file:
    scenario = Scenario.load_scenario_from_hdf5(
        scenario_file=scenario_file, device=device
    )

incident_ray_direction = torch.tensor([0.0, -1.0, 0.0, 0.0], device=device)

# Align heliostat.
scenario.heliostats.heliostat_list[0].set_aligned_surface_with_incident_ray_direction(
    incident_ray_direction=incident_ray_direction, device=device
)

# Create raytracer
raytracer = HeliostatRayTracer(
    scenario=scenario, world_size=world_size, rank=rank, batch_size=1, random_seed=rank
)

# Perform heliostat-based raytracing.
final_bitmap = raytracer.trace_rays(
    incident_ray_direction=incident_ray_direction, device=device
)

plt.imshow(final_bitmap.cpu().detach(), cmap="inferno")
plt.title(f"Flux Density Distribution from rank: {rank}")
plt.savefig(f"rank_{rank}.png")

if is_distributed:
    torch.distributed.all_reduce(final_bitmap, op=torch.distributed.ReduceOp.SUM)

# Make sure the code after the yield statement in the environment Generator
# is called, to clean up the distributed process group.
try:
    next(environment_generator)
except StopIteration:
    pass

final_bitmap = raytracer.normalize_bitmap(final_bitmap)

plt.imshow(final_bitmap.cpu().detach(), cmap="inferno")
plt.title("Total Flux Density Distribution")
plt.savefig("final.png")
