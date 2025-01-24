import os
import pathlib

import h5py
import psutil
import torch
from matplotlib import pyplot as plt

from artist import ARTIST_ROOT
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.scenario import Scenario
from artist.util import set_logger_config, utils

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set up logger
set_logger_config()

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The distributed environment is setup and destroyed using a Generator object.
environment_generator = utils.setup_distributed_environment(device=device)

is_distributed, rank, world_size = next(environment_generator)

if device.type == "cuda":
    gpu_count = torch.cuda.device_count()
    device_id = rank % gpu_count
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

if device.type == "cpu":
    num_cores = os.cpu_count()
    if isinstance(num_cores, int):
        cores_per_rank = num_cores // world_size
    start_core = rank * cores_per_rank
    end_core = start_core + cores_per_rank - 1
    process = psutil.Process(os.getpid())
    process.cpu_affinity(list(range(start_core, end_core + 1)))

# If you have already generated the tutorial scenario yourself, you can leave this boolean as False. If not, set it to
# true and a pre-generated scenario file will be used for this tutorial!
use_pre_generated_scenario = True
scenario_path = (
    pathlib.Path(ARTIST_ROOT) / "please/insert/the/path/to/the/scenario/here/name.h5"
)
if use_pre_generated_scenario:
    scenario_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tutorials/data/test_scenario_paint_single_heliostat.h5"
    )

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
plt.savefig(f"rank_{rank}.png")
torch.distributed.barrier()

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
plt.savefig("final.png")
