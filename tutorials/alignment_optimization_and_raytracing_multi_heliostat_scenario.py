import pathlib

from artist.raytracing.heliostat_tracing import HeliostatRayTracer
import h5py
import torch

from artist import ARTIST_ROOT
from artist.scenario import Scenario
from artist.util import set_logger_config, utils

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set up logger
set_logger_config()

# Set the device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# The distributed environment is setup and destroyed using a Generator object.
environment_generator = utils.setup_distributed_environment(device=device)

is_distributed, rank, world_size = next(environment_generator)

if device.type == "cuda":
    torch.cuda.set_device(rank % torch.cuda.device_count())

# If you have already generated the tutorial scenario yourself, you can leave this boolean as False. If not, set it to
# true and a pre-generated scenario file will be used for this tutorial!
use_pre_generated_scenario = True
scenario_path = (
    pathlib.Path(ARTIST_ROOT) / "please/insert/the/path/to/the/scenario/here/name.h5"
)
if use_pre_generated_scenario:
    scenario_path = (
        pathlib.Path(ARTIST_ROOT) / "tutorials/data/four_heliostat_scenario.h5"
    )

# Load the scenario.
with h5py.File(scenario_path, "r") as config_h5:
    scenario = Scenario.load_scenario_from_hdf5(scenario_file=config_h5, device=device)

# distributed raytracing with 4 heliostats
incident_ray_direction = torch.tensor([0.0, -1.0, 0.0, 0.0], device=device)

# Loop option:
# Remove loop, align multiple heliostats at once
for i in range(len(scenario.heliostats.heliostat_list)):
    scenario.heliostats.heliostat_list[i].set_aligned_surface_with_incident_ray_direction(
        incident_ray_direction=incident_ray_direction, device=device
    )
    
    # Create raytracer
    raytracer = HeliostatRayTracer(
        scenario=scenario,
        index = i,
        world_size=world_size,
        rank=rank,
        batch_size=100,
    )

    # Perform heliostat-based raytracing.
    final_bitmap = raytracer.trace_rays(
        incident_ray_direction=incident_ray_direction, device=device
    )
    if is_distributed:
        final_bitmap = torch.distributed.all_reduce(
            final_bitmap, op=torch.distributed.ReduceOp.SUM
        )

    final_bitmap = raytracer.normalize_bitmap(final_bitmap)


    import matplotlib.pyplot as plt
    plt.imshow(final_bitmap.cpu().detach().numpy())
    plt.savefig(pathlib.Path(ARTIST_ROOT) / f"distributed_test_{i}.png")




# DDP Option:
# Generate Dataset from heliostats?
# 4 Heliostate und 4 GPUs --> 1 Heliostat per GPU kann so benutzt werden wie bisher
# 2000 Heliostate und 4 GPUs --> 500 Heliostate per GPU nacheinander oder parallel, parallel mit aktueller Datenstruktur nicht möglich

# Example
# world size = 4 
# dataset has 2000 samples (dataset.__len__() returns 2000) 
# each rank will work on 500 samples 
# batch size = 10 --> each rank has to process 50 batches

# DataLoader für Distortions
# 1. DistortionDataset(Dataset) mit __init__, __len__, __getitem__, 
#       __getitem__ gibt beide Parameter zurück (self.distortions_u und self.distortion_e)
# 2. DistributedSampler()
# 3. DataLoader()

# Dataloader für Heliostate.
# 1. HeliostatDataset(Dataset) mit __init__, __len__, __getitem__, 
#       __getitem__ gibt beide Parameter zurück (heliostat.surface_points, heliostat.normals)  
#               --> Das reicht aber eigentlich gar nicht.
#                   Wir brauchen auch heliostat.aimpoint, heliostat.kinematic
#                   heliostat.current_aligned_surface_points,
#                   heliostat.current_aligned_surface_normals,
#                   heliostat.is_aligned = False
#                   heliostat.preferred_reflection_direction?

# 2. DistributedSampler()
# 3. DataLoader() mit batch_size auswählen wie viele Heliostate gleichzeitig von einem rank behandelt werden.

# Alignment von vielen Heliostaten gleichzeitig:
# Problem?
# - align-funktion ist aktuell Teil eines Heliostats. (Kann nur auf einen Heliostat aufgerufen werden.)
# - Orientierungsmatritzen gleichzeitig ausrechnen überhaupt möglich? - Jeder Heliostat hat eigene Parameter.
# - Sobald alle Orientierungsmatritzen bekannt sind können alle Heliostate gleichzeitig gedreht werden.
# - Kann oder sollte man das anders implementieren?
# - bleibt ansosnten nichts anderes übrig als Heliostate in einem Loop zu alignen?

# Raytracing von vielen Heliostaten sollte erstmal gleichzeitig gehen.




# Choose calibration data for all heliostats
calibration_properties_paths = []
calibration_properties_path = (
    pathlib.Path(ARTIST_ROOT) / "tests/data/calibration_properties.json"
)
