import pathlib
from artist import ARTIST_ROOT
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.scenario import Scenario
import h5py
from scripts import utils
import torch


if __name__ == "__main__":
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    
    # The distributed environment is setup and destroyed using a Generator object.
    environment_generator = utils.setup_distributed_environment(device=device)
    
    is_distributed, rank, world_size = next(environment_generator)

    if device.type == "cuda":
        torch.cuda.set_device(rank % torch.cuda.device_count())

    # Load the scenario.
    with h5py.File(
        pathlib.Path(ARTIST_ROOT) / "tests/data" / "test_scenario_stral.h5", "r"
    ) as config_h5:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=config_h5, device=device
        )
    incident_ray_direction = torch.tensor([0.0, -1.0, 0.0, 0.0], device=device)

    # Align heliostat.
    scenario.heliostats.heliostat_list[
        0
    ].set_aligned_surface_with_incident_ray_direction(
        incident_ray_direction=incident_ray_direction, device=device
    )

    # Create raytracer
    raytracer = HeliostatRayTracer(
        scenario=scenario, world_size=world_size, rank=rank, batch_size=100,
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

    # Make sure the code after the yield statement in the environment Generator
    # is called, to clean up the distributed process group.
    try:
        next(environment_generator)
    except StopIteration:
        pass
    