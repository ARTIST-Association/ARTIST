"""This pytest considers loading a heliostat surface from a point cloud."""

import pathlib
import warnings

import h5py
import pytest
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from artist import ARTIST_ROOT, Scenario
from artist.raytracing.heliostat_tracing import DistortionsDataset, HeliostatRayTracer

warnings.filterwarnings("always")
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
    warnings.warn(
        "MPI is not available and distributed computing not possible. ARTIST will run on a single machine!",
        ImportWarning,
    )

if MPI is not None:
    # Setup MPI
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()
else:
    world_size = 1
    rank = 0


@pytest.mark.parametrize(
    "incident_ray_direction,expected_value, scenario_config",
    [
        (torch.tensor([0.0, -1.0, 0.0, 0.0]), "south.pt", "parallel_test_scenario"),
        (torch.tensor([1.0, 0.0, 0.0, 0.0]), "east.pt", "parallel_test_scenario"),
        (torch.tensor([-1.0, 0.0, 0.0, 0.0]), "west.pt", "parallel_test_scenario"),
        (torch.tensor([0.0, 0.0, 1.0, 0.0]), "above.pt", "parallel_test_scenario"),
    ],
)
def test_compute_bitmaps(
    incident_ray_direction, expected_value, scenario_config
) -> None:
    """
    Compute resulting flux density distribution (bitmap) for the given test case.

    With the aligned surface and the light direction, calculate the reflected rays on the heliostat surface.
    Calculate the intersection on the receiver.
    Compute the bitmaps and normalize them.
    Compare the calculated bitmaps with the expected ones.

    Parameters
    ----------
    incident_ray_direction : torch.Tensor
        The incident ray direction used for the test.
    expected_value : str
        The path to the expected value bitmap.
    scenario_config : str
        The name of the scenario to be loaded.
    """
    torch.manual_seed(7)

    with h5py.File(f"{ARTIST_ROOT}/scenarios/{scenario_config}.h5", "r") as config_h5:
        scenario = Scenario.load_scenario_from_hdf5(scenario_file=config_h5)

    receiver = scenario.receiver
    sun = scenario.light_source
    heliostat = scenario.heliostats.heliostat_list[0]

    heliostat.set_aligned_surface(incident_ray_direction=incident_ray_direction)
    heliostat.set_preferred_reflection_direction(rays=-incident_ray_direction)

    # Currently heliostat raytracing is only possible for heliostats with the same number of surface points/normals
    number_of_surface_points = heliostat.preferred_reflection_direction.size(0)

    # Create data set
    distortions_dataset = DistortionsDataset(
        light_source=sun,
        number_of_points=number_of_surface_points,
        random_seed=7,
    )

    # Create distributed sampler
    distortions_sampler = DistributedSampler(
        dataset=distortions_dataset, shuffle=False, num_replicas=world_size, rank=rank
    )

    # Create dataloader
    distortions_loader = DataLoader(
        distortions_dataset,
        batch_size=2,
        shuffle=False,
        sampler=distortions_sampler,
    )

    # Create raytracer
    raytracer = HeliostatRayTracer()

    final_bitmap = torch.zeros((receiver.resolution_x, receiver.resolution_y))
    for batch_u, batch_e in distortions_loader:
        rays = raytracer.scatter_rays(
            heliostat.preferred_reflection_direction,
            batch_u,
            batch_e,
        )

        intersections = raytracer.line_plane_intersections(
            receiver.plane_normal,
            receiver.center,
            rays,
            heliostat.current_aligned_surface_points,
        )

        dx_ints = intersections[:, :, 0] + receiver.plane_x / 2 - receiver.center[0]
        dy_ints = intersections[:, :, 2] + receiver.plane_y / 2 - receiver.center[2]

        indices = (
            (-1 <= dx_ints)
            & (dx_ints < receiver.plane_x + 1)
            & (-1 <= dy_ints)
            & (dy_ints < receiver.plane_y + 1)
        )

        total_bitmap = raytracer.sample_bitmap(
            dx_ints,
            dy_ints,
            indices,
            receiver.plane_x,
            receiver.plane_y,
            receiver.resolution_x,
            receiver.resolution_y,
        )

        final_bitmap += total_bitmap

    if MPI is not None:
        final_bitmap = comm.allreduce(final_bitmap, op=MPI.SUM)
    final_bitmap = raytracer.normalize_bitmap(
        final_bitmap,
        distortions_dataset.distortions_u.numel(),
        receiver.plane_x,
        receiver.plane_y,
    )

    if rank == 0:
        expected_path = (
            pathlib.Path(ARTIST_ROOT)
            / "tests/field/test_bitmaps_load_surface_stral"
            / expected_value
        )

        expected = torch.load(expected_path)
        torch.testing.assert_close(final_bitmap.T, expected, atol=5e-4, rtol=5e-4)
