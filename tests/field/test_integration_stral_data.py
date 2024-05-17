"""This pytest considers loading a heliostat surface from a point cloud."""

import pathlib
import warnings

import h5py
import matplotlib.pyplot as plt
import pytest
import torch

from artist import ARTIST_ROOT, Scenario
from artist.raytracing.heliostat_tracing import HeliostatRayTracer

warnings.filterwarnings("always")

# Attempt to import MPI.
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
    warnings.warn(
        "MPI is not available and distributed computing not possible. ARTIST will run on a single machine!",
        ImportWarning,
    )

# Set up MPI.
if MPI is not None:
    comm = MPI.COMM_WORLD
    world_size = comm.size
    rank = comm.rank
else:
    world_size = 1
    rank = 0


@pytest.mark.parametrize(
    "incident_ray_direction, expected_value, scenario_config",
    [
        (torch.tensor([0.0, -1.0, 0.0, 0.0]), "south.pt", "test_scenario"),
        (torch.tensor([1.0, 0.0, 0.0, 0.0]), "east.pt", "test_scenario"),
        (torch.tensor([-1.0, 0.0, 0.0, 0.0]), "west.pt", "test_scenario"),
        (torch.tensor([0.0, 0.0, 1.0, 0.0]), "above.pt", "test_scenario"),
        (
            torch.tensor([0.0, -1.0, 0.0, 0.0]),
            "south.pt",
            "test_individual_measurements_scenario",
        ),  # Test if loading with individual measurements works
    ],
)
def test_compute_bitmaps(
    incident_ray_direction: torch.Tensor, expected_value: str, scenario_config: str
) -> None:
    """
    Compute resulting flux density distribution (bitmap) for the given test case.

    With the aligned surface and the light direction, reflect the rays at every normal on the heliostat surface to
    calculate the preferred reflection direction.
    Then perform heliostat based raytracing. This uses distortions based on the model of the sun to generate additional
    rays, calculates the intersections on the receiver, and computes the bitmap.
    Then normalize the bitmaps and compare them with the expected value.

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

    # Load the scenario.
    with h5py.File(f"{ARTIST_ROOT}/scenarios/{scenario_config}.h5", "r") as config_h5:
        scenario = Scenario.load_scenario_from_hdf5(scenario_file=config_h5)

    # Align heliostat.
    scenario.heliostats.heliostat_list[0].set_aligned_surface(
        incident_ray_direction=incident_ray_direction
    )
    scenario.heliostats.heliostat_list[0].set_preferred_reflection_direction(
        rays=-incident_ray_direction
    )

    # Create raytracer - currently only possible for one heliostat.
    raytracer = HeliostatRayTracer(
        scenario=scenario, world_size=world_size, rank=rank, batch_size=100
    )

    # Perform heliostat-based raytracing.
    final_bitmap = raytracer.trace_rays()

    # Apply all-reduce if MPI is used.
    if MPI is not None:
        final_bitmap = comm.allreduce(final_bitmap, op=MPI.SUM)
    final_bitmap = raytracer.normalize_bitmap(final_bitmap)

    if rank == 0:
        expected_path = (
            pathlib.Path(ARTIST_ROOT)
            / "tests/field/test_bitmaps_load_surface_stral"
            / expected_value
        )
        plt.imshow(final_bitmap.T.detach().numpy(), cmap="twilight")
        plt.show()
        expected = torch.load(expected_path)
        plt.imshow(expected.detach().numpy(), cmap="twilight")
        plt.show()

        torch.testing.assert_close(final_bitmap.T, expected, atol=5e-4, rtol=5e-4)
