"""Tests loading a heliostat surface from STRAL data and performing ray tracing."""

import pathlib
import warnings

import h5py
import pytest
import torch

from artist import ARTIST_ROOT
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.scenario import Scenario

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


@pytest.fixture(params=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def device(request: pytest.FixtureRequest) -> torch.device:
    """
    Return the device on which to initialize tensors.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The pytest fixture used to consider different test cases.

    Returns
    -------
    torch.device
        The device on which to initialize tensors.
    """
    return torch.device(request.param)


@pytest.mark.parametrize(
    "incident_ray_direction, expected_value, scenario_config",
    [
        (torch.tensor([0.0, -1.0, 0.0, 0.0]), "south", "test_scenario"),
        (torch.tensor([1.0, 0.0, 0.0, 0.0]), "east", "test_scenario"),
        (torch.tensor([-1.0, 0.0, 0.0, 0.0]), "west", "test_scenario"),
        (torch.tensor([0.0, 0.0, 1.0, 0.0]), "above", "test_scenario"),
        (
            torch.tensor([0.0, -1.0, 0.0, 0.0]),
            "individual_south",
            "test_individual_measurements_scenario",
        ),  # Test if loading with individual measurements works
    ],
)
def test_compute_bitmaps(
    incident_ray_direction: torch.Tensor,
    expected_value: str,
    scenario_config: str,
    device: torch.device,
) -> None:
    """
    Compute the resulting flux density distribution (bitmap) for the given test case.

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
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    # Load the scenario.
    with h5py.File(f"{ARTIST_ROOT}/scenarios/{scenario_config}.h5", "r") as config_h5:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=config_h5, device=device
        )

    # Align heliostat.
    scenario.heliostats.heliostat_list[0].set_aligned_surface_with_incident_ray_direction(
        incident_ray_direction=incident_ray_direction.to(device), device=device
    )

    # Create raytracer - currently only possible for one heliostat.
    raytracer = HeliostatRayTracer(
        scenario=scenario, world_size=world_size, rank=rank, batch_size=10
    )

    # Perform heliostat-based raytracing.
    final_bitmap = raytracer.trace_rays(
        incident_ray_direction=incident_ray_direction.to(device), device=device
    )

    # Apply all-reduce if MPI is used.
    if MPI is not None:
        final_bitmap = comm.allreduce(final_bitmap, op=MPI.SUM)
    final_bitmap = raytracer.normalize_bitmap(final_bitmap)

    if rank == 0:
        expected_path = (
            pathlib.Path(ARTIST_ROOT)
            / "tests/field/test_bitmaps_load_surface_stral"
            / f"{expected_value}_{device.type}.pt"
        )
        expected = torch.load(expected_path).to(device)
        torch.testing.assert_close(final_bitmap.T, expected, atol=5e-4, rtol=5e-4) 
