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


@pytest.mark.parametrize(
    "incident_ray_direction, ray_direction_string, scenario_config",
    [
        (torch.tensor([0.0, -1.0, 0.0, 0.0]), "stral_south", "test_scenario_stral"),
        (torch.tensor([1.0, 0.0, 0.0, 0.0]), "stral_east", "test_scenario_stral"),
        (torch.tensor([-1.0, 0.0, 0.0, 0.0]), "stral_west", "test_scenario_stral"),
        (torch.tensor([0.0, 0.0, 1.0, 0.0]), "stral_above", "test_scenario_stral"),
        (
            torch.tensor([0.0, -1.0, 0.0, 0.0]),
            "individual_south",
            "test_scenario_stral_individual_measurements",
        ),
        (torch.tensor([0.0, -1.0, 0.0, 0.0]), "paint_south", "test_scenario_paint"),
    ],
)
def test_integration_alignment(
    incident_ray_direction: torch.Tensor,
    ray_direction_string: str,
    scenario_config: str,
    device: torch.device,
) -> None:
    """
    Align helisotats from different scenarios using the kinematic module to test the alignment process.

    With the aligned surface and the light direction, reflect the rays at every normal on the heliostat surface to
    calculate the preferred reflection direction.
    Then perform heliostat based raytracing. This uses distortions based on the model of the sun to generate additional
    rays, calculates the intersections on the receiver, and computes the bitmap.
    Then normalize the bitmaps and compare them with the expected value.

    Parameters
    ----------
    incident_ray_direction : torch.Tensor
        The incident ray direction used for the test.
    ray_direction_string : str
        String value describing the ray direction.
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
    with h5py.File(
        pathlib.Path(ARTIST_ROOT) / "tests/data" / f"{scenario_config}.h5", "r"
    ) as config_h5:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=config_h5, device=device
        )

    # Align heliostat.
    scenario.heliostats.heliostat_list[
        0
    ].set_aligned_surface_with_incident_ray_direction(
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
            / "tests/data/expected_bitmaps_integration"
            / f"{ray_direction_string}_{device.type}.pt"
        )
        expected = torch.load(expected_path, map_location=device, weights_only=True)
        torch.testing.assert_close(final_bitmap.T, expected, atol=5e-4, rtol=5e-4)