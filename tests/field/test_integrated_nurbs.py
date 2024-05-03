import warnings

import h5py
import pytest
import torch

from artist import ARTIST_ROOT, Scenario
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util import nurbs_converters

torch.manual_seed(7)

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


@pytest.fixture(scope="module")
def common_setup() -> Scenario:
    """
    Create the scenario to use across different tests and learn the NURBS.

    Returns
    -------
    Scenario
        The loaded scenario.
    """
    with h5py.File(f"{ARTIST_ROOT}/scenarios/test_scenario.h5", "r") as config_h5:
        scenario = Scenario.load_scenario_from_hdf5(scenario_file=config_h5)

    # The width and height of the heliostat
    width = 2
    height = 2

    # The number of control points for the NURBS surface
    num_control_points_e = 7
    num_control_points_n = 7

    nurbs_surface = nurbs_converters.deflectometry_to_nurbs(
        scenario.heliostats.heliostat_list[0].concentrator.facets.surface_points[::100],
        scenario.heliostats.heliostat_list[0].concentrator.facets.surface_normals[
            ::100
        ],
        width,
        height,
        num_control_points_e,
        num_control_points_n,
        num_epochs=500,
    )

    (
        surface_points,
        surface_normals,
    ) = nurbs_surface.calculate_surface_points_and_normals()

    scenario.heliostats.heliostat_list[
        0
    ].concentrator.facets.surface_points = surface_points
    scenario.heliostats.heliostat_list[
        0
    ].concentrator.facets.surface_normals = surface_normals

    return scenario


@pytest.mark.parametrize(
    "incident_ray_direction, expected_value",
    [
        (torch.tensor([0.0, -1.0, 0.0, 0.0]), "south.pt"),
        (torch.tensor([1.0, 0.0, 0.0, 0.0]), "east.pt"),
        (torch.tensor([-1.0, 0.0, 0.0, 0.0]), "west.pt"),
        (torch.tensor([0.0, 0.0, 1.0, 0.0]), "above.pt"),
    ],
)
def test_nurbs(
    common_setup: Scenario, incident_ray_direction: torch.Tensor, expected_value: str
):
    """
    Test the NURBS surface and the raytracing process.

    Parameters
    ----------
    common_setup : Scenario
        The scenario shared across different tests.
    incident_ray_direction : torch.Tensor
        The incident ray direction.
    expected_value : str
        Path to the expected bitmap.
    """
    scenario = common_setup

    scenario.heliostats.heliostat_list[0].set_aligned_surface(incident_ray_direction)

    scenario.heliostats.heliostat_list[0].set_preferred_reflection_direction(
        -incident_ray_direction
    )
    raytracer = HeliostatRayTracer(
        scenario=scenario, world_size=world_size, rank=rank, batch_size=5
    )
    final_bitmap = raytracer.trace_rays()

    if MPI is not None:
        final_bitmap = comm.allreduce(final_bitmap, op=MPI.SUM)
    final_bitmap = raytracer.normalize_bitmap(final_bitmap)
