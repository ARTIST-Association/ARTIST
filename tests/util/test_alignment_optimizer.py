import json
import pathlib
import warnings

from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util import utils
import pytest
import torch

from artist import ARTIST_ROOT
from artist.util.alignment_optimizer import AlignmentOptimizer

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

@pytest.mark.parametrize("optimizer_method", [
    ("optimize_kinematic_parameters_with_motor_positions"),
    ("optimize_kinematic_parameters_with_raytracing")
])
def test_alignment_optimizer(optimizer_method: str,
                             device : torch.Tensor,
) -> None:
    """
    Test that the alignment optimizer is working as desired.

    Parameters
    ----------
    optimizer_method : str
        Defines the optimization method.
    device : torch.device
        The device on which to initialize tensors.
    
    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    scenario_path = pathlib.Path(ARTIST_ROOT) / "scenarios/test_scenario_paint.h5"
    calibration_properties_path = (
        pathlib.Path(ARTIST_ROOT)
        / "measurement_data/download_test/AA39/Calibration/86500-calibration-properties.json"
    )

    # Load the calibration data.
    with open(calibration_properties_path, 'r') as file:
        calibration_dict = json.load(file)
        sun_azimuth = torch.tensor(calibration_dict["Sun_azimuth"], device=device)
        sun_elevation = torch.tensor(calibration_dict["Sun_elevation"], device=device)
        incident_ray_direction = utils.convert_3d_direction_to_4d_format(utils.azimuth_elevation_to_enu(sun_azimuth, sun_elevation, degree=True), device=device)
        
    alignment_optimizer = AlignmentOptimizer(
        scenario_path=scenario_path,
        calibration_properties_path=calibration_properties_path,
    )

    _, scenario = getattr(alignment_optimizer, optimizer_method)(device=device)

    # Assertion
    # Create raytracer
    raytracer = HeliostatRayTracer(
        scenario=scenario, world_size=world_size, rank=rank, batch_size=1
    )

    # Perform heliostat-based raytracing.
    final_bitmap = raytracer.trace_rays(
        incident_ray_direction=incident_ray_direction, device=device
    )

    # Apply all-reduce if MPI is used.
    if MPI is not None:
        final_bitmap = comm.allreduce(final_bitmap, op=MPI.SUM)
    final_bitmap = raytracer.normalize_bitmap(final_bitmap)

    if rank == 0:
        expected_path = (
            pathlib.Path(ARTIST_ROOT)
            / "tests/util/test_bitmaps_alignment_optimization"
            / f"{optimizer_method}_{device.type}.pt"
        )
        expected = torch.load(expected_path, map_location=device)
        torch.testing.assert_close(final_bitmap.T, expected, atol=5e-4, rtol=5e-4)