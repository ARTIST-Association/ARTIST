import pathlib

from artist.scenario import Scenario
from artist.util import utils
import h5py
import pytest
import torch

from artist import ARTIST_ROOT
from artist.util.alignment_optimizer import AlignmentOptimizer


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
    "optimizer_method", ,
    [
        ("motor_positions"),
        ("raytracing"),
    ],
)
def test_alignment_optimizer_methods(
    optimizer_method: str,
    device: torch.Tensor,
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

    scenario_path = pathlib.Path(ARTIST_ROOT) / "tests/data/test_scenario_paint.h5"
    calibration_properties_path = (
        pathlib.Path(ARTIST_ROOT) / "tests/data/calibration_properties.json"
    )

    # Load the scenario.
    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(scenario_file=scenario_file, device=device)

    # Get optimizable parameters. (This will choose all 28 kinematic parameters)
    parameters = utils.get_rigid_body_kinematic_parameters_from_scenario(scenario=scenario)

    # Set up optimizer
    optimizer = torch.optim.Adam(parameters, lr=0.001)

    # Set up learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=20,
        threshold=0.1,
        threshold_mode="abs",
    )

    # Choose calibration data
    calibration_properties_path = pathlib.Path(ARTIST_ROOT) / "tutorials/data/test_calibration_properties.json"

    # Load the calibration data
    center_calibration_image, incident_ray_direction, motor_positions = utils.get_calibration_properties(calibration_properties_path=calibration_properties_path, device=device)

    # Create alignment optimizer
    alignment_optimizer = AlignmentOptimizer(
        scenario=scenario,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    optimized_parameters, _ = alignment_optimizer.optimize(
        tolerance=1e-7,
        max_epoch=150,
        center_calibration_image=center_calibration_image,
        incident_ray_direction=incident_ray_direction,
        motor_positions=motor_positions,
        device=device
    )

    expected_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/data/expected_optimized_alignment_parameters"
        / f"{optimizer_method}_{device.type}.pt"
    )
    expected = torch.load(expected_path, map_location=device, weights_only=True)

    torch.testing.assert_close(optimized_parameters, expected, atol=0.01, rtol=0.01)
