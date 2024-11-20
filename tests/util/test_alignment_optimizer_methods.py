import pathlib

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
    "optimizer_method",
    [
        ("optimize_kinematic_parameters_with_motor_positions"),
        ("optimize_kinematic_parameters_with_raytracing"),
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

    alignment_optimizer = AlignmentOptimizer(
        scenario_path=scenario_path,
        calibration_properties_path=calibration_properties_path,
    )

    optimized_parameters, _ = getattr(alignment_optimizer, optimizer_method)(
        device=device
    )

    expected_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/data/expected_optimized_alignment_parameters"
        / f"{optimizer_method}_{device.type}.pt"
    )
    expected = torch.load(expected_path, map_location=device, weights_only=True)

    torch.testing.assert_close(optimized_parameters, expected, atol=0.01, rtol=0.01)
