import pathlib
from artist import ARTIST_ROOT
from artist.util.alignment_optimizer import AlignmentOptimizer
import pytest
import torch

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

def test_alignment_optimizer(device: torch.device):
    """
    TODO docstrings
    """
    scenario_path = pathlib.Path(ARTIST_ROOT) / "scenarios/test_scenario_paint.h5"
    calibration_properties_path = pathlib.Path(ARTIST_ROOT) / "measurement_data/download_test/AA39/Calibration/86500-calibration-properties.json"
    
    alignment_optimizer = AlignmentOptimizer(scenario_path=scenario_path,
                                             calibration_properties_path=calibration_properties_path)
    
    optimized_kinematic_parameters = alignment_optimizer.optimize_kinematic_parameters(device=device)

    assert True


