import pytest
import torch

@pytest.fixture(params=["cpu", "cuda:3"] if torch.cuda.is_available() else ["cpu"])
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
