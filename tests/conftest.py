import platform

import pytest
import torch


@pytest.fixture(params=[torch.device("cpu"), "gpu"], ids=["cpu", "gpu"])
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
    device = request.param
    if device == "gpu":
        os_name = platform.system()
        if os_name == "Linux" or os_name == "Windows":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif os_name == "Darwin":
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return device
