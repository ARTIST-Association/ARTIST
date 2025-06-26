import os
import platform

import pytest
import torch

from artist.util import config_dictionary


@pytest.fixture(params=["cpu", "gpu"])
def device(request: pytest.FixtureRequest) -> torch.device:
    """
    Return the device on which to initialize tensors.

    The "gpu" device is skipped in CI environments.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The pytest fixture used to consider different test cases.

    Returns
    -------
    torch.device
        The device on which to initialize tensors.
    """
    param = request.param

    if param == "gpu":
        if os.environ.get("CI", "false").lower() == "true":
            pytest.skip("Skipping GPU test in CI environment")

        os_name = platform.system()
        if os_name in {config_dictionary.linux, config_dictionary.windows}:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif os_name == config_dictionary.mac:
            return torch.device("cpu")
        else:
            return torch.device("cpu")

    return torch.device("cpu")
