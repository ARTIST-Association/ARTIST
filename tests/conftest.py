import os
import platform
import random
from collections.abc import Generator

import numpy as np
import pytest

from artist.util.environment import DdpSetup

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

from artist.util import constants


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
        if os_name in {constants.linux, constants.windows}:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif os_name == constants.mac:
            return torch.device("cpu")
        else:
            return torch.device("cpu")

    return torch.device("cpu")


@pytest.fixture
def ddp_setup_for_testing() -> DdpSetup:
    """
    Return a single device distributed setup used in tests.

    The device and groups_to_ranks_mapping should be set in every test as needed.

    Returns
    -------
    DdpSetup
        A minimal distributed environment configuration suitable for single-device testing.
    """
    return DdpSetup(
        device=None,
        is_distributed=False,
        is_nested=False,
        rank=0,
        world_size=1,
        process_subgroup=None,
        groups_to_ranks_mapping={0: [0, 1]},
        heliostat_group_rank=0,
        heliostat_group_world_size=1,
        ranks_to_groups_mapping={
            0: [0],
            1: [0],
        },
    )


@pytest.fixture(scope="session", autouse=True)
def enforce_determinism() -> Generator[None, None, None]:
    """
    Pytest fixture that enforces deterministic behavior across all tests.

    This fixture automatically runs once per pytest session and ensures reproducibility
    of results by:
      - Setting fixed random seeds for Python, NumPy, and PyTorch.
      - Enabling deterministic algorithms in PyTorch.
      - Disabling cuDNN benchmarking to prevent algorithm auto-selection.
      - Making cuDNN operations deterministic (where supported).

    Note: Some PyTorch CUDA operations (e.g. `grid_sampler_2d_backward_cuda`) do not
    have deterministic implementations. These raise a ``RuntimeError`` when
    ``torch.use_deterministic_algorithms(True)`` is active. Tests that use such
    operations must temporarily disable deterministic mode via
    ``torch.use_deterministic_algorithms(False)`` for the affected call.
    """
    seed = 7

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    print(f"[pytest] Deterministic mode enabled (seed={seed})")
    print(f"[pytest] CUDA available: {torch.cuda.is_available()}")
    print(
        f"[pytest] Deterministic algorithms: {torch.are_deterministic_algorithms_enabled()}"
    )

    yield
