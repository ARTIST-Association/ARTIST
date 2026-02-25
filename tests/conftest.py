import os
import platform
import random

import numpy as np
import pytest

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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
            return torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        elif os_name == config_dictionary.mac:
            return torch.device("cpu")
        else:
            return torch.device("cpu")

    return torch.device("cpu")


@pytest.fixture
def ddp_setup_for_testing() -> dict[
    str,
    torch.device
    | bool
    | int
    | torch.distributed.ProcessGroup
    | dict[int, list[int]]
    | None,
]:
    """
    Return a single device distributed setup used in tests.

    The device and groups_to_ranks_mapping should be set in every test as needed.

    Returns
    -------
    dict[str, torch.device | bool | int | torch.distributed.ProcessGroup | dict[int, list[int]] | None],
        The single device distributed setup used in tests.
    """
    return {
        config_dictionary.device: None,
        config_dictionary.is_distributed: False,
        config_dictionary.is_nested: False,
        config_dictionary.rank: 0,
        config_dictionary.world_size: 1,
        config_dictionary.process_subgroup: None,
        config_dictionary.groups_to_ranks_mapping: None,
        config_dictionary.heliostat_group_rank: 0,
        config_dictionary.heliostat_group_world_size: 1,
        config_dictionary.ranks_to_groups_mapping: None,
    }


@pytest.fixture(scope="session", autouse=True)
def enforce_determinism():
    """
    Pytest fixture that enforces deterministic behavior across all tests.

    This fixture automatically runs once per pytest session and ensures reproducibility
    of results by:
      - Setting fixed random seeds for Python, NumPy, and PyTorch.
      - Enabling deterministic algorithms in PyTorch.
      - Disabling cuDNN benchmarking to prevent algorithm auto-selection.
      - Making cuDNN operations deterministic (where supported).

    Some PyTorch CUDA operations (e.g. `grid_sampler_2d_backward_cuda`) do not have
    deterministic implementations. These may raise a `RuntimeError` when
    `torch.use_deterministic_algorithms(True)` is enabled.
    For these operations, determinism is temporarily disabled.
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
