import os
from unittest.mock import patch

import pytest
import torch

from artist.util.environment_setup import (
    get_device,
    setup_distributed_environment,
)


@pytest.mark.parametrize(
    "is_distributed, rank, world_size",
    [
        (True, 0, 2),
        (True, 1, 3),
        (False, 0, 1),
    ],
)
def test_setup_global_distributed_environment(
    is_distributed: bool, rank: int, world_size: int, device: torch.device
) -> None:
    """
    Test the setup of the distributed environment.

    Parameters
    ----------
    is_distributed : bool
        Distributed mode enabled or disabled.
    rank : int
        The rank of the current process.
    world_size : int
        The world size or total number of processes.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    # Set up mock environment variables
    mock_env = {
        "WORLD_SIZE": str(world_size),
        "RANK": str(rank),
    }

    if not is_distributed:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "torch.distributed.init_process_group",
                side_effect=RuntimeError("Mocked failure"),
            ) as mock_init_pg,
            patch(
                "torch.distributed.get_world_size",
                return_value=int(mock_env["WORLD_SIZE"]),
            ),
            patch("torch.distributed.get_rank", return_value=int(mock_env["RANK"])),
            patch("torch.distributed.barrier") as mock_barrier,
            patch("torch.distributed.destroy_process_group") as mock_destroy_pg,
        ):
            # Test the generator
            gen = setup_distributed_environment(device=device)
            device, is_distributed_out, rank_out, world_size_out = next(gen)

            # Assert outputs
            assert is_distributed_out == is_distributed
            assert rank_out == rank
            assert world_size_out == world_size

            mock_init_pg.assert_called_once()
            mock_barrier.assert_not_called()

            # Ensure cleanup
            gen.close()
            mock_destroy_pg.assert_not_called()

    if is_distributed:
        # Patch environment variables
        with (
            patch.dict(os.environ, mock_env, clear=True),
            patch("torch.distributed.init_process_group") as mock_init_pg,
            patch(
                "torch.distributed.get_world_size",
                return_value=int(mock_env["WORLD_SIZE"]),
            ),
            patch("torch.distributed.get_rank", return_value=int(mock_env["RANK"])),
            patch("torch.distributed.barrier") as mock_barrier,
            patch("torch.distributed.destroy_process_group") as mock_destroy_pg,
        ):
            # Test the generator
            gen = setup_distributed_environment(device=device)
            device, is_distributed_out, rank_out, world_size_out = next(gen)

            # Assert outputs
            assert is_distributed_out == is_distributed
            assert rank_out == rank
            assert world_size_out == world_size

            mock_init_pg.assert_called_once_with(
                backend="nccl" if device.type == "cuda" else "gloo",
                init_method="env://",
            )
            mock_barrier.assert_not_called()

            # Ensure cleanup
            gen.close()

            mock_barrier.assert_called_once()
            mock_destroy_pg.assert_called_once()


@pytest.mark.parametrize(
    "os_name, cuda_available, mps_available, expected",
    [
        ("Linux", True, False, "cuda"),
        ("Linux", False, False, "cpu"),
        ("Windows", True, False, "cuda"),
        ("Windows", False, False, "cpu"),
        ("Darwin", False, True, "mps"),
        ("Darwin", False, False, "cpu"),
        ("InvalidOS", False, False, "cpu"),
    ],
)
def test_get_device_logic(
    monkeypatch: pytest.MonkeyPatch,
    os_name: str,
    cuda_available: bool,
    mps_available: bool,
    expected: str,
) -> None:
    """
    Test the the get device method.

    Parameters
    ----------
    monkeypatch : MonkeyPatch
        Pytest's monkeypatch fixture for patching system or hardware states.
    os_name : str
        Name of the operating system.
    cuda_available : bool
        Simulated CUDA availability.
    mps_available : bool
        Simulated MPS availability.
    expected : str
        Expected `device.type` returned by `get_device`.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    monkeypatch.setattr("platform.system", lambda: os_name)
    monkeypatch.setattr("torch.cuda.is_available", lambda: cuda_available)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: mps_available)

    device = get_device()
    assert device.type == expected
