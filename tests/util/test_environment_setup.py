import os
from unittest.mock import MagicMock, patch

import pytest
import torch

from artist.util.environment_setup import (
    create_subgroups_for_nested_ddp,
    distribute_groups_among_ranks,
    get_device,
    initialize_ddp_environment,
)


@pytest.mark.parametrize(
    "is_distributed, rank, world_size",
    [
        (True, 0, 2),
        (True, 1, 3),
        (False, 0, 1),
    ],
)
def test_initialize_ddp_environment(
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
        ):
            device, is_distributed_out, rank_out, world_size_out = (
                initialize_ddp_environment(device=device)
            )

            assert is_distributed_out == is_distributed
            assert rank_out == rank
            assert world_size_out == world_size

            mock_init_pg.assert_called_once()

    if is_distributed:
        with (
            patch.dict(os.environ, mock_env, clear=True),
            patch("torch.distributed.init_process_group") as mock_init_pg,
            patch(
                "torch.distributed.get_world_size",
                return_value=int(mock_env["WORLD_SIZE"]),
            ),
            patch("torch.distributed.get_rank", return_value=int(mock_env["RANK"])),
        ):
            device, is_distributed_out, rank_out, world_size_out = (
                initialize_ddp_environment(device=device)
            )

            assert is_distributed_out == is_distributed
            assert rank_out == rank
            assert world_size_out == world_size

            mock_init_pg.assert_called_once_with(
                backend="nccl" if device.type == "cuda" else "gloo",
                init_method="env://",
            )


@pytest.mark.parametrize(
    "rank, groups_to_ranks_mapping, expected",
    [
        (0, {0: [0, 1, 2]}, (0, 1)),
        (2, {0: [0, 1, 2]}, (0, 1)),
        (4, {0: [0], 1: [1], 2: [2], 3: [0], 4: [1]}, (1, 2)),
        (5, {0: [0, 1, 2]}, (0, 1)),
    ],
)
def test_create_subgroups_for_nested_ddp(
    rank: int, groups_to_ranks_mapping: dict[int, list[int]], expected: tuple[int, int]
):
    """
    Test the creation of process subgroups.

    Parameters
    ----------
    rank : int
        The current process.
    groups_to_ranks_mapping : dict[int, list[int]]
        The mapping from heliostat group to rank.
    expected : tuple[int, int]
        The expected return values.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    with patch("torch.distributed.new_group", return_value=MagicMock()):
        heliostat_group_rank, heliostat_group_world_size, _ = (
            create_subgroups_for_nested_ddp(
                rank=rank, groups_to_ranks_mapping=groups_to_ranks_mapping
            )
        )
        assert heliostat_group_rank == expected[0]
        assert heliostat_group_world_size == expected[1]


@pytest.mark.parametrize(
    "world_size, number_of_heliostat_groups, expected_mapping, expected_is_nested",
    [
        (1, 3, {0: [0, 1, 2]}, False),
        (3, 3, {0: [0], 1: [1], 2: [2]}, False),
        (5, 3, {0: [0], 1: [1], 2: [2], 3: [0], 4: [1]}, True),
    ],
)
def test_distribute_groups_among_ranks(
    world_size: int,
    number_of_heliostat_groups: int,
    expected_mapping: dict[int, list[int]],
    expected_is_nested: bool,
):
    """
    Test the distribution of groups among ranks.

    Parameters
    ----------
    world_size : int
        Total number of processes in the global process group.
    number_of_heliostat_groups : int
        The number of heliostat groups.
    expected_mapping : dict[int, list[int]]
        The expected mapping.
    expected_is_nested : bool
        Whether nested setup is expected.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    mapping, is_nested = distribute_groups_among_ranks(
        world_size, number_of_heliostat_groups
    )
    assert mapping == expected_mapping
    assert is_nested is expected_is_nested


@pytest.mark.parametrize(
    "os_name, cuda_available, mps_available, expected",
    [
        ("Linux", True, False, "cuda"),
        ("Linux", False, False, "cpu"),
        ("Windows", True, False, "cuda"),
        ("Windows", False, False, "cpu"),
        ("Darwin", False, True, "cpu"),
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
    Test the get device method.

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
