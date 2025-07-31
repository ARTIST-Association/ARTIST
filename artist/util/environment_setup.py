import logging
import platform
from collections import defaultdict
from contextlib import contextmanager
from itertools import cycle, islice
from typing import Generator

import torch

from artist.util import config_dictionary

log = logging.getLogger(__name__)
"""A logger for the environment."""


def initialize_ddp_environment(
    device: torch.device | None = None,
) -> tuple[torch.device, bool, int, int]:
    """
    Set up the distributed environment.

    Based on the available devices, the outer process group is initialized with the
    appropriate backend. For computation on GPUs the nccl backend optimized for
    NVIDIA GPUs is chosen. For computation on CPUs gloo is used as backend. If
    the program is run without the intention of being distributed, the world_size
    will be set to 1, accordingly the only rank is 0.

    Parameters
    ----------
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Yields
    ------
    torch.device
        The device for each rank.
    bool
        Distributed mode enabled or disabled.
    int
        The rank of the current process.
    int
        The world size or total number of processes.
    """
    device = get_device(device=device)

    backend = "nccl" if device.type == "cuda" else "gloo"

    is_distributed = False
    rank, world_size = 0, 1

    try:
        # Attempt to initialize the process group.
        torch.distributed.init_process_group(backend=backend, init_method="env://")
        is_distributed = True
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        if rank == 0:
            log.info("Distributed Mode enabled.")
            log.info(f"Using backend: {backend}.")
        log.info(
            f"Distributed process group initialized: Rank {rank}, World Size {world_size}"
        )

    except Exception:
        log.info("Distributed Mode disabled. Running in single-device mode.")

    # Explicitly set the device per process in distributed mode (only) for device type cuda.
    # mps is currently (2025) single-device only and cpu handles the distribution automatically.
    if device.type == "cuda" and is_distributed:
        device_id = rank % torch.cuda.device_count()
        if device_id == 1:
            device_id = 3
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)

    return device, is_distributed, rank, world_size


def create_subgroups_for_nested_ddp(
    rank: int, groups_to_ranks_mapping: dict[int, list[int]]
) -> tuple[
    int,
    int,
    torch.distributed.ProcessGroup | None,
]:
    """
    Assign the current process (rank) to a subgroup based on a predefined group assignment map.

    Parameters
    ----------
    rank : int
        The current process.
    groups_to_ranks_mapping : dict[int, list[int]]
        The mapping from heliostat group to rank.

    Returns
    -------
    int
        The rank within the heliostat group.
    int
        The world size of the heliostat group.
    torch.distributed.ProcessGroup | None
        The distributed process group.
    """
    ranks_to_groups_mapping = defaultdict(list)
    for single_rank, groups in groups_to_ranks_mapping.items():
        for group in groups:
            ranks_to_groups_mapping[group].append(single_rank)

    group_handles = {}
    # Set default values for when the current proces (rank) is not in a heliostat group.
    heliostat_group_rank = 0
    heliostat_group_world_size = 1
    process_subgroup = None
    found_rank_in_group = False

    for group_index, group_ranks in ranks_to_groups_mapping.items():
        process_group = torch.distributed.new_group(ranks=group_ranks)
        group_handles[group_index] = process_group

        if rank in group_ranks:
            heliostat_group_rank = group_ranks.index(rank)
            heliostat_group_world_size = len(group_ranks)
            process_subgroup = process_group
            found_rank_in_group = True
    if not found_rank_in_group:
        log.warning(
            f"The rank {rank} is not in a heliostat group. Using default values of \n"
            f"-Heliostat group rank: {heliostat_group_rank}\n"
            f"-Heliostat group world size: {heliostat_group_world_size}"
        )

    return heliostat_group_rank, heliostat_group_world_size, process_subgroup


@contextmanager
def setup_distributed_environment(
    number_of_heliostat_groups: int,
    device: torch.device | None = None,
) -> Generator[
    tuple[
        torch.device,
        bool,
        bool,
        int,
        int,
        torch.distributed.ProcessGroup | None,
        dict[int, list[int]],
        int,
        int,
    ],
    None,
    None,
]:
    """
    Set up the distributed environment.

    Parameters
    ----------
    device : Optional[torch.device]
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.
    heliostat_group_assignments : Optional[dict[str, list[int]]]
        The mapping from rank to heliostat group.

    Yields
    ------
    device : torch.device
        The torch device assigned to this rank.
    is_distributed : bool
        Whether the environment is running in distributed mode.
    is_nested : bool
        Indicates whether the distributed setup is nested or not.
    rank : int
        The global rank of the current process.
    world_size : int
        Total number of processes in the global process group.
    process_subgroup : Optional[torch.distributed.ProcessGroup]
        The ProcessGroup object representing the subgroup.
    groups_to_ranks_mapping : dict[int, list[int]]
        The mapping from rank to heliostat group.
    heliostat_group_rank : int
        The rank of the current process within its assigned subgroup.
    heliostat_group_world_size : int
        Number of processes in the current process subgroup.
    """
    device, is_distributed, rank, world_size = initialize_ddp_environment(device=device)

    groups_to_ranks_mapping, is_nested = distribute_groups_among_ranks(
        world_size=world_size, number_of_heliostat_groups=number_of_heliostat_groups
    )

    if is_nested:
        (
            heliostat_group_rank,
            heliostat_group_world_size,
            process_subgroup,
        ) = create_subgroups_for_nested_ddp(
            rank=rank, groups_to_ranks_mapping=groups_to_ranks_mapping
        )
    else:
        heliostat_group_rank = 0
        heliostat_group_world_size = 1
        process_subgroup = None

    try:
        yield (
            device,
            is_distributed,
            is_nested,
            rank,
            world_size,
            process_subgroup,
            groups_to_ranks_mapping,
            heliostat_group_rank,
            heliostat_group_world_size,
        )
    finally:
        if is_distributed:
            try:
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
            except Exception as e:
                print(f"Distributed cleanup failed: {e}")


def distribute_groups_among_ranks(
    world_size: int, number_of_heliostat_groups: int
) -> tuple[dict[int, list[int]], bool]:
    """
    Distribute groups among ranks in round-robin fashion.

    If there are fewer ranks than groups, some ranks receive multiple groups.
    If there are more ranks than groups, some groups are handled by multiple ranks, enabling nested distribution.

    Parameters
    ----------
    world_size : int
        Total number of processes in the global process group.
    number_of_heliostat_groups : int
        The number of heliostat groups.

    Returns
    -------
    dict[int, list[int]]
        The dictionary mapping heliostat groups to ranks.
    bool
        Indicates whether the distributed setup is nested or not.
    """
    groups_to_ranks_mapping: dict[int, list[int]] = {i: [] for i in range(world_size)}
    groups = list(range(number_of_heliostat_groups))

    is_nested = world_size > number_of_heliostat_groups

    if is_nested:
        groups = list(islice(cycle(groups), world_size))

    group_iters = cycle(groups_to_ranks_mapping.values())
    for group in groups:
        next(group_iters).append(group)

    return groups_to_ranks_mapping, is_nested


def get_device(
    device: torch.device | None = None,
) -> torch.device:
    """
    Get the correct GPU device type for common operating systems, default to CPU if none is found.

    Parameters
    ----------
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS. MPS (for Mac) is not supported due to
        limitations in torch.

    Returns
    -------
    torch.device
        The device.
    """
    os_name = platform.system()

    if device is None:
        log.info(
            "No device type provided. The device will default to GPU based on availability and OS, otherwise to CPU."
        )
        if os_name == config_dictionary.linux or os_name == config_dictionary.windows:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            log.info(
                f"OS: {os_name}, cuda available: {torch.cuda.is_available()}, selected device type: {device.type}"
            )
        elif os_name == config_dictionary.mac:
            device = torch.device("cpu")
            log.warning("Setting device to CPU. ARTIST only supports CPU for MacOS.")
        else:
            log.warning(
                f"OS '{os_name}' not recognized. ARTIST is optimized for GPU computations but will run on CPU."
            )
            device = torch.device("cpu")
    elif device.type == "mps":
        log.warning(
            "You are forcing ARTIST to run with MPS - this is not supported and will fail!."
        )

    return device
