import logging
import platform
from contextlib import contextmanager
from itertools import cycle, islice
from typing import Generator, Optional

import torch

from artist.util import config_dictionary

log = logging.getLogger(__name__)
"""A logger for the environment."""


def initialize_ddp_environment(
    device: Optional[torch.device] = None,
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
    device : Optional[torch.device]
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA, MPS, or CPU) based on availability and OS.

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
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)

    return device, is_distributed, rank, world_size


def create_subgroup_for_single_rank(
    rank: int, ranks_to_groups_mapping: dict[int, list[int]]
) -> tuple[
    Optional[int],
    Optional[int],
    Optional[torch.distributed.ProcessGroup],
]:
    """
    Assign the current process (rank) to a subgroup based on a predefined group assignment map.

    Parameters
    ----------
    rank : int
        The current process (rank).
    ranks_to_groups_mapping : dict[int, list[int]]
        The mapping from rank to heliostat group.

    Returns
    -------
        Optional[int]
            The rank within the heliostat group.
        Optional[int]
            The world size of the heliostat group.
        Optional[torch.distributed.ProcessGroup]]
            The distributed process group.
    """
    for heliostat_group_index, heliostat_group_ranks in ranks_to_groups_mapping.items():
        if rank in heliostat_group_ranks:
            process_subgroup = torch.distributed.new_group(ranks=heliostat_group_ranks)
            log.info(f"Rank {rank} joined heliostat group '{heliostat_group_index}' with ranks {heliostat_group_ranks}")
            heliostat_group_rank = heliostat_group_ranks.index(rank)
            heliostat_group_world_size = len(heliostat_group_ranks)
            return heliostat_group_rank, heliostat_group_world_size, process_subgroup

    return None, None, None


@contextmanager
def setup_distributed_environment(
    number_of_heliostat_groups: int,
    device: Optional[torch.device] = None,
) -> Generator[
    tuple[
        torch.device,
        bool,
        int,
        int,
        Optional[torch.distributed.ProcessGroup],
        dict[int, list[int]],
        int,
        int
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
        device (CUDA, MPS, or CPU) based on availability and OS.
    heliostat_group_assignments : Optional[dict[str, list[int]]]
        The mapping from rank to heliostat group.

    Yields
    ------
    device : torch.device
        The torch device assigned to this rank.
    is_distributed : bool
        Whether the environment is running in distributed mode.
    rank : int
        The global rank of the current process.
    world_size : int
        Total number of processes in the global process group.
    process_subgroup : Optional[torch.distributed.ProcessGroup]
        The ProcessGroup object representing the subgroup.
    ranks_to_groups_mapping : dict[int, list[int]]
        The mapping from rank to heliostat group.
    heliostat_group_rank : int
        The rank of the current process within its assigned subgroup.
    heliostat_group_world_size : Optional[int]
        Number of processes in the current process subgroup.
    """
    device, is_distributed, rank, world_size = initialize_ddp_environment(device)

    ranks_to_groups_mapping = distribute_groups_among_ranks(
        world_size=world_size,
        number_of_heliostat_groups=number_of_heliostat_groups
    )
    if is_distributed:
        (
            heliostat_group_rank,
            heliostat_group_world_size,
            process_subgroup,
        ) = create_subgroup_for_single_rank(
            rank=rank,
            ranks_to_groups_mapping=ranks_to_groups_mapping)
    else:
        heliostat_group_rank = 0
        heliostat_group_world_size = 1
        process_subgroup = None
    
    try:
        yield (
            device,
            is_distributed,
            rank,
            world_size,
            process_subgroup,
            ranks_to_groups_mapping,
            heliostat_group_rank,
            heliostat_group_world_size,
        )
    finally:
        if is_distributed:
            torch.distributed.barrier(
                device_ids=[torch.cuda.current_device()]
                if torch.distributed.get_backend() == "nccl"
                else None
            )
            torch.distributed.destroy_process_group()

def distribute_groups_among_ranks(world_size: int, 
                                  number_of_heliostat_groups: int) -> dict[int, list[int]]:
    """
    Distribute ranks among groups in round-robin fashion.
    
    If there are fewer ranks than groups, it repeats ranks to ensure 
    each group gets at least one rank.
    
    Parameters
    ----------
    world_size : int
        Total number of processes in the global process group.
    number_of_heliostat_groups : int

    Returns
    -------
    dict[int, list[int]]: A dictionary mapping group names to lists of assigned elements.
    """
    groups_to_ranks_mapping = {i: [] for i in range(world_size)}
    if world_size < number_of_heliostat_groups:
        groups = list(range(number_of_heliostat_groups))
    else:
        groups = list(islice(cycle(groups), world_size))

    group_iters = cycle(groups_to_ranks_mapping.values())
    for group in groups:
        next(group_iters).append(group)

    return groups_to_ranks_mapping

def distribute_ranks_among_groups(world_size: int, 
                                  number_of_heliostat_groups: int) -> dict[int, list[int]]:
    """
    Distribute ranks among groups in round-robin fashion.
    
    If there are fewer ranks than groups, it repeats ranks to ensure 
    each group gets at least one rank.
    
    Parameters
    ----------
    world_size : int
        Total number of processes in the global process group.
    number_of_heliostat_groups : int

    Returns
    -------
    dict[int, list[int]]: A dictionary mapping group names to lists of assigned elements.
    """
    ranks_to_groups_mapping = {i: [] for i in range(number_of_heliostat_groups)}
    ranks = list(range(world_size))

    repeated_elements = list(islice(cycle(ranks), number_of_heliostat_groups))
    remaining_elements = ranks[number_of_heliostat_groups:]
    all_elements = repeated_elements + remaining_elements

    group_iters = cycle(ranks_to_groups_mapping.values())
    for element in all_elements:
        next(group_iters).append(element)

    return ranks_to_groups_mapping


def get_device(device: Optional[torch.device] = None) -> torch.device:
    """
    Get the correct GPU device type for common operating systems, default to CPU if none is found.

    Parameters
    ----------
    device : Optional[torch.device]
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA, MPS, or CPU) based on availability and OS.

    Returns
    -------
    torch.device
        The device.
    """
    if device is None:
        log.info(
            "No device type provided. The device will default to GPU based on availability and OS, otherwise to CPU."
        )

        os_name = platform.system()

        if os_name == config_dictionary.linux or os_name == config_dictionary.windows:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            log.info(
                f"OS: {os_name}, cuda available: {torch.cuda.is_available()}, selected device type: {device.type}"
            )
        elif os_name == config_dictionary.mac:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            log.info(
                f"OS: Mac, mps available: {torch.backends.mps.is_available()}, selected device type: {device.type}"
            )
        else:
            log.warning(
                f"OS '{os_name}' not recognized. ARTIST is optimized for GPU computations but will run on CPU."
            )
            device = torch.device("cpu")

    return device
