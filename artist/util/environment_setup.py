import logging
import platform
from contextlib import contextmanager
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
    rank: int, heliostat_group_map: Optional[dict[int, list[int]]]
) -> tuple[
    Optional[int],
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
    heliostat_group_map : Optional[dict[int, list[int]]]
        The mapping from rank to heliostat group.

    Returns
    -------
        Optional[int]
            The heliostat group id.
        Optional[int]
            The rank within the heliostat group.
        Optional[int]
            The world size of the heliostat group.
        Optional[torch.distributed.ProcessGroup]]
            The distributed process group.
    """
    if not heliostat_group_map:
        return None, None, None, None

    for index, ranks in heliostat_group_map.items():
        if rank in ranks:
            group_id = index
            group_ranks = ranks
            group = torch.distributed.new_group(ranks=group_ranks)
            log.info(f"Rank {rank} joined group '{group_id}' with ranks {group_ranks}")
            group_rank = group_ranks.index(rank)
            group_world_size = len(group_ranks)
            return group_id, group_rank, group_world_size, group

    return None, None, None, None


@contextmanager
def setup_distributed_environment(
    device: Optional[torch.device] = None,
    heliostat_group_assignments: Optional[dict[str, list[int]]] = None,
) -> Generator[
    tuple[
        torch.device,
        bool,
        int,
        Optional[int],
        int,
        Optional[int],
        Optional[int],
        Optional[torch.distributed.ProcessGroup],
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
    heliostat_group_rank : Optional[int]
        The rank of the current process within its assigned subgroup (if any).
    world_size : int
        Total number of processes in the global process group.
    heliostat_group_world_size : Optional[int]
        Number of processes in the current process's subgroup (if any).
    heliostat_group_id : Optional[int]
        ID of the subgroup this rank belongs to.
    process_subgroup : Optional[torch.distributed.ProcessGroup]
        The ProcessGroup object representing the subgroup (if any).
    """
    device, is_distributed, rank, world_size = initialize_ddp_environment(device)

    (
        heliostat_group_id,
        heliostat_group_rank,
        heliostat_group_world_size,
        process_subgroup,
    ) = (None, None, None, None)
    if is_distributed:
        (
            heliostat_group_id,
            heliostat_group_rank,
            heliostat_group_world_size,
            process_subgroup,
        ) = create_subgroup_for_single_rank(rank, heliostat_group_assignments)

    try:
        yield (
            device,
            is_distributed,
            rank,
            heliostat_group_rank,
            world_size,
            heliostat_group_world_size,
            heliostat_group_id,
            process_subgroup,
        )
    finally:
        if is_distributed:
            print(
                f"[Rank: {rank}, group: {heliostat_group_id}, device: {torch.cuda.current_device()}] Entering final barrier",
                flush=True,
            )
            torch.distributed.barrier(
                device_ids=[torch.cuda.current_device()]
                if torch.distributed.get_backend() == "nccl"
                else None
            )
            torch.distributed.destroy_process_group()


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
