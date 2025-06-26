import logging
import platform
from typing import Generator

import torch

from artist.util import config_dictionary

log = logging.getLogger(__name__)
"""A logger for the environment."""


def setup_global_distributed_environment(
    device: torch.device | None = None,
) -> Generator[tuple[torch.device, bool, int, int], None, None]:
    """
    Set up the distributed environment and destroy it in the end.

    Based on the available devices, the process group is initialized with the
    appropriate backend. For computation on GPUs the nccl backend optimized for
    NVIDIA GPUs is chosen. For computation on CPUs gloo is used as backend. If
    the program is run without the intention of being distributed, the world_size
    will be set to 1, accordingly the only rank is 0.

    Parameters
    ----------
    device : torch.device | None
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
    device = get_device()

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
        gpu_count = torch.cuda.device_count()
        device_id = rank % gpu_count
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)

    try:
        yield device, is_distributed, rank, world_size
    finally:
        if is_distributed:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()


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
