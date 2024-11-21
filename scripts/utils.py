import logging
import os
import sys
from typing import Union
import colorlog
import torch

def setup_logging(name: str,
                  log_level: int = logging.INFO
) -> logging.Logger:
    """
    Sets up a logger.

    Parameters
    ----------
    name : str
        The name of the logger.
    log_level : int
        The logging level or description (default is logging.INFO).
    
    Returns
    -------
    logging.Logger
        The logger.
    """
    log = logging.getLogger(name)
    log_formatter = colorlog.ColoredFormatter(
        fmt="[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s]"
        "[%(log_color)s%(levelname)s%(reset)s] - %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={},
    )
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(log_formatter)
    log.addHandler(handler)
    log.setLevel(log_level)
    
    return log

def setup_distributed_environment(device: Union[torch.device, str] = "cuda") -> tuple[bool, int, int]:
    """
    Sets up the distributed environment.

    Based on the available devices, the process group is initialized with the
    appropriate backend. For computation on GPUs the nccl backend optimized for
    NVIDIA GPUs is chosen. For computation on CPUs gloo is used as backend. If
    the program is run without the intention of being distributed, the world_size
    will be set to 1, accordingly the only rank is 0.

    Parameters
    ----------
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).
    
    Returns
    -------
    bool
        Distributed mode enabled or disabled.
    int
        The rank of the current process.
    int 
        The world size or total number of processes.
    """
    log = setup_logging(name="setup environment")

    # Choose backend depending on device type
    device = torch.device(device)
    if device.type == "cuda":
        backend = "nccl"
    else:
        backend = "gloo"
    
    log.info(f"Using device type: {device.type} and backend: {backend}")
    
    # Check if running in distributed mode
    is_distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    log.info(f"Distributed Mode: {'Enabled' if is_distributed else 'Disabled'}")
    
    # Initialize the distributed process group if in distributed mode
    if is_distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        log.info(f"Initializing distributed process group: Rank {rank}/{world_size}")

        torch.distributed.init_process_group(backend=backend, init_method="env://")
        log.info(f"Distributed process group initialized: Rank {rank}, World Size {world_size}")
    else:
        rank = 0
        world_size = 1
        log.info("Running in single-device mode.")
    
    return is_distributed, rank, world_size