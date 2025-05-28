
from typing import Generator, Union

import torch


def setup_global_distributed_environment(
    device: Union[torch.device, str] = "cuda",
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
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

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
    device = torch.device(device)
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
            print(f"Using device type: {device.type} and backend: {backend}.")
            print(f"Distributed Mode: {'Enabled.' if is_distributed else 'Disabled.'}")
        print(
            f"Distributed process group initialized: Rank {rank}, World Size {world_size}"
        )

    except Exception:
        print(f"Using device type: {device.type} and backend: {backend}.")
        print("Running in single-device mode.")

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
