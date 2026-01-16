import torch

from artist.util.environment_setup import get_device


def mean_loss_per_heliostat(
    loss_per_sample: torch.Tensor,
    number_of_samples_per_heliostat: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Calculate the mean loss per heliostat from a loss per sample.

    Parameters
    ----------
    loss_per_sample : torch.Tensor
        Loss per sample.
        Tensor of shape [number_of_samples].
    number_of_samples_per_heliostat : int
        Number of samples per heliostat.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        Loss per heliostat.
        Tensor of shape [number_of_heliostats].
    """
    device = get_device(device=device)

    number_of_chunks = int(loss_per_sample.numel() // number_of_samples_per_heliostat)
    loss_per_sample = loss_per_sample[
        : number_of_chunks * number_of_samples_per_heliostat
    ]

    loss_reshaped = loss_per_sample.view(
        number_of_chunks, number_of_samples_per_heliostat
    )

    mean_loss_per_heliostat = loss_reshaped.mean(dim=1)

    return mean_loss_per_heliostat
