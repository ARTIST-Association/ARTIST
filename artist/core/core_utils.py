import torch

from artist.util.environment_setup import get_device


def per_heliostat_reduction(
    per_sample_values: torch.Tensor,
    active_heliostats_mask: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Compute mean losses for each heliostat with multiple samples.

    If the active heliostats of one group have different amounts of samples to train on, i.e.,
    one heliostat is trained with more samples than another, this function makes sure that
    each heliostat still contributes equally to the overall loss of the group. This function
    computes the mean loss for each heliostat.

    Parameters
    ----------
    per_sample_values : torch.Tensor
        The per sample values to be reduced.
        Tensor of shape [number_of_samples].
    active_heliostats_mask : torch.Tensor
        A mask defining which heliostats are activated.
        Tensor of shape [number_of_heliostats].
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The mean loss per heliostat.
        Tensor of shape [number_of_heliostats].
    """
    device = get_device(device=device)

    # A sample to heliostat index mapping.
    heliostat_ids = torch.repeat_interleave(
        torch.arange(len(active_heliostats_mask), device=device),
        active_heliostats_mask,
    )

    loss_sum_per_heliostat = torch.zeros(len(active_heliostats_mask), device=device)
    loss_sum_per_heliostat = loss_sum_per_heliostat.index_add(
        0, heliostat_ids, per_sample_values
    )

    # Compute MSE loss per heliostat on each rank.
    number_of_samples_per_heliostat = torch.zeros(
        len(active_heliostats_mask), device=device
    )
    number_of_samples_per_heliostat.index_add_(
        0, heliostat_ids, torch.ones_like(per_sample_values, device=device)
    )

    counts_clamped = number_of_samples_per_heliostat.clamp_min(1.0)
    mean_loss_per_heliostat = loss_sum_per_heliostat / counts_clamped
    mean_loss_per_heliostat = torch.where(
        number_of_samples_per_heliostat > 0, mean_loss_per_heliostat, torch.inf
    )

    return mean_loss_per_heliostat


def scale_loss(
    loss: torch.Tensor, reference: torch.Tensor, weight: float
) -> torch.Tensor:
    """
    Scale one loss so that its weighted contribution is a ratio of the reference loss.

    Parameters
    ----------
    loss : torch.Tensor
        The loss to be scaled.
        Tensor of shape [1].
    reference : torch.Tensor
        The reference loss.
        Tensor of shape [1].
    weight : float
        The weight or ratio used for the scaling.

    Returns
    -------
    torch.Tensor
        The scaled loss.
        Tensor of shape [1].
    """
    epsilon = 1e-12
    scale = (reference * weight) / (loss + epsilon)
    return loss * scale
