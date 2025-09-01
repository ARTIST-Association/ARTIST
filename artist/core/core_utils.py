import torch

from artist.util.environment_setup import get_device


def per_heliostat_reduction(
    per_sample_values: torch.Tensor,
    active_heliostats_mask: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Compute mean losses for each heliostat with multiple samples.

    If the active heliostats of one group have different amounts of samples to train on, i.e.
    one heliostat is trained with more samples than another, this function makes sure that
    each heliostat still contributes equally to the overall loss of the group. This function
    computes the mean loss for each heliostat.

    Parameters
    ----------
    per_sample_values : torch.Tensor
        The per sample values to be reduced.
        Tensor of shape [number_of_active_heliostats, ...]
    active_heliostats_mask : torch.Tensor
        A mask defining which heliostats are activated.
        Tensor of shape [number_of_heliostats].
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The mean loss per heliostat.
        Tensor of shape [number_of_active_heliostats].
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

    # Compute mean MSE per heliostat on each rank.
    number_of_samples_per_heliostat = torch.zeros(
        len(active_heliostats_mask), device=device
    )
    number_of_samples_per_heliostat.index_add_(
        0, heliostat_ids, torch.ones_like(per_sample_values, device=device)
    )

    counts_clamped = number_of_samples_per_heliostat.clamp_min(1.0)
    mean_loss_per_heliostat = loss_sum_per_heliostat / counts_clamped
    mean_loss_per_heliostat = mean_loss_per_heliostat * (
        number_of_samples_per_heliostat > 0
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
    reference_loss :  torch.Tensor
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


def kl_divergence(
    predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-12
) -> torch.Tensor:
    r"""
    Compute the Kullback-Leibler divergence :math:`D_{\mathrm{KL}}(P \parallel Q)` between two distributions.

    The computation is performed elementwise over the last two dimensions and
    summed to give a per-batch divergence value. Both input tensors are assumed
    to be nonnegative but not necessarily normalized. A small constant `epsilon`
    is added to avoid division by zero and log of zero.
    The kl-divergence is defined by:

    .. math::

        D_{\mathrm{KL}}(P \parallel Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)},

    where :math:`P` is the target distribution and :math:`Q` is the approximation or prediction
    of :math:`Q`. The kl-divergence is an asymetric function. Switching :math:`P` and :math:`Q`
    has the following effect:
    :math:`P \parallel Q` Penalizes extra mass in the prediction where the target has none.
    :math:`Q \parallel P` Penalizes missing mass in the prediction where the target has mass.

    Parameters
    ----------
    predictions : torch.Tensor
        The predicted distributions.
        Tensor of shape [number_of_flux_distributions, bitmap_resolution_e, bitmap_resolution_u].
    targets : torch.Tensor
        The target distributions.
        Tensor of shape [number_of_flux_distributions, bitmap_resolution_e, bitmap_resolution_u].

    Returns
    -------
    torch.Tensor
        The kl-divergence for each distribution.
        Tensor of shape [number_of_flux_distributions].
    """
    return targets * (torch.log((targets + epsilon) / (predictions + epsilon)))
