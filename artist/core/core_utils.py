import torch


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


def scale_loss(
    loss: torch.Tensor,
    reference_loss: torch.Tensor,
    weight: float,
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
    scale = (reference_loss * weight) / (loss + epsilon)
    return loss * scale
