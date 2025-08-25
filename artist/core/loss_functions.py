from typing import Any

import torch

from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, utils
from artist.util.environment_setup import get_device


def focal_spot_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    scenario: Scenario,
    target_area_index: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Compute the loss for an optimization using a focal spot coordinate as a target.

    Parameters
    ----------
    predictions : torch.Tensor
        The flux distribution.
        Tensor of shape [1, bitmap_resolution_e, bitmap_resolution_u].
    targets : torch.Tensor
        The desired focal spot.
        Tensor of shape [1, 4].
    scenario : Scenario
        The scenario.
    target_area_index : int
        The index of the target used for the optimization.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The mean squared error loss between the computed focal spot and the target focal spot.
        Tensor of shape [1].
    """
    device = get_device(device=device)

    focal_spot = utils.get_center_of_mass(
        bitmaps=predictions,
        target_centers=scenario.target_areas.centers[target_area_index],
        target_widths=scenario.target_areas.dimensions[target_area_index][0],
        target_heights=scenario.target_areas.dimensions[target_area_index][1],
        device=device,
    )

    loss_function = torch.nn.MSELoss()
    loss = loss_function(
        focal_spot,
        targets,
    )

    return loss


def pixel_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_area_dimensions: torch.Tensor,
    number_of_rays: int,
    device: torch.device | None = None,
):
    """
    Compute the pixel loss during an optimization.

    The computation is performed elementwise over the last two dimensions and
    summed to give the pixel loss. The predictions and targets do not need to
    be normalized and scaled. This function takes care of that internally.

    Parameters
    ----------
    predictions : torch.Tensor
        The flux distribution.
        Tensor of shape [1, bitmap_resolution_e, bitmap_resolution_u].
    targets : torch.Tensor
        The desired focal spot.
        Tensor of shape [1, 4].
    target_area_dimensions : torch.Tensor
        The dimensions of the tower target areas aimed at.
        Tensor of shape [number_of_flux_distributions, 2].
    number_of_rays : int
        The number of rays used to generate the flux.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The pixel loss between the predictions and targets.
        Tensor of shape [number_of_flux_distributions].
    """
    device = get_device(device=device)

    # Normalize the flux distributions.
    normalized_predictions = utils.normalize_bitmaps(
        flux_distributions=predictions,
        target_area_widths=target_area_dimensions[:, 0],
        target_area_heights=target_area_dimensions[:, 1],
        number_of_rays=number_of_rays,
    )
    normalized_targets = utils.normalize_bitmaps(
        flux_distributions=targets,
        target_area_widths=torch.full(
            (targets.shape[0],),
            config_dictionary.utis_crop_width,
            device=device,
        ),
        target_area_heights=torch.full(
            (targets.shape[0],),
            config_dictionary.utis_crop_height,
            device=device,
        ),
        number_of_rays=targets.sum(dim=[1, 2]),
    )
    loss = ((normalized_predictions - normalized_targets) ** 2).mean(dim=(1, 2))

    return loss


def distribution_loss_kl_divergence(
    predictions: torch.Tensor, targets: torch.Tensor, **kwargs: Any
) -> torch.Tensor:
    """
    Compute the loss for an optimization using distributions as target.

    The computation is performed elementwise over the last two dimensions and
    summed to give the kl-divergence as loss. The elements in predictions and targets
    need to be greater or equal to zero. This function internally normalizes both
    tensors.

    Parameters
    ----------
    predictions : torch.Tensor
        The flux distribution.
        Tensor of shape [1, bitmap_resolution_e, bitmap_resolution_u].
    targets : torch.Tensor
        The target distributions.
        Tensor of shape [number_of_flux_distributions, bitmap_resolution_e, bitmap_resolution_u].
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    torch.Tensor
        The KL-divergence loss between the predictions and targets.
        Tensor of shape [number_of_flux_distributions].
    """
    target_distributions = targets / (targets.sum(dim=(1, 2), keepdim=True) + 1e-12)
    flux_shifted = predictions - predictions.min()
    predicted_distributions = flux_shifted / (
        flux_shifted.sum(dim=(1, 2), keepdim=True) + 1e-12
    )

    loss = kl_divergence(
        predictions=target_distributions, targets=predicted_distributions
    )

    return loss


def kl_divergence(
    predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-12
) -> torch.Tensor:
    """
    Compute the Kullback-Leibler divergence D_KL(P || Q) between two distributions.

    The computation is performed elementwise over the last two dimensions and
    summed to give a per-batch divergence value. Both input tensors are assumed
    to be nonnegative but not necessarily normalized. A small constant `epsilon`
    is added to avoid division by zero and log of zero.

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
    return (targets * (torch.log((targets + epsilon) / (predictions + epsilon)))).sum(
        dim=(1, 2)
    )


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
    reference_loss :  torch.Tensor
        The reference loss.
    weight : float
        The weight or ratio used for the scaling.

    Returns
    -------
    torch.Tensor
        The scaled loss.
    """
    epsilon = 1e-12
    scale = (reference_loss * weight) / (loss + epsilon)
    return loss * scale
