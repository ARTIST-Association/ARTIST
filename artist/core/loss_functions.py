from typing import Any, Callable

import torch

from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, utils
from artist.util.environment_setup import get_device


def vector_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    reduction_dimensions: tuple[int] | None,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Compute the MSE loss for an optimization comparing two tensors.

    Parameters
    ----------
    predictions : torch.Tensor
        The predicted tensor.
        Tensor of shape [number_of_samples, 4].
    targets : torch.Tensor
        The target tensor.
        Tensor of shape [number_of_samples, 4].
    reduction_dimensions : tuple[int] | None
        The dimensions to reduce in the final loss.
    **kwargs : Any
        Additional keyword arguments used for specific loss definitions only.

    Returns
    -------
    torch.Tensor
        The mean squared error loss between the predicted and the target tensors.
        Tensor of shape [number_of_samples].
    """
    loss_function = torch.nn.MSELoss(reduction="none")
    loss = loss_function(
        predictions,
        targets,
    )
    reduced_loss = loss.mean(
        *(() if reduction_dimensions is None else (reduction_dimensions,))
    )

    if reduced_loss.dim() == 0:
        reduced_loss = reduced_loss.unsqueeze(0)

    return reduced_loss


def focal_spot_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    scenario: Scenario,
    target_area_mask: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Compute the loss for an optimization using focal spot coordinates as targets.

    Parameters
    ----------
    predictions : torch.Tensor
        The flux distributions.
        Tensor of shape [number_of_flux_distributions, bitmap_resolution_e, bitmap_resolution_u].
    targets : torch.Tensor
        The desired focal spots.
        Tensor of shape [number_of_flux_distributions, 4].
    scenario : Scenario
        The scenario.
    target_area_mask : torch.Tensor
        The target area mapping for the flux distributions.
        Tensor of shape [number_of_flux_distributions].
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The mean squared error loss between the computed focal spot and the target focal spot.
        Tensor of shape [number_of_flux_distributions].
    """
    device = get_device(device=device)

    focal_spot = utils.get_center_of_mass(
        bitmaps=predictions,
        target_centers=scenario.target_areas.centers[target_area_mask],
        target_widths=scenario.target_areas.dimensions[target_area_mask][:, 0],
        target_heights=scenario.target_areas.dimensions[target_area_mask][:, 1],
        device=device,
    )

    loss_function = torch.nn.MSELoss(reduction="none")
    loss = loss_function(
        focal_spot,
        targets,
    )

    return loss.mean(dim=1)


def pixel_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_area_dimensions: torch.Tensor,
    number_of_rays: int,
    device: torch.device | None = None,
):
    """
    Compute the pixel loss between two flux distributions during an optimization.

    The computation is performed elementwise over the last two dimensions and
    summed to give the pixel loss. The predictions and targets do not need to
    be normalized and scaled. This function takes care of that internally.

    Parameters
    ----------
    predictions : torch.Tensor
        The predicted flux distribution.
        Tensor of shape [number_of_flux_distributions, bitmap_resolution_e, bitmap_resolution_u].
    targets : torch.Tensor
        The target flux distribution.
        Tensor of shape [number_of_flux_distributions, bitmap_resolution_e, bitmap_resolution_u].
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
    predictions: torch.Tensor,
    targets: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    r"""
    Compute the loss for an optimization using distributions as target.

    The computation is performed elementwise over the last two dimensions and
    summed to give the Kullback-Leibler divergence (kl-div) as loss. The elements in predictions
    and targets need to be greater or equal to zero. This function internally normalizes both
    tensors.
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
        The flux distribution.
        Tensor of shape [number_of_flux_distributions, bitmap_resolution_e, bitmap_resolution_u].
    targets : torch.Tensor
        The target distributions.
        Tensor of shape [number_of_flux_distributions, bitmap_resolution_e, bitmap_resolution_u].
    **kwargs : Any
        Additional keyword arguments used for specific loss definitions only.

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


def loss_per_heliostat(
    active_heliostats_mask: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_function: Callable[..., torch.Tensor],
    device: torch.device | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Compute mean losses for each heliostat with multiple samples.

    If the active heliostats of one group have different amounts of samples to train on, i.e.
    one heliostat is trained with more samples than another, this function makes sure that
    each heliostat still contributes equally to the overall loss of the group. This function
    has a variable loss function to compute the loss per sample and then computes the mean loss
    for each heliostat.

    Parameters
    ----------
    active_heliostats_mask : torch.Tensor
        A mask defining which heliostats are activated.
        Tensor of shape [number_of_heliostats].
    predictions : torch.Tensor
        The predicted values for all samples from all active heliostats.
        Tensor of shape [number_of_active_heliostats, bitmap_resolution_e, bitmap_resolution_u].
    targets : torch.Tensor
        The target values for all samples from all active heliostats.
        Tensor of shape [number_of_active_heliostats, bitmap_resolution_e, bitmap_resolution_u].
    loss_function : Callable[..., torch.Tensor]
        A callable function that computes the loss. It accepts predictions and targets
        and optionally other keyword arguments and return a tensor with loss values.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.
    **kwargs : Any
        Additional keyword arguments used for specific loss definitions only.

    Returns
    -------
    torch.Tensor
        The mean loss per heliostat.
        Tensor of shape [number_of_active_heliostats].
    """
    device = get_device(device=device)

    # Compute per-sample losses.
    per_sample_losses = loss_function(
        predictions=predictions,
        targets=targets,
        device=device,
        **kwargs,
    )

    # A sample to heliostat index mapping.
    heliostat_ids = torch.repeat_interleave(
        torch.arange(len(active_heliostats_mask), device=device),
        active_heliostats_mask,
    )

    loss_sum_per_heliostat = torch.zeros(len(active_heliostats_mask), device=device)
    loss_sum_per_heliostat = loss_sum_per_heliostat.index_add(
        0, heliostat_ids, per_sample_losses
    )

    # Compute mean MSE per heliostat on each rank.
    number_of_samples_per_heliostat = torch.zeros(
        len(active_heliostats_mask), device=device
    )
    number_of_samples_per_heliostat.index_add_(
        0, heliostat_ids, torch.ones_like(per_sample_losses, device=device)
    )

    counts_clamped = number_of_samples_per_heliostat.clamp_min(1.0)
    mean_loss_per_heliostat = loss_sum_per_heliostat / counts_clamped
    mean_loss_per_heliostat = mean_loss_per_heliostat * (
        number_of_samples_per_heliostat > 0
    )

    return mean_loss_per_heliostat


def total_variation_loss(
    surfaces: torch.Tensor,
    number_of_neighbors: int = 20,
    sigma: float | None = None,
    batch_size: int = 512,
    epsilon: float = 1e-8,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Compute the total variation loss for facetted surfaces.

    This loss term can be used as an addition to the overall loss during the surface reconstruction
    optimization. It supresses the noise in the surface. It measures the noise in the surface by
    taking absolute differences in the z values of the provided points. This loss implementation
    focuses on local smoothness by applying a Gaussian distance weight and thereby letting
    closer points contribute more. This loss implementation is batched and can handle multiple
    surfaces which are further batched in facets.

    Parameters
    ----------
    surfaces : torch.Tensor
        The surfaces.
        Tensor of shape [number_of_active_heliostats, number_of_facets_per_surface, number_of_surface_points_per_facet, 4].
    number_of_neighbors : int
        The number of nearest neighbors to consider (default is 20).
    sigma : float | None
        Determines how quickly the weight falls off as the distance increases (default is None).
    batch_size : int
        Used to process smaller batches of points instead of creating full distance matrices for all points (default is 512).
    epsilon : float
        A small vlaue used to prevent divisions by zero (defualt is 1e-8).
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The total variation loss for all provided surfaces.
        Tensor of shape [number_of_active_heliostats, number_of_facets_per_surface].
    """
    device = get_device(device=device)

    number_of_surfaces, number_of_facets, number_of_surface_points_per_facet, _ = (
        surfaces.shape
    )
    coordinates = surfaces[:, :, :, :2]
    z_values = surfaces[:, :, :, 2]

    if sigma is None:
        coordinates_std = coordinates.std(dim=1).mean().item()
        sigma = max(coordinates_std * 0.1, 1e-6)
    sigma = float(sigma + 1e-12)

    variation_loss_sum = torch.zeros(
        (number_of_surfaces, number_of_facets), device=device
    )
    number_of_valid_neighbors = torch.zeros(
        (number_of_surfaces, number_of_facets), device=device
    )

    # Iterate over query points in batches to limit memory usage.
    for start_index in range(0, number_of_surface_points_per_facet, batch_size):
        end_index = min(start_index + batch_size, number_of_surface_points_per_facet)
        number_of_points_in_batch = end_index - start_index

        batch_coordinates = coordinates[:, :, start_index:end_index, :]
        batch_z_values = z_values[:, :, start_index:end_index]

        # Compute pairwise distances between the current batch coordinates and all coordinates and exclude identities.
        distances = torch.cdist(batch_coordinates, coordinates)
        rows = torch.arange(number_of_points_in_batch, device=device)
        cols = (start_index + rows).to(device)
        self_mask = torch.zeros_like(distances, dtype=torch.bool)
        self_mask[:, :, rows, cols] = True
        masked_distances = torch.where(
            self_mask, torch.full_like(distances, 1e9), distances
        )

        # Select the k nearest neighbors (or fewer if the coordinate is near an edge).
        number_of_neighbors_to_select = min(
            number_of_neighbors, number_of_surface_points_per_facet - 1
        )
        selected_distances, selected_indices = torch.topk(
            masked_distances, number_of_neighbors_to_select, largest=False, dim=3
        )
        valid_mask = selected_distances < 1e9

        # Get all z_values of the selected neighbors and the absolute z_value_variations.
        z_values_neighbors = torch.gather(
            z_values.unsqueeze(2).expand(-1, -1, number_of_points_in_batch, -1),
            3,
            selected_indices,
        )
        z_value_variations = torch.abs(
            batch_z_values.unsqueeze(-1) - z_values_neighbors
        )
        z_value_variations = z_value_variations * valid_mask.type_as(z_value_variations)

        # Accumulate weighted z_value_variations.
        weights = torch.exp(-0.5 * (selected_distances / sigma) ** 2)
        weights = weights * valid_mask.type_as(weights)
        variation_loss_sum = variation_loss_sum + (weights * z_value_variations).sum(
            dim=(2, 3)
        )
        number_of_valid_neighbors = number_of_valid_neighbors + valid_mask.type_as(
            z_value_variations
        ).sum(dim=(2, 3))

    # Batched total variation losses.
    variation_loss_final = variation_loss_sum / (number_of_valid_neighbors + epsilon)

    return variation_loss_final
