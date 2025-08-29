from typing import Any

import torch

from artist.core.core_utils import kl_divergence
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, utils
from artist.util.environment_setup import get_device


class BaseLoss:
    """Abstract base class for all loss functions."""

    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute the loss.

        This method must be implemented by subclasses.
        
        Parameters
        ----------
        predictions : torch.Tensor
            The predictions.
        ...
        """
        raise NotImplementedError("Subclasses must implement the __call__ method.")

    def loss_per_heliostat(
        self,
        per_sample_loss: torch.Tensor,
        active_heliostats_mask: torch.Tensor,
        device: torch.device | None = None,
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
            0, heliostat_ids, per_sample_loss
        )

        # Compute mean MSE per heliostat on each rank.
        number_of_samples_per_heliostat = torch.zeros(
            len(active_heliostats_mask), device=device
        )
        number_of_samples_per_heliostat.index_add_(
            0, heliostat_ids, torch.ones_like(per_sample_loss, device=device)
        )

        counts_clamped = number_of_samples_per_heliostat.clamp_min(1.0)
        mean_loss_per_heliostat = loss_sum_per_heliostat / counts_clamped
        mean_loss_per_heliostat = mean_loss_per_heliostat * (
            number_of_samples_per_heliostat > 0
        )

        return mean_loss_per_heliostat


# TODO: explain: use .sum() as reduction (of mse) because 4d tensors and last element is 0 or change to mean?

class VectorLoss(BaseLoss):
    """
    Implement the vector loss.

    Args:
        BaseLoss (_type_): _description_
    """

    def __call__(self, prediction: torch.Tensor, ground_truth: torch.Tensor, target_area_mask: torch.Tensor, reduction_dimensions: tuple[int], device: torch.device | None, **kwargs: Any) -> torch.Tensor:
        loss_function = torch.nn.MSELoss(reduction="none")
        loss = loss_function(prediction, ground_truth)

        return loss.sum(dim=reduction_dimensions)

class FocalSpotLoss(BaseLoss):
    """
    Implement the focal spot loss.

    Args:
        BaseLoss (_type_): _description_
    """

    def __init__(self, scenario: Scenario):
        self.scenario = scenario

    def __call__(self, prediction: torch.Tensor, ground_truth: torch.Tensor, target_area_mask: torch.Tensor, reduction_dimensions: tuple[int], device: torch.device | None, **kwargs: Any) -> torch.Tensor:
        device = get_device(device=device)

        focal_spot = utils.get_center_of_mass(
            bitmaps=prediction,
            target_centers=self.scenario.target_areas.centers[target_area_mask],
            target_widths=self.scenario.target_areas.dimensions[target_area_mask][:, 0],
            target_heights=self.scenario.target_areas.dimensions[target_area_mask][:, 1],
            device=device,
        )
        loss_function = torch.nn.MSELoss(reduction="none")
        loss = loss_function(focal_spot, ground_truth)
        
        return loss.sum(dim=reduction_dimensions)
    

class PixelLoss(BaseLoss):
    """
    Implement the focal spot loss.

    Args:
        BaseLoss (_type_): _description_
    """

    def __init__(self, scenario: Scenario):
        self.scenario = scenario

    def __call__(self, prediction: torch.Tensor, ground_truth: torch.Tensor, target_area_mask: torch.Tensor, reduction_dimensions: tuple[int], device: torch.device | None, **kwargs: Any) -> torch.Tensor:
        device = get_device(device=device)
        
        normalized_predictions = utils.normalize_bitmaps(
            flux_distributions=prediction,
            target_area_widths=self.scenario.target_areas.dimensions[target_area_mask][:, 0],
            target_area_heights=self.scenario.target_areas.dimensions[target_area_mask][:, 1],
            number_of_rays=self.scenario.light_sources.light_source_list[0].number_of_rays,
        )
        normalized_ground_truth = utils.normalize_bitmaps(
            flux_distributions=ground_truth,
            target_area_widths=torch.full(
                (ground_truth.shape[0],),
                config_dictionary.utis_crop_width,
                device=device,
            ),
            target_area_heights=torch.full(
                (ground_truth.shape[0],),
                config_dictionary.utis_crop_height,
                device=device,
            ),
            number_of_rays=ground_truth.sum(dim=[1, 2]),
        )
        
        loss_function = torch.nn.MSELoss(reduction="none")
        loss = loss_function(normalized_predictions, normalized_ground_truth)

        return loss.sum(dim=reduction_dimensions)
    

class KLDivergenceLoss(BaseLoss):
    """
    Implement the focal spot loss.

    Args:
        BaseLoss (_type_): _description_
    """

    def __call__(self, prediction: torch.Tensor, ground_truth: torch.Tensor, target_area_mask: torch.Tensor, reduction_dimensions: tuple[int], device: torch.device | None, **kwargs: Any) -> torch.Tensor:
        ground_truth_distributions = ground_truth / (ground_truth.sum(dim=(1, 2), keepdim=True) + 1e-12)
        flux_shifted = prediction - prediction.min()
        predicted_distributions = flux_shifted / (
            flux_shifted.sum(dim=(1, 2), keepdim=True) + 1e-12
        )

        loss = kl_divergence(
            predictions=ground_truth_distributions, targets=predicted_distributions
        )

        return loss.sum(dim=reduction_dimensions)
 	
    
