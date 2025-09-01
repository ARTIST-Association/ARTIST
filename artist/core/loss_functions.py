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
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        target_area_mask: torch.Tensor,
        reduction_dimensions: tuple[int],
        device: torch.device | None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute the loss.

        Parameters
        ----------
        prediction : torch.Tensor
            The predicted values.
        ground_truth : torch.Tensor
            The ground truth.
        target_area_mask : torch.Tensor
            The indices of target areas corresponding to each sample.
            Tensor of shape [number_of_samples].
        reduction_dimensions : tuple[int]
            The dimensions along which to reduce the final loss.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Raises
        ------
        NotImplementedError
            This abstract method must be overridden.
        """
        raise NotImplementedError("Must be overridden!")


class VectorLoss(BaseLoss):
    """A loss defined as the elementwise squared distance (Euclidean distance) between predicted vectors and the ground truth."""

    def __call__(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        target_area_mask: torch.Tensor,
        reduction_dimensions: tuple[int],
        device: torch.device | None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute the vector loss.

        Parameters
        ----------
        prediction : torch.Tensor
            The predicted values.
            Tensor of shape [number_of_samples, 4].
        ground_truth : torch.Tensor
            The ground truth.
            Tensor of shape [number_of_samples, 4].
        target_area_mask : torch.Tensor
            The indices of target areas corresponding to each sample.
            Tensor of shape [number_of_samples].
        reduction_dimensions : tuple[int]
            The dimensions along which to reduce the final loss.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The summed MSE vector loss reduced along the specified dimensions.
        """
        loss_function = torch.nn.MSELoss(reduction="none")
        loss = loss_function(prediction, ground_truth)

        return loss.sum(dim=reduction_dimensions)


class FocalSpotLoss(BaseLoss):
    """
    A loss defined as the elementwise squared distance (Euclidean distance) between predicted focal spots and the ground truth.

    Attributes
    ----------
    scenario : Scenario
        The scenario.
    """

    def __init__(self, scenario: Scenario) -> None:
        """
        Initialize the focal spot loss.

        Parameters
        ----------
        scenario : Scenario
            The scenario.
        """
        self.scenario = scenario

    def __call__(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        target_area_mask: torch.Tensor,
        reduction_dimensions: tuple[int],
        device: torch.device | None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute the focal spot loss.

        First the focal spots of the prediction are computed, then the loss is computed and reduced
        along the specified dimensions.

        Parameters
        ----------
        prediction : torch.Tensor
            The predicted values.
            Tensor of shape [number_of_samples, bitmap_resolution_e, bitmap_resolution_u].
        ground_truth : torch.Tensor
            The ground truth.
            Tensor of shape [number_of_samples, 4].
        target_area_mask : torch.Tensor
            The indices of target areas corresponding to each sample.
            Tensor of shape [number_of_samples].
        reduction_dimensions : tuple[int]
            The dimensions along which to reduce the final loss.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The summed MSE focal spot loss reduced along the specified dimensions.
        """
        device = get_device(device=device)

        focal_spot = utils.get_center_of_mass(
            bitmaps=prediction,
            target_centers=self.scenario.target_areas.centers[target_area_mask],
            target_widths=self.scenario.target_areas.dimensions[target_area_mask][:, 0],
            target_heights=self.scenario.target_areas.dimensions[target_area_mask][
                :, 1
            ],
            device=device,
        )
        loss_function = torch.nn.MSELoss(reduction="none")
        loss = loss_function(focal_spot, ground_truth)

        return loss.sum(dim=reduction_dimensions)


class PixelLoss(BaseLoss):
    """
    A loss defined as the elementwise squared distance (Euclidean distance) between each pixel of predicted bitmaps and the ground truth.

    Attributes
    ----------
    scenario : Scenario
        The scenario.
    """

    def __init__(self, scenario: Scenario) -> None:
        """
        Initialize the pixel loss.

        Parameters
        ----------
        scenario : Scenario
            The scenario.
        """
        self.scenario = scenario

    def __call__(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        target_area_mask: torch.Tensor,
        reduction_dimensions: tuple[int],
        device: torch.device | None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute the pixel loss.

        First the predicted bitmaps and the ground truth are normalized, then the loss is
        computed and reduced along the specified dimensions.

        Parameters
        ----------
        prediction : torch.Tensor
            The predicted values.
            Tensor of shape [number_of_samples, bitmap_resolution_e, bitmap_resolution_u].
        ground_truth : torch.Tensor
            The ground truth.
            Tensor of shape [number_of_samples, bitmap_resolution_e, bitmap_resolution_u].
        target_area_mask : torch.Tensor
            The indices of target areas corresponding to each sample.
            Tensor of shape [number_of_samples].
        reduction_dimensions : tuple[int]
            The dimensions along which to reduce the final loss.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The summed MSE pixel loss reduced along the specified dimensions.
        """
        device = get_device(device=device)

        normalized_predictions = utils.normalize_bitmaps(
            flux_distributions=prediction,
            target_area_widths=self.scenario.target_areas.dimensions[target_area_mask][
                :, 0
            ],
            target_area_heights=self.scenario.target_areas.dimensions[target_area_mask][
                :, 1
            ],
            number_of_rays=self.scenario.light_sources.light_source_list[
                0
            ].number_of_rays,
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
    """A loss defined as the Kullback-Leibler divergence between predicted values and the ground truth."""

    def __call__(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        target_area_mask: torch.Tensor,
        reduction_dimensions: tuple[int],
        device: torch.device | None,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""
        Compute the Kullback-Leibler divergence loss.

        The elements in the prediction and ground truth are normalized and shifted, to be greater or
        equal to zero. The kl-divergence is defined by:

        .. math::

            D_{\mathrm{KL}}(P \parallel Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)},

        where :math:`P` is the ground truth distribution and :math:`Q` is the approximation or prediction
        of :math:`Q`. The kl-divergence is an asymetric function. Switching :math:`P` and :math:`Q`
        has the following effect:
        :math:`P \parallel Q` Penalizes extra mass in the prediction where the ground truth has none.
        :math:`Q \parallel P` Penalizes missing mass in the prediction where the ground truth has mass.

        Parameters
        ----------
        prediction : torch.Tensor
            The predicted values.
            Tensor of shape [number_of_samples, bitmap_resolution_e, bitmap_resolution_u].
        ground_truth : torch.Tensor
            The ground truth.
            Tensor of shape [number_of_samples, bitmap_resolution_e, bitmap_resolution_u].
        target_area_mask : torch.Tensor
            The indices of target areas corresponding to each sample.
            Tensor of shape [number_of_samples].
        reduction_dimensions : tuple[int]
            The dimensions along which to reduce the final loss.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The summed kl-divergence loss reduced along the specified dimensions.
        """
        ground_truth_distributions = ground_truth / (
            ground_truth.sum(dim=(1, 2), keepdim=True) + 1e-12
        )
        flux_shifted = prediction - prediction.min()
        predicted_distributions = flux_shifted / (
            flux_shifted.sum(dim=(1, 2), keepdim=True) + 1e-12
        )

        loss = kl_divergence(
            predictions=ground_truth_distributions, targets=predicted_distributions
        )

        return loss.sum(dim=reduction_dimensions)
