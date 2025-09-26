from typing import Any

import torch

from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, utils
from artist.util.environment_setup import get_device


class Loss:
    """
    Abstract base class for all loss functions.

    Attributes
    ----------
    loss_function : torch.nn.Module
        A torch module implementing a loss.
    """

    def __init__(self, loss_function: torch.nn.Module) -> None:
        """
        Initialize the the base loss.

        Parameters
        ----------
        loss_function : torch.nn.Module
            A torch module implementing a loss.
        """
        self.loss_function = loss_function

    def __call__(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""
        Compute the loss.

        Parameters
        ----------
        prediction : torch.Tensor
            The predicted values.
            Tensor of variable shape.
        ground_truth : torch.Tensor
            The ground truth.
            Tensor of variable shape.
        \*\*kwargs : Any
            Keyword arguments.

        Raises
        ------
        NotImplementedError
            This abstract method must be overridden.
        """
        raise NotImplementedError("Must be overridden!")


class VectorLoss(Loss):
    """
    A loss defined as the elementwise squared distance (Euclidean distance) between predicted vectors and the ground truth.

    Attributes
    ----------
    loss_function : torch.nn.Module
        A torch module implementing a loss.
    """

    def __init__(self) -> None:
        """Initialize the vector loss."""
        super().__init__(loss_function=torch.nn.MSELoss(reduction="none"))

    def __call__(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""
        Compute the vector loss.

        Parameters
        ----------
        prediction : torch.Tensor
            The predicted values.
            Tensor of shape [number_of_samples, ...].
        ground_truth : torch.Tensor
            The ground truth.
            Tensor of shape [number_of_samples, ...].
        \*\*kwargs : Any
            Keyword arguments.
            The ``reduction_dimensions`` is an expected keyword argument for the vector loss.

        Raises
        ------
        ValueError
            If expected keyword arguments are not passed.

        Returns
        -------
        torch.Tensor
            The summed MSE vector loss reduced along the specified dimensions.
            Tensor of shape [number_of_samples].
        """
        expected_kwargs = ["reduction_dimensions"]
        for key in expected_kwargs:
            if key not in kwargs:
                raise ValueError(
                    f"The vector loss expects {key} as keyword argument. Please add this argument."
                )

        loss = self.loss_function(prediction, ground_truth)

        return loss.sum(dim=kwargs["reduction_dimensions"])


class FocalSpotLoss(Loss):
    """
    A loss defined as the elementwise squared distance (Euclidean distance) between predicted focal spots and the ground truth.

    Attributes
    ----------
    loss_function : torch.nn.Module
        A torch module implementing a loss.
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
        super().__init__(loss_function=torch.nn.MSELoss(reduction="none"))
        self.scenario = scenario

    def __call__(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""
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
        \*\*kwargs : Any
            Keyword arguments.
            The ``reduction_dimensions``, ``target_area_mask`` and optionally ``device`` are expected keyword arguments for the focal spot loss.

        Raises
        ------
        ValueError
            If expected keyword arguments are not passed.

        Returns
        -------
        torch.Tensor
            The summed MSE focal spot loss reduced along the specified dimensions.
            Tensor of shape [number_of_samples].
        """
        expected_kwargs = ["reduction_dimensions", "device", "target_area_mask"]
        errors = []
        for key in expected_kwargs:
            if key not in kwargs:
                errors.append(f"Please add {key} as keyword argument.")
        if errors:
            raise ValueError(
                f"The focal spot loss expects {expected_kwargs} as keyword arguments. "
                + " ".join(errors)
            )

        device = get_device(device=kwargs["device"])

        target_area_mask = kwargs["target_area_mask"]

        focal_spot = utils.get_center_of_mass(
            bitmaps=prediction,
            target_centers=self.scenario.target_areas.centers[target_area_mask],
            target_widths=self.scenario.target_areas.dimensions[target_area_mask][:, 0],
            target_heights=self.scenario.target_areas.dimensions[target_area_mask][
                :, 1
            ],
            device=device,
        )

        loss = self.loss_function(focal_spot, ground_truth)

        return loss.sum(dim=kwargs["reduction_dimensions"])


class PixelLoss(Loss):
    """
    A loss defined as the elementwise squared distance (Euclidean distance) between each pixel of predicted bitmaps and the ground truth.

    Attributes
    ----------
    loss_function : torch.nn.Module
        A torch module implementing a loss.
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
        super().__init__(loss_function=torch.nn.MSELoss(reduction="none"))
        self.scenario = scenario

    def __call__(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""
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
        \*\*kwargs : Any
            Keyword arguments.
            The ``reduction_dimensions``, ``target_area_mask`` and optionally ``device`` are expected keyword arguments for the pixel loss.

        Raises
        ------
        ValueError
            If expected keyword arguments are not passed.

        Returns
        -------
        torch.Tensor
            The summed MSE pixel loss reduced along the specified dimensions.
            Tensor of shape [number_of_samples].
        """
        expected_kwargs = ["reduction_dimensions", "device", "target_area_mask"]
        errors = []
        for key in expected_kwargs:
            if key not in kwargs:
                errors.append(f"Please add {key} as keyword argument.")
        if errors:
            raise ValueError(
                f"The vector loss expects {expected_kwargs} as keyword arguments. "
                + " ".join(errors)
            )

        device = get_device(device=kwargs["device"])

        target_area_mask = kwargs["target_area_mask"]

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

        loss = self.loss_function(normalized_predictions, normalized_ground_truth)

        return loss.sum(dim=kwargs["reduction_dimensions"])


class KLDivergenceLoss(Loss):
    """
    A loss defined as the Kullback-Leibler divergence between predicted values and the ground truth.

    Attributes
    ----------
    loss_function : torch.nn.Module
        A torch module implementing a loss.
    """

    def __init__(self) -> None:
        """Initialize the Kullback-Leibler divergence loss."""
        super().__init__(
            loss_function=torch.nn.KLDivLoss(reduction="none", log_target=True)
        )

    def __call__(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""
        Compute the Kullback-Leibler divergence loss :math:`D_{\mathrm{KL}}(P \parallel Q)`.

        The elements in the prediction and ground truth are normalized and shifted, to be greater or
        equal to zero. The kl-divergence is defined by:

        .. math::

            D_{\mathrm{KL}}(P \parallel Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)},

        where :math:`P` is the ground truth distribution and :math:`Q` is the approximation or prediction
        of :math:`Q`. The kl-divergence is an asymmetric function. Switching :math:`P` and :math:`Q`
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
        \*\*kwargs : Any
            Keyword arguments.
            The ``reduction_dimensions`` is an expected keyword argument for the kl-divergence loss.

        Raises
        ------
        ValueError
            If expected keyword arguments are not passed.

        Returns
        -------
        torch.Tensor
            The summed kl-divergence loss reduced along the specified dimensions.
            Tensor of shape [number_of_samples].
        """
        expected_kwargs = ["reduction_dimensions"]
        for key in expected_kwargs:
            if key not in kwargs:
                raise ValueError(
                    f"The kl-divergence loss expects {key} as keyword argument. Please add this argument."
                )

        if ground_truth.min() < 0:
            ground_truth = ground_truth - ground_truth.min()
        ground_truth_distributions = (
            ground_truth / (ground_truth.sum(dim=(1, 2), keepdim=True) + 1e-12)
        ) + 1e-12
        if prediction.min() < 0:
            prediction = prediction - prediction.min()
        predicted_distributions = (
            prediction / (prediction.sum(dim=(1, 2), keepdim=True) + 1e-12)
        ) + 1e-12

        log_input = torch.log(predicted_distributions)
        log_target = torch.log(ground_truth_distributions)

        loss = self.loss_function(log_target, log_input)

        return loss.sum(dim=kwargs["reduction_dimensions"])


class AngleLoss(Loss):
    """
    A loss defined as the angular difference between the prediction and ground truth.

    Attributes
    ----------
    loss_function : torch.nn.Module
        A torch module implementing a loss.
    """

    def __init__(self) -> None:
        """Initialize the angle loss."""
        super().__init__(loss_function=torch.nn.CosineSimilarity(dim=-1))

    def __call__(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""
        Compute the cosine similarity between the prediction and ground truth.

        Parameters
        ----------
        prediction : torch.Tensor
            The predicted values.
            Tensor of shape [number_of_samples, 4].
        ground_truth : torch.Tensor
            The ground truth.
            Tensor of shape [number_of_samples, 4].
        \*\*kwargs : Any
            Keyword arguments.

        Returns
        -------
        torch.Tensor
            The summed loss reduced along the specified dimensions.
            Tensor of shape [number_of_samples].
        """
        cos_sim = self.loss_function(prediction, ground_truth)

        loss = 1.0 - cos_sim

        return loss
