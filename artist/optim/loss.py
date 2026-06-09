from typing import Any, Callable

import torch

from artist.flux import bitmap
from artist.geometry import coordinates
from artist.scenario.scenario import Scenario
from artist.util import indices
from artist.util.env import get_device


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
        Initialize the base loss.

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
            Shape is variable.
        ground_truth : torch.Tensor
            The ground truth.
            Shape is variable.
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

    See Also
    --------
    :class:`Loss` : Reference to the parent class.
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
            Shape is ``[number_of_samples, ...]``.
        ground_truth : torch.Tensor
            The ground truth.
            Shape is ``[number_of_samples, ...]``.
        \*\*kwargs : Any
            Keyword arguments.
            ``reduction_dimensions`` is an expected keyword argument for the vector loss.

        Raises
        ------
        ValueError
            If expected keyword arguments are not passed.

        Returns
        -------
        torch.Tensor
            The summed MSE vector loss reduced along the specified dimensions.
            Shape is ``[number_of_samples]``.
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
    A loss defined as Euclidean distance between the predicted focal spot coordinate and the ground-truth focal spot coordinate.

    Attributes
    ----------
    loss_function : torch.nn.Module
        A torch module implementing a loss.
    scenario : Scenario
        The scenario.

    See Also
    --------
    :class:`Loss` : Reference to the parent class.
    """

    def __init__(self, scenario: Scenario) -> None:
        """
        Initialize the focal spot loss.

        Parameters
        ----------
        scenario : Scenario
            The scenario.
        """
        super().__init__(loss_function=None)
        self.scenario = scenario

    def __call__(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""
        Compute the focal spot loss.

        First the focal spots of the prediction and ground truth flux maps are computed, then the loss is computed and reduced
        along the specified dimensions.

        Parameters
        ----------
        prediction : torch.Tensor
            The predicted values.
            Shape is ``[number_of_samples, bitmap_resolution_e, bitmap_resolution_u]``.
        ground_truth : torch.Tensor
            The ground truth.
            Shape is ``[number_of_samples, 4]``.
        \*\*kwargs : Any
            Keyword arguments.
            ``target_area_indices`` and ``device`` are expected keyword arguments for the focal spot loss.

        Raises
        ------
        ValueError
            If expected keyword arguments are not passed.

        Returns
        -------
        torch.Tensor
            The focal spot loss.
            Shape is ``[number_of_samples]``.
        """
        expected_kwargs = ["device", "target_area_indices"]
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

        target_area_indices = kwargs["target_area_indices"]

        focal_spots_bitmap = bitmap.get_center_of_mass(
            bitmaps=prediction,
            device=device,
        )

        focal_spot_coordinates_prediction = (
            coordinates.bitmap_coordinates_to_target_coordinates(
                bitmap_coordinates=focal_spots_bitmap,
                bitmap_resolution=torch.tensor(
                    [
                        prediction.shape[indices.batched_bitmap_u],
                        prediction.shape[indices.batched_bitmap_e],
                    ],
                    device=device,
                ),
                solar_tower=self.scenario.solar_tower,
                target_area_indices=target_area_indices,
                device=device,
            )
        )

        focal_spots_ground_truth = bitmap.get_center_of_mass(
            bitmaps=ground_truth,
            device=device,
        )

        focal_spot_coordinates_ground_truth = (
            coordinates.bitmap_coordinates_to_target_coordinates(
                bitmap_coordinates=focal_spots_ground_truth,
                bitmap_resolution=torch.tensor(
                    [
                        prediction.shape[indices.batched_bitmap_u],
                        prediction.shape[indices.batched_bitmap_e],
                    ],
                    device=device,
                ),
                solar_tower=self.scenario.solar_tower,
                target_area_indices=target_area_indices,
                device=device,
            )
        )

        return torch.norm(
            focal_spot_coordinates_prediction[:, :3]
            - focal_spot_coordinates_ground_truth[:, :3],
            dim=1,
        )


class PixelLoss(Loss):
    """
    A loss defined as the elementwise squared error between each pixel of predicted bitmaps and the ground truth.

    Attributes
    ----------
    loss_function : torch.nn.Module
        A torch module implementing a loss.
    scenario : Scenario
        The scenario.

    See Also
    --------
    :class:`Loss` : Reference to the parent class.
    """

    def __init__(self) -> None:
        """Initialize the pixel loss."""
        super().__init__(loss_function=torch.nn.MSELoss(reduction="none"))

    def __call__(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""
        Compute the normalized pixel-wise loss.

        To make the loss invariant to the overall magnitude of the ground truth flux, the summed loss is divided
        by the total ground truth intensity per sample.

        Parameters
        ----------
        prediction : torch.Tensor
            The predicted values.
            Shape is ``[number_of_samples, bitmap_resolution_e, bitmap_resolution_u]``.
        ground_truth : torch.Tensor
            The ground truth.
            Shape is ``[number_of_samples, bitmap_resolution_e, bitmap_resolution_u]``.
        \*\*kwargs : Any
            Keyword arguments.
            ``reduction_dimensions`` is an expected keyword arguments for the pixel loss.

        Raises
        ------
        ValueError
            If expected keyword arguments are not passed.

        Returns
        -------
        torch.Tensor
            The summed MSE pixel loss reduced along the specified dimensions.
            Shape is ``[number_of_samples]``.
        """
        expected_kwargs = ["reduction_dimensions"]
        errors = []
        for key in expected_kwargs:
            if key not in kwargs:
                errors.append(f"Please add {key} as keyword argument.")
        if errors:
            raise ValueError(
                f"The vector loss expects {expected_kwargs} as keyword arguments. "
                + " ".join(errors)
            )

        return self.loss_function(prediction, ground_truth).sum(
            dim=kwargs["reduction_dimensions"]
        ) / ground_truth.sum(dim=(1, 2))


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
        equal to zero. The KL-divergence is defined by:

        .. math::

            D_{\mathrm{KL}}(P \parallel Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)},

        where :math:`P` is the ground truth distribution and :math:`Q` is the approximation or prediction
        of :math:`Q`. The KL-divergence is an asymmetric function. Switching :math:`P` and :math:`Q`
        has the following effect:
        :math:`P \parallel Q` Penalizes extra mass in the prediction where the ground truth has none.
        :math:`Q \parallel P` Penalizes missing mass in the prediction where the ground truth has mass.

        Parameters
        ----------
        prediction : torch.Tensor
            The predicted values.
            Shape is ``[number_of_samples, bitmap_resolution_e, bitmap_resolution_u]``.
        ground_truth : torch.Tensor
            The ground truth.
            Shape is ``[number_of_samples, bitmap_resolution_e, bitmap_resolution_u]``.
        \*\*kwargs : Any
            Keyword arguments.
            ``reduction_dimensions`` is an expected keyword argument for the KL-divergence loss.

        Raises
        ------
        ValueError
            If expected keyword arguments are not passed.

        Returns
        -------
        torch.Tensor
            The summed KL-divergence loss reduced along the specified dimensions.
            Shape is ``[number_of_samples]``.
        """
        expected_kwargs = ["reduction_dimensions"]
        for key in expected_kwargs:
            if key not in kwargs:
                raise ValueError(
                    f"The KL-divergence loss expects {key} as keyword argument. Please add this argument."
                )

        eps = 1e-12
        ground_truth_distributions = torch.nn.functional.normalize(
            ground_truth,
            p=1,
            dim=(indices.batched_bitmap_e, indices.batched_bitmap_u),
            eps=eps,
        )
        predicted_distributions = torch.nn.functional.normalize(
            prediction,
            p=1,
            dim=(indices.batched_bitmap_e, indices.batched_bitmap_u),
            eps=eps,
        )

        loss = self.loss_function(
            torch.log(predicted_distributions + eps),
            torch.log(ground_truth_distributions + eps),
        )

        return loss.sum(dim=kwargs["reduction_dimensions"])


class AngleLoss(Loss):
    """
    A loss defined as the angular difference between the prediction and ground truth.

    Attributes
    ----------
    loss_function : torch.nn.Module
        A torch module implementing a loss.

    See Also
    --------
    :class:`Loss` : Reference to the parent class.
    """

    def __init__(self) -> None:
        """Initialize the angle loss."""
        super().__init__(loss_function=None)

    def __call__(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""
        Compute the angular distance between prediction and ground truth.

        Parameters
        ----------
        prediction : torch.Tensor
            The predicted values.
            Shape is ``[number_of_samples, 4]``.
        ground_truth : torch.Tensor
            The ground truth.
            Shape is ``[number_of_samples, 4]``.
        \*\*kwargs : Any
            Keyword arguments.

        Returns
        -------
        torch.Tensor
            The summed loss reduced along the specified dimensions.
            Shape is ``[number_of_samples]``.
        """
        prediction = torch.nn.functional.normalize(prediction[:, :3])
        ground_truth = torch.nn.functional.normalize(ground_truth[:, :3])
        return torch.acos((prediction * ground_truth).sum(dim=-1).clamp(-1.0, 1.0))


class CosineSimilarityLoss(Loss):
    """
    A loss defined as the cosine similarity between prediction and ground truth.

    Attributes
    ----------
    loss_function : torch.nn.Module
        A torch module implementing a loss.

    See Also
    --------
    :class:`Loss` : Reference to the parent class.
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
            Shape is ``[number_of_samples, 4]``.
        ground_truth : torch.Tensor
            The ground truth.
            Shape is ``[number_of_samples, 4]``.
        \*\*kwargs : Any
            Keyword arguments.

        Returns
        -------
        torch.Tensor
            The summed loss reduced along the specified dimensions.
            Shape is ``[number_of_samples]``.
        """
        return 1.0 - self.loss_function(prediction, ground_truth)


def reduce_loss_per_sample(
    loss_per_sample: torch.Tensor,
    number_of_samples_per_heliostat: int,
    reduction: Callable[..., Any],
) -> torch.Tensor:
    """
    Calculate the loss per heliostat from a loss per sample.

    The reduction operation is chosen via the `reduction` parameter.

    Parameters
    ----------
    loss_per_sample : torch.Tensor
        Loss per sample.
        Tensor of shape [number_of_samples].
    number_of_samples_per_heliostat : int
        Number of samples per heliostat.
    reduction : Callable[..., Any]
        Reduction function applied across the sample dimension for each heliostat.

    Returns
    -------
    torch.Tensor
        Loss per heliostat.
        Tensor of shape [number_of_heliostats].
    """
    number_of_heliostats = int(
        loss_per_sample.numel() // number_of_samples_per_heliostat
    )
    loss_per_sample = loss_per_sample[
        : number_of_heliostats * number_of_samples_per_heliostat
    ]

    reduced = reduction(
        loss_per_sample.view(number_of_heliostats, number_of_samples_per_heliostat)
    )

    if not isinstance(reduced, torch.Tensor):
        reduced = reduced.values

    return reduced
