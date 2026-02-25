from typing import Any

import torch


class Regularizer:
    """
    Abstract base class for all regularizers.

    Attributes
    ----------
    reduction_dimensions : tuple[int, ...]
        The dimensions along which to reduce the regularization term.
    """

    def __init__(self, reduction_dimensions: tuple[int, ...]) -> None:
        """
        Initialize the base regularizer.

        Parameters
        ----------
        reduction_dimensions : tuple[int, ...]
            The dimensions along which to reduce the regularization term.
        """
        self.reduction_dimensions = reduction_dimensions

    def __call__(
        self,
        current_control_points: torch.Tensor,
        original_control_points: torch.Tensor,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""
        Compute the regularization.

        Parameters
        ----------
        current_control_points : torch.Tensor
            The current control points.
            Tensor of shape [number_of_heliostats, number_of_facets_per_surface, number_of_control_points_u_direction, number_of_control_points_v_direction, 3].
        original_control_points : torch.Tensor
            The current control points.
            Tensor of shape [number_of_heliostats, number_of_facets_per_surface, number_of_control_points_u_direction, number_of_control_points_v_direction, 3].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        \*\*kwargs : Any
            Keyword arguments.

        Raises
        ------
        NotImplementedError
            This abstract method must be overridden.
        """
        raise NotImplementedError("Must be overridden!")


class SmoothnessRegularizer(Regularizer):
    """
    Penalize localized control-point variations to enforce smooth deformations.

    The regularization is applied to the displacement of control points relative to the original
    surface control points using a discrete second-order Laplacian operator.

    See Also
    --------
    :class:`Regularizer` : Reference to the parent class.
    """

    def __init__(self, reduction_dimensions: tuple[int, ...]) -> None:
        """
        Initialize the regularizer.

        Parameters
        ----------
        reduction_dimensions : tuple[int, ...]
            Dimensions along which to reduce the loss.
        """
        super().__init__(reduction_dimensions)

    def __call__(
        self,
        current_control_points: torch.Tensor,
        original_control_points: torch.Tensor,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute the Laplacian regularization loss.

        The loss measures how much each control-point displacement differs from the average of its four immediate
        neighbors, thereby penalizing localized, non-smooth deformations.

        Parameters
        ----------
        current_control_points : torch.Tensor
            The current control points.
            Tensor of shape [number_of_heliostats, number_of_facets_per_surface, number_of_control_points_u_direction, number_of_control_points_v_direction, 3].
        original_control_points : torch.Tensor
            The current control points.
            Tensor of shape [number_of_heliostats, number_of_facets_per_surface, number_of_control_points_u_direction, number_of_control_points_v_direction, 3].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            Laplacian regularization loss per surface.
        """
        control_points_delta = current_control_points - original_control_points

        # Pad to handle edges with replication.
        delta_padded = torch.nn.functional.pad(
            control_points_delta, (0, 0, 1, 1, 1, 1), mode="replicate"
        )

        # Discrete Laplacian of all neighbors (up, down, left, right).
        laplace = (
            4 * control_points_delta
            - delta_padded[:, :, :-2, 1:-1, :]
            - delta_padded[:, :, 2:, 1:-1, :]
            - delta_padded[:, :, 1:-1, :-2, :]
            - delta_padded[:, :, 1:-1, 2:, :]
        )

        laplacian_loss = (laplace**2).mean(dim=(2, 3, 4))
        laplacian_loss = laplacian_loss.sum(dim=self.reduction_dimensions)

        return laplacian_loss


class IdealSurfaceRegularizer(Regularizer):
    """
    Penalizes deviations of control points from the original control points.

    See Also
    --------
    :class:`Regularizer` : Reference to the parent class.
    """

    def __init__(self, reduction_dimensions: tuple[int, ...]) -> None:
        """
        Initialize the regularizer.

        Parameters
        ----------
        reduction_dimensions : tuple[int, ...]
            Dimensions along which to reduce the loss.
        """
        super().__init__(reduction_dimensions)

    def __call__(
        self,
        current_control_points: torch.Tensor,
        original_control_points: torch.Tensor,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute the L2 loss between current control points and original control points.

        Parameters
        ----------
        current_control_points : torch.Tensor
            The current control points.
            Tensor of shape [number_of_heliostats, number_of_facets_per_surface, number_of_control_points_u_direction, number_of_control_points_v_direction, 3].
        original_control_points : torch.Tensor
            The current control points.
            Tensor of shape [number_of_heliostats, number_of_facets_per_surface, number_of_control_points_u_direction, number_of_control_points_v_direction, 3].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            L2 deviation loss per surface.
        """
        delta = current_control_points - original_control_points
        delta_squared = delta**2

        per_facet_loss = delta_squared.mean(dim=(2, 3, 4))
        loss = per_facet_loss.sum(dim=self.reduction_dimensions)

        return loss
