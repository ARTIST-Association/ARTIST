from typing import Any

import torch

from artist.util import index_mapping
from artist.util.environment_setup import get_device


class Regularizer:
    """
    Abstract base class for all regularizers.

    Attributes
    ----------
    weight : float
        The weight of the regularization term.
    reduction_dimensions : tuple[int, ...]
        The dimensions along which to reduce the regularization term.
    """

    def __init__(self, weight: float, reduction_dimensions: tuple[int, ...]) -> None:
        """
        Initialize the base regularizer.

        Parameters
        ----------
        weight : float
            The weight of the regularization term.
        reduction_dimensions : tuple[int, ...]
            The dimensions along which to reduce the regularization term.
        """
        self.weight = weight
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
        original_surface_points : torch.Tensor
            The original surface points.
            Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_surface_points, 4].
        surface_points : torch.Tensor
            The surface points of the predicted surface.
            Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_surface_points, 4].
        surface_normals : torch.Tensor
            The surface normals of the predicted surface.
            Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_surface_normals, 4].
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
    A regularizer that penalizes high curvature in NURBS surfaces
    by applying a discrete Laplacian on control points.

    Encourages smooth, low-frequency deformations.
    """

    def __init__(self, weight: float, reduction_dimensions: tuple[int, ...] = (0, 1, 2)):
        super().__init__(weight, reduction_dimensions)

    def __call__(
        self,
        current_control_points: torch.Tensor,
        original_control_points: torch.Tensor,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute the Laplacian regularization loss.

        Parameters
        ----------
        surface_points : torch.Tensor
            The NURBS control points or evaluated surface points.
            Tensor of shape [number_of_surfaces, number_of_facets, number_of_surface_points, 4].
            Only the first 3 coordinates (x,y,z) are used.
        device : torch.device | None
            Torch device for computations.

        Returns
        -------
        torch.Tensor
            Laplacian regularization loss per surface, shape [number_of_surfaces].
        """
        delta = current_control_points - original_control_points

        # Pad delta to handle edges with replication
        delta_padded = torch.nn.functional.pad(delta, (0, 0, 1, 1, 1, 1), mode='replicate')

        # Compute 2D discrete Laplacian: L = 4*delta - (up + down + left + right)
        L = (
            4 * delta
            - delta_padded[:, :, :-2, 1:-1, :]  # up
            - delta_padded[:, :, 2:, 1:-1, :]   # down
            - delta_padded[:, :, 1:-1, :-2, :]  # left
            - delta_padded[:, :, 1:-1, 2:, :]   # right
        )

        laplacian_loss = (L ** 2).mean(dim=(2, 3, 4))
        laplacian_loss = laplacian_loss.sum(dim=self.reduction_dimensions)

        laplacian_loss = self.weight * laplacian_loss

        return laplacian_loss


class IdealSurfaceRegularizer(Regularizer):
    """
    A regularizer that penalizes the deviation of NURBS control points
    from the ideal surface. Encourages the optimized surface to stay
    close to the reference geometry.
    """

    def __init__(self, weight: float, reduction_dimensions: tuple[int, ...] = (0, 1)):
        """
        Initialize the control-point deviation regularizer.

        Parameters
        ----------
        weight : float
            Weight of the regularization term.
        reduction_dimensions : tuple[int, ...]
            Dimensions along which to reduce the loss (default: sum over surfaces and facets).
        """
        super().__init__(weight, reduction_dimensions)

    def __call__(
        self,
        current_control_points: torch.Tensor,
        original_control_points: torch.Tensor,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute the L2 loss between current control points and ideal control points.

        Parameters
        ----------
        original_surface_points : torch.Tensor
            The reference (ideal) control points.
            Shape: [S, F, U, V, 3]
        surface_points : torch.Tensor
            The current control points being optimized.
            Shape: [S, F, U, V, 3]
        device : torch.device | None
            Device for computations.

        Returns
        -------
        torch.Tensor
            L2 deviation loss per surface.
        """
        delta = current_control_points - original_control_points
        delta_squared = delta ** 2

        # Regularization per facet.
        per_facet_loss = delta_squared.mean(dim=(2, 3, 4))
        loss = per_facet_loss.sum(dim=self.reduction_dimensions)

        loss = self.weight * loss
        return loss