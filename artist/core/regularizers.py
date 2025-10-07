from typing import Any

import torch

from artist.util import config_dictionary
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
        current_nurbs_control_points: torch.Tensor,
        original_nurbs_control_points: torch.Tensor,
        surface_points: torch.Tensor,
        surface_normals: torch.Tensor,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""
        Compute the regularization.

        Parameters
        ----------
        current_nurbs_control_points : torch.Tensor
            The predicted nurbs control points.
            Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_control_points_u_direction, number_of_control_points_v_direction, 3].
        original_nurbs_control_points : torch.Tensor
            The original, unchanged control points.
            Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_control_points_u_direction, number_of_control_points_v_direction, 3].
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


class TotalVariationRegularizer(Regularizer):
    """
    A regularizer measuring the total variation in a surface.

    Attributes
    ----------
    weight : float
        The weight of the regularization term.
    reduction_dimensions : tuple[int]
        The dimensions along which to reduce the regularization term.
    surface : str
        Specifies which part of a surface is regularized (either the surface points or the surface normals).
    number_of_neighbors : int
        The number of nearest neighbors to consider.
    sigma : float | None
        Determines how quickly the weight falls off as the distance increases.
    batch_size : int
        Used to process smaller batches of points instead of creating full distance matrices for all points.
    epsilon : float
        A small value used to prevent divisions by zero.
    """

    def __init__(
        self,
        weight: float,
        reduction_dimensions: tuple[int, ...],
        surface: str,
        number_of_neighbors: int = 20,
        sigma: float | None = None,
        batch_size: int = 512,
        epsilon: float = 1e-8,
    ) -> None:
        """
        Initialize the total variation regularizer.

        Parameters
        ----------
        weight : float
            The weight of the regularization term.
        reduction_dimensions : tuple[int, ...]
            The dimensions along which to reduce the regularization term.
        surface : str
            Specifies which part of a surface is regularized (either the surface points or the surface normals).
        number_of_neighbors : int
            The number of nearest neighbors to consider (default is 20).
        sigma : float | None
            Determines how quickly the weight falls off as the distance increases (default is None).
        batch_size : int
            Used to process smaller batches of points instead of creating full distance matrices for all points (default is 512).
        epsilon : float
            A small value used to prevent divisions by zero (default is 1e-8).
        """
        self.weight = weight
        self.reduction_dimensions = reduction_dimensions
        self.surface = surface
        self.number_of_neighbors = number_of_neighbors
        self.sigma = sigma
        self.batch_size = batch_size
        self.epsilon = epsilon

    def __call__(
        self,
        current_nurbs_control_points: torch.Tensor,
        original_nurbs_control_points: torch.Tensor,
        surface_points: torch.Tensor,
        surface_normals: torch.Tensor,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""
        Compute the regularization.

        This regularization suppresses the noise in the surface. It measures the noise in the surface by
        taking absolute differences in the z-values of the provided points. This loss implementation
        focuses on local smoothness by applying a Gaussian distance weight and thereby letting
        closer points contribute more.

        Parameters
        ----------
        current_nurbs_control_points : torch.Tensor
            The predicted nurbs control points.
            Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_control_points_u_direction, number_of_control_points_v_direction, 3].
        original_nurbs_control_points : torch.Tensor
            The original, unchanged control points.
            Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_control_points_u_direction, number_of_control_points_v_direction, 3].
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

        Returns
        -------
        torch.Tensor
            The total variation loss for all provided surfaces.
            Tensor of shape [number_of_surfaces].
        """
        device = get_device(device=device)

        if self.surface == config_dictionary.surface_points:
            regularization_variable = surface_points
        if self.surface == config_dictionary.surface_normals:
            regularization_variable = surface_normals

        number_of_surfaces, number_of_facets, number_of_surface_points_per_facet, _ = (
            regularization_variable.shape
        )
        coordinates = regularization_variable[:, :, :, :2]
        z_values = regularization_variable[:, :, :, 2]

        if self.sigma is None:
            coordinates_std = coordinates.std(dim=1).mean().item()
            sigma = max(coordinates_std * 0.1, 1e-6)
        else:
            sigma = float(self.sigma + 1e-12)

        variation_loss_sum = torch.zeros(
            (number_of_surfaces, number_of_facets), device=device
        )
        number_of_valid_neighbors = torch.zeros(
            (number_of_surfaces, number_of_facets), device=device
        )

        # Iterate over query points in batches to limit memory usage.
        for start_index in range(
            0, number_of_surface_points_per_facet, self.batch_size
        ):
            end_index = min(
                start_index + self.batch_size, number_of_surface_points_per_facet
            )
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

            # Select the k-nearest neighbors (or fewer if the coordinate is near an edge).
            number_of_neighbors_to_select = min(
                self.number_of_neighbors, number_of_surface_points_per_facet - 1
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
            z_value_variations = z_value_variations * valid_mask.type_as(
                z_value_variations
            )

            # Accumulate weighted z_value_variations.
            weights = torch.exp(-0.5 * (selected_distances / sigma) ** 2)
            weights = weights * valid_mask.type_as(weights)
            variation_loss_sum = variation_loss_sum + (
                weights * z_value_variations
            ).sum(dim=(2, 3))
            number_of_valid_neighbors = number_of_valid_neighbors + valid_mask.type_as(
                z_value_variations
            ).sum(dim=(2, 3))

        # Batched total variation losses.
        variation_loss = variation_loss_sum / (number_of_valid_neighbors + self.epsilon)

        return variation_loss.sum(dim=self.reduction_dimensions)


class IdealSurfaceRegularizer(Regularizer):
    """
    A regularizer measuring the difference between a predicted surface and the ideal.

    Attributes
    ----------
    weight : float
        The weight of the regularization term.
    reduction_dimensions : tuple[int]
        The dimensions along which to reduce the regularization term.
    """

    def __init__(self, weight: float, reduction_dimensions: tuple[int, ...]) -> None:
        """
        Initialize the ideal surface regularizer.

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
        current_nurbs_control_points: torch.Tensor,
        original_nurbs_control_points: torch.Tensor,
        surface_points: torch.Tensor,
        surface_normals: torch.Tensor,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""
        Compute the regularization.

        This regularization suppresses large changes in the control points positions. The real
        surface is expected to be close to the ideal surface, therefore large changes are penalized.

        Parameters
        ----------
        current_nurbs_control_points : torch.Tensor
            The predicted nurbs control points.
            Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_control_points_u_direction, number_of_control_points_v_direction, 3].
        original_nurbs_control_points : torch.Tensor
            The original, unchanged control points.
            Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_control_points_u_direction, number_of_control_points_v_direction, 3].
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

        Returns
        -------
        torch.Tensor
            The differences from the predicted surfaces to the ideal surfaces.
            Tensor of shape [number_of_surfaces].
        """
        loss_function = torch.nn.MSELoss(reduction="none")

        loss = loss_function(
            current_nurbs_control_points, original_nurbs_control_points
        )

        return loss.sum(dim=self.reduction_dimensions)
