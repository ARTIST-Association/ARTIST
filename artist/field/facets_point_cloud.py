"""This file contains the functionality to create a heliostat surface from a loaded point cloud."""

import torch

from artist.field.facets import Facet


class PointCloudFacet(Facet):
    """
    Implementation of the heliostat surface loaded from a point cloud.

    Attributes
    ----------
    surface_points : torch.Tensor
        The surface points.
    surface_normals : torch.Tensor
        The surface normals.

    See Also
    --------
    :class:`Facet` : Reference to the parent class.
    """

    def __init__(
        self,
        surface_points: torch.Tensor,
        surface_normals: torch.Tensor,
    ) -> None:
        """
        Initialize the surface from a point cloud.

        Parameters
        ----------
        surface_points : torch.Tensor
            The surface points.
        surface_normals : torch.Tensor
            The surface normals.
        """
        super().__init__()

        self.surface_points = surface_points
        self.surface_normals = surface_normals
