from typing import Any

import torch


class Concentrator(torch.nn.Module):
    """
    Implementation of the concentrator module.

    Attributes
    ----------
    facets : List[Facet]
        The facets of the concentrator.
    """

    def __init__(
        self,
        facets_type: Any,
        surface_points: torch.Tensor,
        surface_normals: torch.Tensor,
    ) -> None:
        """
        Initialize the concentrator.

        Parameters
        ----------
        facets_type : Any
            The facet type, for example point cloud.
        surface_points : torch.Tensor
            The surface points on the concentrator.
        surface_normals : torch.Tensor
            The corresponding normal vectors to the points.
        """
        super().__init__()
        self.facets = facets_type(
            surface_points=surface_points, surface_normals=surface_normals
        )