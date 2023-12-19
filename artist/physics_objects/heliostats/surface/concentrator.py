from typing import List, Tuple

import torch
from artist.physics_objects.heliostats.surface.facets.facets import AFacetModule

from artist.physics_objects.module import AModule


class ConcentratorModule(AModule):
    """
    Implementation of the concentrator module.

    Attributes
    ----------
    facets : List[AFacetModule]
        The facets of the concentrator.

    Methods
    -------
    get_surface()
        Compute the surface points and surface normals of the concentrator.

    See also
    --------
    :class:AModule : The parent class.
    """

    def __init__(self, facets: List[AFacetModule]) -> None:
        """
        Initialize the concentrator.

        Parameters
        ----------
        facets: List[AFacetModule]
            The facets of the concentrator.
        """
        super().__init__()
        self.facets = facets

    def get_surface(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the surface points and surface normals of the concentrator.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Return the surface points and the surface normals.
        """
        surface_points_list = []
        surface_normals_list = []

        for facet in self.facets:
            surface_points_list.append(facet[0])
            surface_normals_list.append(facet[1])

        surface_points = torch.cat(surface_points_list, 0)
        surface_normals = torch.cat(surface_normals_list, 0)

        return surface_points, surface_normals
