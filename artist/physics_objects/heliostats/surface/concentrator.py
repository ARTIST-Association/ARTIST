from typing import List, Tuple, Union
from yacs.config import CfgNode

from artist.physics_objects.heliostats.surface.facets.point_cloud_facets import PointCloudFacetModule

from artist.physics_objects.module import AModule

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
    :class:AModule : Reference to the parent class.
    """

    def __init__(self, surface_config: CfgNode) -> None:
        """
        Initialize the concentrator.

        Parameters
        ----------
        surface_config : CfgNode
            The config file containing information about the surface.
        """
        super().__init__()

        self.facets = [PointCloudFacetModule(surface_config)]

    def get_surface(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the surface points and surface normals of the concentrator.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Return the surface points and the surface normals.
        """
        surface_points = [facet.ideal_surface_points for facet in self.facets]
        surface_normals = [facet.surface_normals for facet in self.facets]

        return  torch.vstack(surface_points), torch.vstack(surface_normals)
