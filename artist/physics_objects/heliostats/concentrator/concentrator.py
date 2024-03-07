from typing import Any, Dict, Tuple

import h5py
import torch
from yacs.config import CfgNode

from artist.physics_objects.heliostats.concentrator.facets.point_cloud_facets import (
    PointCloudFacetModule,
)
from artist.physics_objects.module import AModule
from artist.util import artist_type_mapping_dict, config_dictionary 

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

    def __init__(self, 
                 parameters_dict: Dict[str, Any],
                 surface_points: torch.Tensor,
                 surface_normals: torch.Tensor
                ) -> None:
        """
        Initialize the concentrator.

        Parameters
        ----------
        heliostat_name : str
            The name of the heliostat being initialized.
        config_file : h5py.File
            An open hdf5 file containing the scenario configuration.
        """
        super().__init__()

        self.facets = artist_type_mapping_dict.facet_type_mapping.get(parameters_dict[config_dictionary.facets_type_key])(surface_points=surface_points, surface_normals=surface_normals)

        
    # def get_surface(self) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Compute the surface points and surface normals of the concentrator.
    #
    #     Returns
    #     -------
    #     Tuple[torch.Tensor, torch.Tensor]
    #         Return the surface points and the surface normals.
    #     """
    #     surface_points = [facet.surface_points for facet in self.facets]
    #     surface_normals = [facet.surface_normals for facet in self.facets]
    #
    #     return torch.vstack(surface_points), torch.vstack(surface_normals)
