from typing import Tuple

import h5py
import torch
from yacs.config import CfgNode

from artist.physics_objects.heliostats.concentrator.facets.point_cloud_facets import (
    PointCloudFacetModule,
)
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

    def __init__(self, heliostat_name: str, config_file: h5py.File) -> None:
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

        facet_type = config_file["heliostats"]["heliostats_list"][heliostat_name][
            "parameters"
        ]["facets_type"][()].decode("utf-8")

        if facet_type == "point_cloud":
            self.facets = PointCloudFacetModule(
                heliostat_name=heliostat_name, config_file=config_file
            )
        else:
            raise NotImplementedError(
                "ARTIST is currently only implemented for a point cloud facet type"
            )

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

        return torch.vstack(surface_points), torch.vstack(surface_normals)
