from typing import List, Optional, Tuple
from yacs.config import CfgNode

import torch

from artist.io.datapoint import HeliostatDataPoint
from artist.physics_objects.heliostats.surface.concentrator import ConcentratorModule
from artist.physics_objects.heliostats.alignment.alignment import AlignmentModule
from artist.physics_objects.heliostats.surface.facets.point_cloud_facets import PointCloudFacetModule
from artist.physics_objects.module import AModule


class HeliostatModule(AModule):
    """
    Implementation of the Heliostat as a module.

    Attributes
    ----------
    position : torch.Tensor
        The position of the heliostat in the field.
    aim_point : torch.Tensor
        The aim point on the receiver.
    concentrator : ConcentratorModule
        The surface of the heliostat.
    alignment : AlignmentModule
        The alignment module of the heliostat.

    Methods
    -------
    get_aligned_surface()
        Compute the aligned surface points and aligned surface normals of the heliostat.

    See also
    --------
    :class:AModule : Reference to the parent class.
    """

    def __init__(
        self, heliostat_config : CfgNode
    ) -> None:
        """
        Initialize the heliostat.

        Parameters
        ----------
        heliostat_config : CfgNode
            The config file with the heliostat data
        """
        super().__init__()
        self.aim_point = torch.tensor(heliostat_config.DEFLECT_DATA.AIM_POINT).reshape(-1, 1)
        self.position = torch.tensor(heliostat_config.DEFLECT_DATA.POSITION_ON_FIELD).reshape(-1, 1)

        self.concentrator = ConcentratorModule(heliostat_config)
        self.alignment = AlignmentModule(heliostat_position=self.position)


    def get_aligned_surface(
        self, datapoint: HeliostatDataPoint
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the aligned surface points and aligned surface normals of the heliostat.

        Parameters
        ----------
        datapoint : HeliostatDataPoint
            Datapoint containing information about the heliostat and the environment.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The aligned surface points and aligned surface normals.
        """
        surface_points, surface_normals = self.concentrator.get_surface()
        aligned_surface_points, aligned_surface_normals = self.alignment.align_surface(
            datapoint, surface_points, surface_normals
        )
        return aligned_surface_points, aligned_surface_normals