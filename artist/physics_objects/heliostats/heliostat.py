from typing import Tuple
from yacs.config import CfgNode

import torch

from artist.io.datapoint import HeliostatDataPoint
from artist.physics_objects.heliostats.concentrator.concentrator import ConcentratorModule
from artist.physics_objects.heliostats.alignment.alignment import AlignmentModule
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
    incident_ray_direction : torch.Tensor
        The direction of the rays.
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
        self, heliostat_config : CfgNode, incident_ray_direction : torch.Tensor
    ) -> None:
        """
        Initialize the heliostat.

        Parameters
        ----------
        heliostat_config : CfgNode
            The config file containing the heliostat data.
        """
        super().__init__()
        self.aim_point = torch.tensor(heliostat_config.DEFLECT_DATA.AIM_POINT).reshape(-1, 1)
        self.position = torch.tensor(heliostat_config.DEFLECT_DATA.POSITION_ON_FIELD).reshape(-1, 1)
        self.incident_ray_direction = incident_ray_direction

        self.concentrator = ConcentratorModule(heliostat_config)
        self.alignment = AlignmentModule(heliostat_config)


    def get_aligned_surface(
        self
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the aligned surface points and aligned surface normals of the heliostat.

        Parameters
        ----------
        aim_point : torch.Tensor
            The desired aim point.
        incident_ray_direction : torch.Tensor
            The direction of the rays.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The aligned surface points and aligned surface normals.
        """
        surface_points, surface_normals = self.concentrator.get_surface()
        aligned_surface_points, aligned_surface_normals = self.alignment.align_surface(
            self.aim_point, self.incident_ray_direction, surface_points, surface_normals
        )
        return aligned_surface_points, aligned_surface_normals