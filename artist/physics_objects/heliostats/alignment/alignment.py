"""
Alignment module for the heliostat.
"""

from typing import Any, Dict, Tuple

import h5py
import torch
from yacs.config import CfgNode

from artist.io.datapoint import HeliostatDataPoint
from artist.physics_objects.heliostats.alignment.kinematic.rigid_body import (
    RigidBodyModule,
)
from artist.physics_objects.module import AModule
from artist.util import artist_type_mapping_dict, config_dictionary


class AlignmentModule(AModule):
    """
    This class implements the alignment module for the heliostat.

    Attributes
    ----------
    kinematic_model : RigidBodyModule
        The kinematic model used.

    Methods
    -------
    align_surface()
        Align given surface points and surface normals according to a given orientation.
    align()
        Compute the orientation from a given aimpoint.
    heliostat_coord_system()
        Construct the heliostat coordinate system.

    See Also
    --------
    :class: AModule : The parent class.
    """

    def __init__(
        self,
        alignment_type: str,
        actuator_type: str,
        position: torch.tensor,
        aim_point: torch.tensor,
    ) -> None:
        """
        Initialize the alignment module.

        Parameters
        ----------
        position : torch.Tensor
            Position of the heliostat for which the alignment model is created.
        """
        super().__init__()
        self.kinematic_model = artist_type_mapping_dict.alignment_type_mapping.get(
            alignment_type
        )(
            actuator_type=actuator_type,
            position=position,
            aim_point=aim_point,
        )

    def align_surface(
        self,
        incident_ray_direction: torch.Tensor,
        surface_points: torch.Tensor,
        surface_normals: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align given surface points and surface normals according to a given orientation.

        Parameters
        ----------
        aim_point : torch.Tensor
            The desired aim point.
        incident_ray_direction : torch.Tensor
            The direction of the rays.
        surface_points : torch.Tensor
            Points on the surface of the heliostat that reflect the light.
        surface_normals : torch.Tensor
            Normals to the surface points.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple containing the aligned surface points and normals.
        """
        orientation = self.align(incident_ray_direction)

        aligned_surface_points = (surface_points.T @ orientation).squeeze(0)
        aligned_surface_normals = (surface_normals.T @ orientation).squeeze(0)

        # aligned_surface_points += self.position
        aligned_surface_normals /= torch.linalg.norm(
            aligned_surface_normals, dim=-1
        ).unsqueeze(-1)
        return (
            aligned_surface_points.squeeze().T.contiguous(),
            aligned_surface_normals.squeeze().T.contiguous(),
        )

    def align(self, incident_ray_direction: torch.Tensor) -> torch.Tensor:
        """
        Compute the orientation from a given aimpoint.

        Parameters
        ----------
        aim_point : torch.Tensor
            The desired aim point.
        incident_ray_direction : torch.Tensor
            The direction of the rays.

        Returns
        -------
        torch.Tensor
            The orientation matrix.
        """
        return self.kinematic_model.compute_orientation_from_aimpoint(
            incident_ray_direction
        )
