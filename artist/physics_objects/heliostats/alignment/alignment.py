"""
Alignment module for the heliostat.
"""

from typing import Tuple

import h5py
import torch
from yacs.config import CfgNode

from artist.io.datapoint import HeliostatDataPoint
from artist.physics_objects.heliostats.alignment.kinematic.rigid_body import (
    RigidBodyModule,
)
from artist.physics_objects.module import AModule
from artist.util import config_dictionary 

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

    def __init__(self, heliostat_name: str, config_file: h5py.File) -> None:
        """
        Initialize the alignment module.

        Parameters
        ----------
        position : torch.Tensor
            Position of the heliostat for which the alignment model is created.
        """
        super().__init__()
        alignment_type = config_file[config_dictionary.heliostat_prefix][config_dictionary.alignment_type_key][()].decode("utf-8")

        if alignment_type == "rigid_body":
            self.kinematic_model = RigidBodyModule(
                heliostat_name=heliostat_name, config_file=config_file
            )
        else:
            raise NotImplementedError(
                "ARTIST currently only supports RigidBody Kinematic models."
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

        #aligned_surface_points += self.position
        aligned_surface_normals /= torch.linalg.norm(
            aligned_surface_normals, dim=-1
        ).unsqueeze(-1)
        return aligned_surface_points.squeeze()[:, :3].T.contiguous(), aligned_surface_normals.squeeze()[:, :3].T.contiguous()

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
