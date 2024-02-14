"""
Alignment module for the heliostat.
"""

from typing import Tuple

import pytorch3d.transforms as throt
import torch

from artist.io.datapoint import HeliostatDataPoint
from artist.physics_objects.heliostats.alignment.rigid_body import (
    RigidBodyModule,
)
from artist.physics_objects.module import AModule


class AlignmentModule(AModule):
    """
    This class implements the alignment module for the heliostat.

    Attributes
    ----------
    position : torch.Tensor
        Position of the heliostat for which the alignment model is created.
    kinematic_model : NeuralNetworkRigidBodyFusion
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

    def __init__(self, position: torch.Tensor) -> None:
        """
        Initialize the alignment module.

        Parameters
        ----------
        position : torch.Tensor
            Position of the heliostat for which the alignment model is created.
        """
        super().__init__()
        self.position = position
        self.kinematic_model = RigidBodyModule(position=position)

    def align_surface(
        self,
        datapoint: HeliostatDataPoint,
        surface_points: torch.Tensor,
        surface_normals: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align given surface points and surface normals according to a given orientation.

        Parameters
        ----------
        datapoint : HeliostatDataPoint
            Contains information about the heliostat and the environment (light source, receiver,...).
        surface_points : torch.Tensor
            Points on the surface of the heliostat that reflect the light.
        surface_normals : torch.Tensor
            Normals to the surface points.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple containing the aligned surface points and normals.
        """
        orientation = self.align(datapoint=datapoint)
        normal_vec = (
            orientation @ torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
        )[:1, :3]
        alignment = torch.stack(
            self.heliostat_coord_system(
                self.position,
                datapoint.light_directions,
                datapoint.desired_aimpoint,
                normal_vec,
            )
        )
        align_origin = throt.Rotate(alignment, dtype=alignment.dtype)
        aligned_surface_points = align_origin.transform_points(
            surface_points.to(torch.float)
        )
        aligned_surface_normals = align_origin.transform_normals(
            surface_normals.to(torch.float)
        )
        aligned_surface_points += self.position
        aligned_surface_normals /= torch.linalg.norm(
            aligned_surface_normals, dim=-1
        ).unsqueeze(-1)
        aligned_surface_normals /= torch.linalg.norm(
            aligned_surface_normals, dim=-1
        ).unsqueeze(-1)
        return aligned_surface_points, aligned_surface_normals

    def align(self, datapoint: HeliostatDataPoint) -> torch.Tensor:
        """
        Compute the orientation from a given aimpoint.

        Parameters
        ----------
        datapoint : HeliostatDataPoint
            Contains information about the heliostat and the environment (lightsource, receiver,...).

        Returns
        -------
        torch.Tensor
            The orientation matrix.
        """
        return self.kinematic_model.compute_orientation_from_aimpoint(datapoint)

    def heliostat_coord_system(
        self,
        Position: torch.Tensor,
        Sun: torch.Tensor,
        Aimpoint: torch.Tensor,
        ideal_normal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construct the heliostat coordinate system.

        Parameters
        ----------
        position : torch.Tensor
            The position of the heliostat.
        sun : torch.Tensor
            The sun vector / direction.
        aimpoint : torch.Tensor
            The aimpoint.
        ideal_normal : torch.Tensor
            The ideal normal vector of the heliostat.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            The heliostat coordination system.
        """
        dtype = Position.dtype
        device = Position.device
        pSun = Sun
        pPosition = Position
        pAimpoint = Aimpoint
        z = pAimpoint - pPosition
        z = z / torch.linalg.norm(z)
        z = pSun + z
        z = z / torch.linalg.norm(z)

        if (z == ideal_normal).all():
            x = torch.tensor([1, 0, 0], dtype=dtype, device=device)
        else:
            x = torch.stack(
                [
                    -z[1],
                    z[0],
                    torch.tensor(0, dtype=dtype, device=device),
                ]
            )
        x /= torch.linalg.norm(x)
        y = torch.linalg.cross(z, x)

        return x, y, z
