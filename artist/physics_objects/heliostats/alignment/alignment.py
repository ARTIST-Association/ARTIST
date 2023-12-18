from typing import List, Tuple
import torch
import pytorch3d.transforms as throt
from artist.physics_objects.module import AModule
from artist.physics_objects.heliostats.alignment.neural_network_rigid_body_fusion import NeuralNetworkRigidBodyFusion
from artist.io.datapoint import HeliostatDataPoint
from artist.util import utils


class AlignmentModule(AModule):
    """
    This class implements the alignemnt module for the Heliostat.

    See Also
    --------
    :class: AModule : Reference to the parent class
    """

    def __init__(self, position: torch.Tensor) -> None:
        """
        Initialize the alignemnt module.

        Parameters
        ----------
        position : torch.Tensor
            Position of the heliostat for which the alignemnt model is created.
        """
        super().__init__()
        self.position = position
        self.kinematicModel = NeuralNetworkRigidBodyFusion(position=position)

    def align_surface(
            self,
            datapoint: HeliostatDataPoint,
            surface_points: torch.Tensor,
            surface_normals: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align given surface points and surface normals according to a given orientation.

        Parameters
        ----------
        datapoint :
            Contains information about the heliostat and the environment (lightsource, receiver,...).
        surface_points :
            Points on the surface of the heliostat that reflect the light.
        surface_normals :
            Normals to the surface points.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple containing the aligned surface points and normals.
        """
        orientation = self.align(datapoint=datapoint)
        normal_vec = (orientation @ torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32))[:1, :3]
        alignment = torch.stack(
            self.heliostat_coord_system(
                self.position,
                datapoint.light_directions,
                datapoint.desired_aimpoint,
                normal_vec,
                # torch.Tensor([0.0, 0.0, 0.0]),
                # torch.Tensor([0.0, 0.0, 0.0]),
            )
        )
        align_origin = throt.Rotate(alignment, dtype=alignment.dtype)

        aligned_surface_normals = align_origin.transform_normals(
            surface_normals.to(torch.float)
        )
        aligned_surface_points = align_origin.transform_points(
            surface_points.to(torch.float)
        )
        aligned_surface_points += self.position
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
        return self.kinematicModel.compute_orientation_from_aimpoint(datapoint)
    
    def heliostat_coord_system(
            self,
            Position: torch.Tensor,
            Sun: torch.Tensor,
            Aimpoint: torch.Tensor,
            ideal_normal: torch.Tensor,
            # disturbance_angles: List[torch.Tensor],
            # rotation_offset: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construct the heliostat coordination system.

        Parameters
        ----------
        position : torch.tensor
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
        #rotation_offset = torch.deg2rad(rotation_offset)
        pSun = Sun
        pPosition = Position
        pAimpoint = Aimpoint
        # print(pSun,pPosition,pAimpoint)
        # Berechnung Idealer Heliostat
        # 0. Iteration
        z = pAimpoint - pPosition
        z = z / torch.linalg.norm(z)
        z = pSun + z
        z = z / torch.linalg.norm(z)

        if (z == ideal_normal).all():
            x = torch.tensor([1, 0, 0], dtype=dtype, device=device)
        else:
            x = torch.stack([
                -z[1],
                z[0],
                torch.tensor(0, dtype=dtype, device=device),
            ])
        x = x / torch.linalg.norm(x)
        y = torch.cross(z, x)

        # # Add heliostat rotation error/disturbance.
        # x_err_rot = utils.axis_angle_rotation(x, disturbance_angles[0])
        # y_err_rot = utils.axis_angle_rotation(y, disturbance_angles[1])
        # z_err_rot = utils.axis_angle_rotation(z, disturbance_angles[2]+rotation_offset)
        # # print(Position)
        # # print(disturbance_angles)
        # full_rot = z_err_rot @ y_err_rot @ x_err_rot

        # x = full_rot @ x
        # y = full_rot @ y
        # z = full_rot @ z
        return x, y, z