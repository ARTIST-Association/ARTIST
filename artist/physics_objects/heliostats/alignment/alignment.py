from typing import Tuple
import torch
import pytorch3d.transforms as throt
from ...module import AModule
from .neural_network_rigid_body_fusion import NeuralNetworkRigidBodyFusion


class AlignmentModule(AModule):
    def __init__(self, position):
        super().__init__()
        self.position = position
        self.kinematicModel = NeuralNetworkRigidBodyFusion(position=position)

    def align_surface(self, datapoint, surface_points, surface_normals):
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

        """
        orientation = self.align(datapoint=datapoint)
        alignment = torch.stack(
            self.heliostat_coord_system(
                self.position,
                -datapoint.light_directions,
                datapoint.desired_aimpoint,
                (orientation @ torch.tensor([0, 0, 0, 1], dtype=torch.float32))[:1, :3],
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
        return aligned_surface_points, aligned_surface_normals

    def align(self, datapoint):
        """
        Compute orientation from aimpoint.

        Parameters
        ----------
        datapoint :

        Returns
        -------

        """
        return self.kinematicModel.compute_orientation_from_aimpoint(datapoint)

    def heliostat_coord_system(
            self,
            position: torch.Tensor,
            sun: torch.Tensor,
            aimpoint: torch.Tensor,
            ideal_normal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = position.dtype
        device = position.device
        p_sun = sun
        p_position = position
        p_aimpoint = aimpoint
        # Calculation ideal heliostat
        # Iteration 0
        z = p_aimpoint - p_position
        z = z / torch.linalg.norm(z)
        z = p_sun + z
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
        x = x / torch.linalg.norm(x)
        y = torch.cross(z, x)
        return x, y, z
