from typing import Tuple
import torch
import pytorch3d.transforms as throt
from .module import AModule
from .NeuralNetworkRigidBodyFusionModule import NeuralNetworkRigidBodyFusion

class AlignmentModule(AModule):
    def __init__(self, position):
        super(AlignmentModule, self).__init__()
        self.position = position
        self.kinematicModel = NeuralNetworkRigidBodyFusion(position=position)

    def align_surface(self, Datapoint, surface_points, surface_normals):
        """
        Align given surface points and surface normals according to a given orientation.

        Keyword arguments:
        Datapoint -- contains information about the Heliostat and the environment (lightsource, receiver,...)
        surface_points -- points on the surface of the heliostat that reflect the light
        surface_normals -- normals to the surface points
        """
        orientation = self.align(Datapoint=Datapoint)
        alignment = torch.stack(self.heliostat_coord_system(
                self.position,
                -Datapoint.light_directions,
                Datapoint.desired_aimpoint,
                (orientation @ torch.tensor([0, 0, 0, 1], dtype=torch.float32))[:1,:3])
            )
        align_origin = throt.Rotate(alignment, dtype=alignment.dtype)
        aligned_surface_points = align_origin.transform_points(surface_points.to(torch.float))
        aligned_surface_normals = align_origin.transform_normals(surface_normals.to(torch.float))
        aligned_surface_points += self.position
        aligned_surface_normals /= torch.linalg.norm(aligned_surface_normals, dim=-1).unsqueeze(-1)
        return aligned_surface_points, aligned_surface_normals

    def align(self, Datapoint):
        """Compute orientation from aimpoint."""
        return self.kinematicModel.computeOrientationFromAimpoint(Datapoint)
    
    def heliostat_coord_system(self,
        Position: torch.Tensor,
        Sun: torch.Tensor,
        Aimpoint: torch.Tensor,
        ideal_normal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = Position.dtype
        device = Position.device
        pSun = Sun
        pPosition = Position
        pAimpoint = Aimpoint
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
        return x, y, z