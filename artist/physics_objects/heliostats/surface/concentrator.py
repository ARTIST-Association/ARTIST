from typing import Tuple

import torch
from yacs.config import CfgNode
from artist.physics_objects.module import AModule

class ConcentratorModule(AModule):
    def __init__(self,
                 surface_config: CfgNode,
                 nurbs_config: CfgNode,
                 device: torch.device,
                 position_on_field: torch.Tensor,
                 receiver_center: torch.Tensor,
                 sun_directions: torch.Tensor):
        super().__init__()
        


    def get_surface(self) -> Tuple[torch.Tensor, torch.Tensor]:
        surface_points, surface_normals = self.multi_nurbs_surface._discrete_points()
        return surface_points, surface_normals