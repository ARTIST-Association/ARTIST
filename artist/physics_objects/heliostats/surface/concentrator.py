from typing import Tuple

import torch
from ...module import AModule
from .nurbs.nurbs_surface import NURBSSurface

class ConcentratorModule(AModule):
    def __init__(self):
        super().__init__()
        self.nurbs_surface = NURBSSurface()
        pass

    def get_surface() -> Tuple[torch.Tensor, torch.Tensor]:
        surface_points, surface_normals = nurbs_surface._calc_normals_and_surface()
        return surface_points, surface_normals