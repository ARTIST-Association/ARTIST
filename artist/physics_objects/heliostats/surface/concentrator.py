
from typing import Tuple

from artist.physics_objects.module import AModule

import torch

from artist.physics_objects.module import AModule
from artist.physics_objects.heliostats.surface.nurbs.nurbs_surface import NURBSSurface

class ConcentratorModule(AModule):
    def __init__(self):
        super().__init__()
        self.nurbs_surface = NURBSSurface()
        pass

    def get_surface(self) -> Tuple[torch.Tensor, torch.Tensor]:
        surface_points, surface_normals = self.nurbs_surface._calc_normals_and_surface()
        return surface_points, surface_normals