from typing import List, Tuple

from artist.physics_objects.module import AModule

import torch
from artist.physics_objects.heliostats.surface.facets.facets import AFacetModule

from artist.physics_objects.module import AModule

class ConcentratorModule(AModule):
    def __init__(self,
                 facets: List[AFacetModule]
    ):
        super().__init__()
        self.facets = facets


    def get_surface(self) -> Tuple[torch.Tensor, torch.Tensor]:
        surface_points = torch.empty(0, 3)
        surface_normals = torch.empty(0, 3)
        
        for facet in self.facets:
            surface_points = torch.cat((surface_points, facet[0]), 0)
            surface_normals = torch.cat((surface_normals, facet[1]), 0)

        return surface_points, surface_normals