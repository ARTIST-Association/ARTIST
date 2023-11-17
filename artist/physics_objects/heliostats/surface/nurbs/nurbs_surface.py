
from typing import Optional, Tuple

import torch


class ANURBSSurface():
    def _calc_normals_and_surface(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            'please override `_calc_normals_and_surface`')

    def discrete_points_and_normals(self) -> Tuple[torch.Tensor, torch.Tensor]:
        discrete_points, normals = self._calc_normals_and_surface()
        return discrete_points, normals

    @property
    def discrete_points(self) -> torch.Tensor:
        discrete_points, _ = self._calc_normals_and_surface()
        return discrete_points

    @property
    def normals(self) -> torch.Tensor:
        _, normals = self._calc_normals_and_surface()
        return normals
    
class NURBSSurface(ANURBSSurface):
    def __init__(self) -> None:
        super().__init__()