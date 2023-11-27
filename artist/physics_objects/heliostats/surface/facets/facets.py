import itertools
from typing import List, Optional, Union
import torch

from artist.physics_objects.heliostats.surface.nurbs.canting import CantingAlgorithm


class AFacetModule(torch.nn.Module):
    # Relative to heliostat position.
    positions: torch.Tensor
    spans_n: torch.Tensor
    spans_e: torch.Tensor

    _discrete_points: List[torch.Tensor]
    _discrete_points_ideal: List[torch.Tensor]
    _normals: List[torch.Tensor]
    _normals_ideal: List[torch.Tensor]

    offsets: torch.Tensor

    _canting_algo: Optional[CantingAlgorithm]
    cant_rots: torch.Tensor

    @property
    def raw_discrete_points(self) -> List[torch.Tensor]:
        return self._discrete_points

    @raw_discrete_points.setter
    def raw_discrete_points(
            self,
            new_discrete_points: List[torch.Tensor],
    ) -> None:
        self._discrete_points = new_discrete_points

    @property
    def raw_discrete_points_ideal(self) -> List[torch.Tensor]:
        return self._discrete_points_ideal

    @raw_discrete_points_ideal.setter
    def raw_discrete_points_ideal(
            self,
            new_discrete_points_ideal: List[torch.Tensor],
    ) -> None:
        self._discrete_points_ideal = new_discrete_points_ideal

    @property
    def raw_normals(self) -> List[torch.Tensor]:
        return self._normals

    @raw_normals.setter
    def raw_normals(self, new_normals: List[torch.Tensor]) -> None:
        self._normals = new_normals

    @property
    def raw_normals_ideal(self) -> List[torch.Tensor]:
        return self._normals_ideal

    @raw_normals_ideal.setter
    def raw_normals_ideal(self, new_normals_ideal: List[torch.Tensor]) -> None:
        self._normals_ideal = new_normals_ideal