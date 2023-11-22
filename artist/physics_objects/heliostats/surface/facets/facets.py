import itertools
from typing import List, Optional, Type, TypeVar, Union
import torch

C = TypeVar('C', bound='Facets')

def _indices_between(
        points: torch.Tensor,
        from_: torch.Tensor,
        to: torch.Tensor,
) -> torch.Tensor:
    indices = (
        (from_ <= points) & (points < to)
    ).all(dim=-1)
    return indices

def facet_point_indices(
        points: torch.Tensor,
        position: torch.Tensor,
        span_n: torch.Tensor,
        span_e: torch.Tensor,
) -> torch.Tensor:
    from_xyz = position + span_e - span_n
    to_xyz = position - span_e + span_n
    # We ignore the z-axis here.
    return _indices_between(
        points[:, :-1],
        from_xyz[:-1],
        to_xyz[:-1],
    )

def merge_facet_vectors(
        facets: 'AFacets',
        facetted_vectors: List[torch.Tensor],
) -> torch.Tensor:
    # We prefer doing canting and merging at once instead of using
    # `th.cat` so that we use less memory.
    total_size = len(facets)
    merged_vectors = torch.empty(
        (total_size, 3),
        dtype=facetted_vectors[0].dtype,
        device=facetted_vectors[0].device,
    )
    return merged_vectors

class AFacets:
    # Relative to heliostat position.
    positions: torch.Tensor
    spans_n: torch.Tensor
    spans_e: torch.Tensor

    _discrete_points: List[torch.Tensor]
    _discrete_points_ideal: List[torch.Tensor]
    _normals: List[torch.Tensor]
    _normals_ideal: List[torch.Tensor]

    offsets: torch.Tensor

    def __len__(self) -> int:
        return sum(
            len(facet_discrete_points)
            for facet_discrete_points in self._discrete_points
        )

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

    def align_discrete_points(
            self,
    ) -> torch.Tensor:
        raise NotImplementedError('please override `align_discrete_points`')

    def align_discrete_points_ideal(
            self,
    ) -> torch.Tensor:
        raise NotImplementedError(
            'please override `align_discrete_points_ideal`')

    def align_normals(self) -> torch.Tensor:
        raise NotImplementedError('please override `align_normals`')

    def align_normals_ideal(self) -> torch.Tensor:
        raise NotImplementedError('please override `align_normals_ideal`')


class Facets(AFacets):
    def __init__(
            self,
            positions: torch.Tensor,
            spans_n: torch.Tensor,
            spans_e: torch.Tensor,
            discrete_points: List[torch.Tensor],
            discrete_points_ideal: List[torch.Tensor],
            normals: List[torch.Tensor],
            normals_ideal: List[torch.Tensor],
            cant_rots: torch.Tensor,
    ) -> None:
        self.positions = positions
        self.spans_n = spans_n
        self.spans_e = spans_e

        self._discrete_points = discrete_points
        self._discrete_points_ideal = discrete_points_ideal
        self._normals = normals
        self._normals_ideal = normals_ideal
        self.cant_rots = cant_rots

    def align_discrete_points(
            self,
    ) -> torch.Tensor:
        return merge_facet_vectors(
            self,
            self._discrete_points
        )

    def align_discrete_points_ideal(
            self,
    ) -> torch.Tensor:
        return merge_facet_vectors(
            self,
            self._discrete_points_ideal,
        )

    def align_normals(self) -> torch.Tensor:
        return merge_facet_vectors(
            self,
            self._normals,
        )

    def align_normals_ideal(self) -> torch.Tensor:
        return merge_facet_vectors(
            self,
            self._normals_ideal,
        )