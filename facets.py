import itertools
from typing import List, Optional, Tuple, TYPE_CHECKING, Type, TypeVar, Union

import torch
import torch as th
from yacs.config import CfgNode

import canting
from canting import CantingAlgorithm
if TYPE_CHECKING:
    from heliostat_models import AbstractHeliostat
import utils

C = TypeVar('C', bound='Facets')


def _broadcast_spans(
        spans: List[List[float]],
        to_length: int,
) -> List[List[float]]:
    if len(spans) == to_length:
        return spans

    assert len(spans) == 1, (
        'will only broadcast spans of length 1. If you did not intend '
        'to broadcast, make sure there is the same amount of facet '
        'positions and spans.'
    )
    return spans * to_length


def get_facet_params(
        cfg: CfgNode,
        dtype: th.dtype,
        device: th.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions: List[List[float]] = utils.with_outer_list(cfg.FACETS.POSITIONS)
    spans_n: List[List[float]] = utils.with_outer_list(cfg.FACETS.SPANS_N)
    spans_n = _broadcast_spans(spans_n, len(positions))
    spans_e: List[List[float]] = utils.with_outer_list(cfg.FACETS.SPANS_E)
    spans_e = _broadcast_spans(spans_e, len(positions))
    position, spans_n, spans_e = map(
        lambda l: th.tensor(l, dtype=dtype, device=device),
        [positions, spans_n, spans_e],
    )
    return position, spans_n, spans_e


def sole_facet(
        height: float,
        width: float,
        dtype: th.dtype,
        device: th.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        th.zeros((1, 3), dtype=dtype, device=device),
        th.tensor([[0, height / 2, 0]], dtype=dtype, device=device),
        th.tensor([[-width / 2, 0, 0]], dtype=dtype, device=device),
    )


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


def cant_and_merge_facet_vectors(
        facets: 'AbstractFacets',
        facetted_vectors: List[torch.Tensor],
        reposition: bool,
        force_canting: bool = False,
) -> torch.Tensor:
    # We prefer doing canting and merging at once instead of using
    # `th.cat` so that we use less memory.
    total_size = len(facets)
    merged_vectors = th.empty(
        (total_size, 3),
        dtype=facetted_vectors[0].dtype,
        device=facetted_vectors[0].device,
    )
    do_canting = (
        not canting.is_like_active(facets._canting_algo)
        or force_canting
    )

    i = 0
    for (
            facet_position,
            cant_rot,
            facet_vectors,
    ) in zip(
            facets.positions,
            facets.cant_rots,
            facetted_vectors,
    ):
        offset = len(facet_vectors)

        if do_canting:
            # We expect the position to be centered on zero for
            # canting, so cant before repositioning.
            # (If we handled discrete points and normals at once, we
            # could also concat and after rotation de-construct here for
            # possibly more speed.)
            facet_vectors = canting.apply_rotation(cant_rot, facet_vectors)

        if reposition:
            facet_vectors = facet_vectors + facet_position

        merged_vectors[i:i + offset] = facet_vectors
        i += offset

    return merged_vectors


def make_unfacetted(values: List[torch.Tensor]) -> torch.Tensor:
    return th.cat(values, dim=0)


def make_facetted(
        values: torch.Tensor,
        facet_offsets: torch.Tensor,
) -> List[torch.Tensor]:
    return list(th.tensor_split(values, facet_offsets[1:].cpu()))


class AbstractFacets:
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

    def __len__(self) -> int:
        return sum(
            len(facet_discrete_points)
            for facet_discrete_points in self._discrete_points
        )

    def _make_offsets(
            self,
            discrete_points: Union[
                List[torch.Tensor],
                List['AbstractHeliostat'],
            ],
    ) -> torch.Tensor:
        return torch.tensor(
            list(itertools.accumulate(
                (
                    [0]
                    + [
                        len(facet_discrete_points)
                        for facet_discrete_points in discrete_points[:-1]
                    ]
                ),
            )),
            dtype=th.long,
            device=self.positions.device,
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
            reposition: bool = True,
            force_canting: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError('please override `align_discrete_points`')

    def align_discrete_points_ideal(
            self,
            reposition: bool = True,
            force_canting: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError(
            'please override `align_discrete_points_ideal`')

    def align_normals(self, force_canting: bool = False) -> torch.Tensor:
        raise NotImplementedError('please override `align_normals`')

    def align_normals_ideal(self, force_canting: bool = False) -> torch.Tensor:
        raise NotImplementedError('please override `align_normals_ideal`')


class Facets(AbstractFacets):
    def __init__(
            self,
            heliostat: 'AbstractHeliostat',
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

        self._canting_algo = heliostat.canting_algo

        self.offsets = self._make_offsets(discrete_points)

        self._discrete_points = discrete_points
        self._discrete_points_ideal = discrete_points_ideal
        self._normals = normals
        self._normals_ideal = normals_ideal
        self.cant_rots = cant_rots

    @classmethod
    def find_facets(
            cls: Type[C],
            heliostat: 'AbstractHeliostat',
            positions: torch.Tensor,
            spans_n: torch.Tensor,
            spans_e: torch.Tensor,
            discrete_points: torch.Tensor,
            discrete_points_ideal: torch.Tensor,
            normals: torch.Tensor,
            normals_ideal: torch.Tensor,
    ) -> C:
        facetted_discrete_points: List[torch.Tensor] = []
        facetted_discrete_points_ideal: List[torch.Tensor] = []
        facetted_normals: List[torch.Tensor] = []
        facetted_normals_ideal: List[torch.Tensor] = []
        cant_rots: torch.Tensor = th.empty(
            (len(positions), 3, 3),
            dtype=positions.dtype,
            device=positions.device,
        )

        canting_params = canting.get_canting_params(heliostat)

        for (i, (position, span_n, span_e)) in enumerate(zip(
                positions,
                spans_n,
                spans_e,
        )):
            # Select points on facet based on positions of ideal points.
            indices = facet_point_indices(
                discrete_points_ideal, position, span_n, span_e)
            facet_discrete_points = discrete_points[indices]
            facet_discrete_points_ideal = discrete_points_ideal[indices]
            facet_normals = normals[indices]
            facet_normals_ideal = normals_ideal[indices]

            (
                facet_discrete_points,
                facet_discrete_points_ideal,
                facet_normals,
                facet_normals_ideal,
                cant_rot,
            ) = canting.decant_facet(
                position,
                facet_discrete_points,
                facet_discrete_points_ideal,
                facet_normals,
                facet_normals_ideal,
                heliostat.cfg.IDEAL.NORMAL_VECS,
                canting_params,
            )

            # Re-center facet around zero.
            facet_discrete_points -= position
            facet_discrete_points_ideal -= position

            facetted_discrete_points.append(facet_discrete_points)
            facetted_discrete_points_ideal.append(facet_discrete_points_ideal)
            facetted_normals.append(facet_normals)
            facetted_normals_ideal.append(facet_normals_ideal)
            cant_rots[i] = cant_rot

        return cls(
            heliostat,
            positions,
            spans_n,
            spans_e,
            facetted_discrete_points,
            facetted_discrete_points_ideal,
            facetted_normals,
            facetted_normals_ideal,
            cant_rots,
        )

    def align_discrete_points(
            self,
            reposition: bool = True,
            force_canting: bool = False,
    ) -> torch.Tensor:
        return cant_and_merge_facet_vectors(
            self,
            self._discrete_points,
            reposition=reposition,
            force_canting=force_canting,
        )

    def align_discrete_points_ideal(
            self,
            reposition: bool = True,
            force_canting: bool = False
    ) -> torch.Tensor:
        return cant_and_merge_facet_vectors(
            self,
            self._discrete_points_ideal,
            reposition=reposition,
            force_canting=force_canting,
        )

    def align_normals(self, force_canting: bool = False) -> torch.Tensor:
        return cant_and_merge_facet_vectors(
            self,
            self._normals,
            reposition=False,
            force_canting=force_canting,
        )

    def align_normals_ideal(self, force_canting: bool = False) -> torch.Tensor:
        return cant_and_merge_facet_vectors(
            self,
            self._normals_ideal,
            reposition=False,
            force_canting=force_canting,
        )


class AlignedFacets(AbstractFacets):
    def __init__(
            self,
            discrete_points: torch.Tensor,
            normals: torch.Tensor,
            offsets: torch.Tensor,
    ) -> None:
        self._discrete_points = make_facetted(discrete_points, offsets)
        self._normals = make_facetted(normals, offsets)
        self.offsets = offsets

    @property
    def positions(self) -> torch.Tensor:  # type: ignore[override]
        raise ValueError('Aligned facet does not have a relative position')

    @property
    def raw_discrete_points_ideal(self) -> List[torch.Tensor]:
        raise ValueError('Aligned facet does not have ideal vectors')

    @raw_discrete_points_ideal.setter
    def raw_discrete_points_ideal(
            self,
            new_discrete_points: List[torch.Tensor],
    ) -> None:
        raise ValueError('Aligned facet does not have ideal vectors')

    @property
    def raw_normals_ideal(self) -> List[torch.Tensor]:
        raise ValueError('Aligned facet does not have ideal vectors')

    @raw_normals_ideal.setter
    def raw_normals_ideal(self, new_normals: List[torch.Tensor]) -> None:
        raise ValueError('Aligned facet does not have ideal vectors')

    @property
    def _canting_algo(  # type: ignore[override]
            self,
    ) -> Optional[CantingAlgorithm]:
        raise ValueError('Aligned facet does not support canting')

    @property
    def cant_rots(self) -> torch.Tensor:  # type: ignore[override]
        raise ValueError('Aligned facet does not support canting')

    def align_discrete_points(
            self,
            reposition: bool = True,
            force_canting: bool = False,
    ) -> torch.Tensor:
        assert reposition, 'Aligned facet cannot not be repositioned'
        assert not force_canting, 'Aligned facet cannot not be canted'
        return make_unfacetted(self._discrete_points)

    def align_discrete_points_ideal(
            self,
            reposition: bool = True,
            force_canting: bool = False,
    ) -> torch.Tensor:
        raise ValueError('Aligned facet does not have ideal vectors')

    def align_normals(self, force_canting: bool = False) -> torch.Tensor:
        assert not force_canting, 'Aligned facet cannot not be canted'
        return make_unfacetted(self._normals)

    def align_normals_ideal(self, force_canting: bool = False) -> torch.Tensor:
        raise ValueError('Aligned facet does not have ideal vectors')
