import functools
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import torch
import torch as th
from yacs.config import CfgNode

import canting
from canting import CantingAlgorithm
import heliostat_models
from heliostat_models import AlignedHeliostat, Heliostat
from nurbs_heliostat import (
    AbstractNURBSHeliostat,
    AlignedNURBSHeliostat,
    NURBSHeliostat,
)
import utils

C = TypeVar('C', bound='MultiNURBSHeliostat')


class MultiNURBSHeliostat(AbstractNURBSHeliostat, Heliostat):
    def __init__(
            self,
            heliostat_config: CfgNode,
            nurbs_config: CfgNode,
            device: th.device,
            setup_params: bool = True,
            receiver_center: Union[torch.Tensor, List[float], None] = None,
    ) -> None:
        super().__init__(
            heliostat_config,
            device,
            setup_params=False,
            receiver_center=receiver_center,
        )
        self.nurbs_cfg = nurbs_config
        if not self.nurbs_cfg.is_frozen():
            self.nurbs_cfg = self.nurbs_cfg.clone()
            self.nurbs_cfg.freeze()
        self._canting_cfg = self.nurbs_cfg.FACETS.CANTING

        facets_and_rots = self._create_facets(
            self.cfg, self.nurbs_cfg, setup_params=setup_params)
        self.facets = [tup[0] for tup in facets_and_rots]
        self.cant_rots = [tup[1] for tup in facets_and_rots]

    def _set_facet_points(
            self,
            facet: NURBSHeliostat,
            facet_index: int,
            position: torch.Tensor,
    ) -> torch.Tensor:
        facet_discrete_points = \
            self._facetted_discrete_points[facet_index]
        facet_discrete_points_ideal = \
            self._facetted_discrete_points_ideal[facet_index]
        facet_normals = self._facetted_normals[facet_index]
        facet_normals_ideal = self._facetted_normals_ideal[facet_index]

        orig_normal = facet_normals_ideal.mean(dim=0)
        orig_normal /= th.linalg.norm(orig_normal)

        target_normal = th.tensor(
            self.cfg.IDEAL.NORMAL_VECS,
            dtype=position.dtype,
            device=self.device,
        )

        # De-cant so the facet is flat on z = 0.
        (
            facet_discrete_points,
            facet_discrete_points_ideal,
            facet_normals,
            facet_normals_ideal,
        ) = canting.cant_facet_to_normal(
            position,
            orig_normal,
            target_normal,
            facet_discrete_points,
            facet_discrete_points_ideal,
            facet_normals,
            facet_normals_ideal,
        )

        # Re-center facet around zero.
        facet_discrete_points -= position
        facet_discrete_points_ideal -= position

        decanted_normal = facet_normals_ideal.mean(dim=0)
        decanted_normal /= th.linalg.norm(decanted_normal)

        cant_rot = utils.get_rot_matrix(decanted_normal, orig_normal)

        facet._discrete_points = facet_discrete_points
        facet._discrete_points_ideal = facet_discrete_points_ideal
        facet._orig_world_points = facet._discrete_points_ideal.clone()
        facet._normals = facet_normals
        facet._normals_ideal = facet_normals_ideal

        added_dims = facet.height + facet.width
        height_ratio = facet.height / added_dims
        width_ratio = facet.width / added_dims

        # FIXME Not perfectly accurate.
        # Only way to really do this is by comparing values for
        # equality over an axis.
        if facet.h_rows is not None:
            facet.h_rows = int(th.pow(
                th.tensor(len(facet._discrete_points)),
                height_ratio,
            ))
        if facet.h_cols is not None:
            facet.h_cols = int(th.ceil(th.pow(
                th.tensor(len(facet._discrete_points)),
                width_ratio,
            )))

        # Handle non-rectangular facet points.
        if (
                facet.h_rows is not None and facet.h_cols is not None
                and facet.h_rows * facet.h_cols != len(facet._discrete_points)
        ):
            facet.h_rows = None
            facet.h_cols = None

        return cant_rot

    @staticmethod
    def _facet_heliostat_config(
            heliostat_config: CfgNode,
            position: torch.Tensor,
            span_x: torch.Tensor,
            span_y: torch.Tensor,
    ) -> CfgNode:
        heliostat_config = heliostat_config.clone()
        heliostat_config.defrost()

        # We change the shape in order to speed up construction.
        # Later, we need to do adjust all loaded values to be the same
        # as the parent heliostat.
        heliostat_config.SHAPE = 'ideal'
        heliostat_config.IDEAL.ROWS = 2
        heliostat_config.IDEAL.COLS = 2

        position = position.tolist()
        heliostat_config.POSITION_ON_FIELD = position
        heliostat_config.IDEAL.FACETS.POSITIONS = [position]
        heliostat_config.IDEAL.FACETS.SPANS_X = [span_x.tolist()]
        heliostat_config.IDEAL.FACETS.SPANS_Y = [span_y.tolist()]
        # Do not cant facet NURBS in their constructor; we do it
        # manually.
        heliostat_config.IDEAL.FACETS.CANTING.FOCUS_POINT = 0
        return heliostat_config

    @staticmethod
    def _facet_nurbs_config(
            nurbs_config: CfgNode,
            span_x: torch.Tensor,
            span_y: torch.Tensor,
    ) -> CfgNode:
        height = (th.linalg.norm(span_x) * 2).item()
        width = (th.linalg.norm(span_y) * 2).item()

        nurbs_config = nurbs_config.clone()
        nurbs_config.defrost()

        nurbs_config.HEIGHT = height
        nurbs_config.WIDTH = width

        nurbs_config.SET_UP_WITH_KNOWLEDGE = False
        nurbs_config.INITIALIZE_WITH_KNOWLEDGE = False
        return nurbs_config

    def _adjust_facet(
            self,
            facet: NURBSHeliostat,
            facet_index: int,
            position: torch.Tensor,
            span_x: torch.Tensor,
            span_y: torch.Tensor,
            orig_nurbs_config: CfgNode,
            nurbs_config: CfgNode,
    ) -> torch.Tensor:
        # "Load" values from parent heliostat.
        facet._discrete_points = self._discrete_points
        facet._discrete_points_ideal = self._discrete_points_ideal
        facet._normals = self._normals
        facet._normals_ideal = self._normals_ideal
        facet.params = self.params
        facet.h_rows = self.rows
        facet.h_cols = self.cols

        facet.height = nurbs_config.HEIGHT
        facet.width = nurbs_config.WIDTH
        # TODO initialize NURBS correctly
        # facet.position_on_field = self.position_on_field + position
        cant_rot = self._set_facet_points(facet, facet_index, position)

        facet.nurbs_cfg.defrost()
        facet.nurbs_cfg.SET_UP_WITH_KNOWLEDGE = \
            orig_nurbs_config.SET_UP_WITH_KNOWLEDGE
        facet.nurbs_cfg.INITIALIZE_WITH_KNOWLEDGE = \
            orig_nurbs_config.INITIALIZE_WITH_KNOWLEDGE
        facet.nurbs_cfg.freeze()

        facet.initialize_control_points(facet.ctrl_points)
        facet.initialize_eval_points()
        return cant_rot

    def _create_facet(
            self,
            facet_index: int,
            position: torch.Tensor,
            span_x: torch.Tensor,
            span_y: torch.Tensor,
            heliostat_config: CfgNode,
            nurbs_config: CfgNode,
            setup_params: bool,
    ) -> Tuple[NURBSHeliostat, torch.Tensor]:
        orig_nurbs_config = nurbs_config
        heliostat_config = self._facet_heliostat_config(
            heliostat_config,
            position,
            span_x,
            span_y,
        )
        nurbs_config = self._facet_nurbs_config(nurbs_config, span_x, span_y)

        facet = NURBSHeliostat(
            heliostat_config,
            nurbs_config,
            self.device,
            setup_params=False,
        )
        cant_rot = self._adjust_facet(
            facet,
            facet_index,
            position,
            span_x,
            span_y,
            orig_nurbs_config,
            nurbs_config,
        )
        if setup_params:
            facet.setup_params()
        return facet, cant_rot

    def _create_facets(
            self,
            heliostat_config: CfgNode,
            nurbs_config: CfgNode,
            setup_params: bool,
    ) -> List[Tuple[NURBSHeliostat, torch.Tensor]]:
        return [
            self._create_facet(
                i,
                position,
                span_x,
                span_y,
                heliostat_config,
                nurbs_config,
                setup_params,
            )
            for (i, (position, span_x, span_y)) in enumerate(zip(
                    self.facet_positions,
                    self.facet_spans_x,
                    self.facet_spans_y,
            ))
        ]

    def __len__(self) -> int:
        return sum(len(facet) for facet in self.facets)

    def setup_params(self) -> None:
        for facet in self.facets:
            facet.setup_params()

    def get_params(self) -> List[torch.Tensor]:
        return [
            param
            for facet in self.facets
            for param in facet.get_params()
        ]

    def _calc_normals_and_surface(
            self,
            reposition: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        total_size = len(self)
        surface_points = th.empty((total_size, 3), device=self.device)
        normals = th.empty((total_size, 3), device=self.device)

        i = 0
        for (facet, cant_rot) in zip(self.facets, self.cant_rots):
            curr_surface_points, curr_normals = \
                facet.discrete_points_and_normals()
            offset = len(curr_surface_points)

            if (
                    canting.canting_enabled(self._canting_cfg)
                    and self._canting_algo is not CantingAlgorithm.ACTIVE
            ):
                # We expect the position to be centered on zero for
                # canting, so cant before repositioning.
                # We could also concat and after rotation de-construct here for
                # possibly more speed.
                curr_surface_points = canting.apply_rotation(
                    cant_rot, curr_surface_points)
                curr_normals = canting.apply_rotation(cant_rot, curr_normals)

            if reposition:
                curr_surface_points = \
                    curr_surface_points + facet.position_on_field

            surface_points[i:i + offset] = curr_surface_points
            normals[i:i + offset] = curr_normals
            i += offset

        return surface_points, normals

    def discrete_points_and_normals(
            self,
            reposition: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        discrete_points, normals = self._calc_normals_and_surface(
            reposition=reposition)
        return discrete_points, normals

    def step(self, verbose: bool = False) -> None:  # type: ignore[override]
        facets = iter(self.facets)
        next(facets).step(verbose)
        for facet in facets:
            facet.step(False)

    @property  # type: ignore[misc]
    @functools.lru_cache()
    def dict_keys(self) -> Set[str]:
        """All keys we assume in the dictionary returned by `_to_dict`."""
        keys = super().dict_keys
        keys = keys.union({  # type: ignore[attr-defined]
            'nurbs_config',
            'facets',
        })
        return keys

    @functools.lru_cache()
    def _fixed_dict(self) -> Dict[str, Any]:
        data = super()._fixed_dict()
        data['nurbs_config'] = self.nurbs_cfg
        return data

    def _to_dict(self) -> Dict[str, Any]:
        data = super()._to_dict()
        data['facets'] = [
            facet._to_dict()
            for facet in self.facets
        ]
        return data

    @classmethod
    def from_dict(  # type: ignore[override]
            cls: Type[C],
            data: Dict[str, Any],
            device: th.device,
            config: Optional[CfgNode] = None,
            nurbs_config: Optional[CfgNode] = None,
            receiver_center: Union[torch.Tensor, List[float], None] = None,
            # Wether to disregard what standard initialization did and
            # load all data we have.
            restore_strictly: bool = False,
            setup_params: bool = True,
    ) -> C:
        if config is None:
            config = data['config']
        if nurbs_config is None:
            nurbs_config = data['nurbs_config']
        if receiver_center is None:
            receiver_center = data['receiver_center']

        self = cls(
            config,
            nurbs_config,
            device,
            receiver_center=receiver_center,
            setup_params=False,
        )
        self._from_dict(data, restore_strictly)

        for (facet, facet_data) in zip(self.facets, data['facets']):
            facet._from_dict(facet_data, restore_strictly)

        if setup_params:
            self.setup_params()
        return self

    def _from_dict(self, data: Dict[str, Any], restore_strictly: bool) -> None:
        super()._from_dict(data, restore_strictly)


class AlignedMultiNURBSHeliostat(AlignedNURBSHeliostat):
    _heliostat: MultiNURBSHeliostat  # type: ignore[assignment]

    def __init__(
            self,
            heliostat: MultiNURBSHeliostat,
            sun_direction: torch.Tensor,
            receiver_center: torch.Tensor,
    ) -> None:
        assert isinstance(heliostat, MultiNURBSHeliostat), \
            'can only align multi-NURBS heliostat'
        AlignedHeliostat.__init__(
            self,  # type: ignore[arg-type]
            heliostat,
            sun_direction,
            receiver_center,
            align_points=False,
        )

        if (
                canting.canting_enabled(self._heliostat._canting_cfg)
                and self._heliostat._canting_algo is CantingAlgorithm.ACTIVE
        ):
            self.facets = [
                facet.align(sun_direction, receiver_center)
                for facet in self._heliostat.facets
            ]
            self.device = self._heliostat.device

    def _calc_normals_and_surface(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
                canting.canting_enabled(self._heliostat._canting_cfg)
                and self._heliostat._canting_algo is CantingAlgorithm.ACTIVE
        ):
            hel_rotated, normal_vectors_rotated = \
                MultiNURBSHeliostat.discrete_points_and_normals(
                    self, reposition=False)  # type: ignore[arg-type]
            hel_rotated = hel_rotated + self._heliostat.position_on_field
        else:
            hel_rotated, normal_vectors_rotated = heliostat_models.rotate(
                self._heliostat, self.align_origin)

            # TODO Remove if translation is added to `rotate` function.
            # Place in field
            hel_rotated = hel_rotated + self._heliostat.position_on_field
            normal_vectors_rotated = (
                normal_vectors_rotated
                / th.linalg.norm(normal_vectors_rotated, dim=-1).unsqueeze(-1)
            )

        return hel_rotated, normal_vectors_rotated


MultiNURBSHeliostat.aligned_cls = AlignedMultiNURBSHeliostat
