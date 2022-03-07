import functools
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import torch
import torch as th
from yacs.config import CfgNode

import heliostat_models
from heliostat_models import AlignedHeliostat, Heliostat
from nurbs_heliostat import (
    AbstractNURBSHeliostat,
    AlignedNURBSHeliostat,
    NURBSHeliostat,
)
import utils

C = TypeVar('C', bound='MultiNURBSHeliostat')


def _indices_between(
        points: torch.Tensor,
        from_: torch.Tensor,
        to: torch.Tensor,
) -> torch.Tensor:
    indices = (
        (from_ <= points) & (points < to)
    ).all(dim=-1)
    return indices


class MultiNURBSHeliostat(AbstractNURBSHeliostat, Heliostat):
    def __init__(
            self,
            heliostat_config: CfgNode,
            nurbs_config: CfgNode,
            device: th.device,
            setup_params: bool = True,
            receiver_center: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(heliostat_config, device, setup_params=False)
        self.nurbs_cfg = nurbs_config
        if not self.nurbs_cfg.is_frozen():
            self.nurbs_cfg = self.nurbs_cfg.clone()
            self.nurbs_cfg.freeze()

        if (
                self.nurbs_cfg.FACETS.CANTING.ENABLED
                and not self.nurbs_cfg.FACETS.CANTING.ACTIVE
        ):
            assert receiver_center is not None, (
                'must have receiver center to cant heliostat '
                'toward when not using active canting'
            )
            self._receiver_center: Optional[torch.Tensor] = th.tensor(
                receiver_center,
                dtype=self.position_on_field.dtype,
                device=device,
            )
        else:
            self._receiver_center = None

        self.facets = self._create_facets(
            self.cfg, self.nurbs_cfg, setup_params=setup_params)

        self._discrete_points_ideal = th.cat([
            facet._discrete_points_ideal + facet.position_on_field
            for facet in self.facets
        ])
        self._ideal_normals = th.cat(
            [facet._ideal_normals for facet in self.facets])

    @staticmethod
    def angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return th.acos(
            th.dot(a, b)
            / (th.linalg.norm(a) * th.linalg.norm(b))
        )

    @staticmethod
    def rot_x_mat(
            angle: torch.Tensor,
            dtype: th.dtype,
            device: th.device,
    ) -> torch.Tensor:
        cos_angle = th.cos(angle)
        sin_angle = th.sin(angle)
        return th.tensor(
            [
                [1, 0, 0],
                [0, cos_angle, -sin_angle],
                [0, sin_angle, cos_angle],
            ],
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def rot_y_mat(
            angle: torch.Tensor,
            dtype: th.dtype,
            device: th.device,
    ) -> torch.Tensor:
        cos_angle = th.cos(angle)
        sin_angle = th.sin(angle)
        return th.tensor(
            [
                [cos_angle, 0, sin_angle],
                [0, 1, 0],
                [-sin_angle, 0, cos_angle],
            ],
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def rot_z_mat(
            angle: torch.Tensor,
            dtype: th.dtype,
            device: th.device,
    ) -> torch.Tensor:
        cos_angle = th.cos(angle)
        sin_angle = th.sin(angle)
        return th.tensor(
            [
                [cos_angle, -sin_angle, 0],
                [sin_angle, cos_angle, 0],
                [0, 0, 1],
            ],
            dtype=dtype,
            device=device,
        )

    def _apply_canting(
            self,
            position_on_field: torch.Tensor,
            discrete_points: torch.Tensor,
            discrete_points_ideal: Optional[torch.Tensor] = None,
            normals: Optional[torch.Tensor] = None,
            ideal_normals: Optional[torch.Tensor] = None,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:

        dtype = position_on_field.dtype

        h_normal = th.tensor(
            self.cfg.IDEAL.NORMAL_VECS,
            dtype=dtype,
            device=self.device,
        )
        hel_to_recv = self._receiver_center - self.position_on_field
        focus_distance = th.linalg.norm(hel_to_recv)
        focus_point = h_normal * focus_distance
        target_normal = focus_point - position_on_field
        target_normal /= th.linalg.norm(target_normal)

        target_y_angle = self.angle(
            h_normal,
            target_normal * th.tensor(
                [1, 0, 1],
                dtype=dtype,
                device=self.device,
            ),
        )
        if target_normal[0] < 0:
            target_y_angle = -target_y_angle
        rot_y = self.rot_y_mat(
            target_y_angle, dtype=dtype, device=self.device)
        rot_h_normal = rot_y @ h_normal

        target_x_angle = self.angle(
            rot_h_normal,
            target_normal,
        )
        if target_normal[1] > 0:
            target_x_angle = -target_x_angle
        rot_x = self.rot_x_mat(
            target_x_angle, dtype=dtype, device=self.device)

        full_rot = rot_x @ rot_y

        def look_at_receiver(hel_points):
            return th.matmul(full_rot, hel_points.T).T

        hel_rotated = look_at_receiver(discrete_points)

        if discrete_points_ideal is None:
            return hel_rotated

        hel_ideal_rotated = look_at_receiver(discrete_points_ideal)

        normals_rotated = look_at_receiver(normals)
        normals_rotated = (
            normals_rotated
            / th.linalg.norm(normals_rotated, dim=-1).unsqueeze(-1)
        )

        ideal_normals_rotated = look_at_receiver(ideal_normals)
        ideal_normals_rotated = (
            ideal_normals_rotated
            / th.linalg.norm(ideal_normals_rotated, dim=-1).unsqueeze(-1)
        )
        return (
            hel_rotated,
            hel_ideal_rotated,
            normals_rotated,
            ideal_normals_rotated,
        )

    def _set_facet_points(
            self,
            facet: NURBSHeliostat,
            position: torch.Tensor,
            span_x: torch.Tensor,
            span_y: torch.Tensor,
    ) -> None:
        from_xyz = position - span_y - span_x
        to_xyz = position + span_y + span_x
        # We ignore the z-axis here.
        indices = _indices_between(
            self._discrete_points_ideal[:, :-1],
            from_xyz[:-1],
            to_xyz[:-1],
        )

        facet_discrete_points = self._discrete_points[indices] - position
        facet_discrete_points_ideal = \
            self._discrete_points_ideal[indices] - position
        facet_normals = self._normals[indices]
        facet_ideal_normals = self._ideal_normals[indices]

        if (
                self.nurbs_cfg.FACETS.CANTING.ENABLED
                and not self.nurbs_cfg.FACETS.CANTING.ACTIVE
        ):
            (
                facet_discrete_points,
                facet_discrete_points_ideal,
                facet_normals,
                facet_ideal_normals,
            ) = self._apply_canting(
                position,
                facet_discrete_points,
                facet_discrete_points_ideal,
                facet_normals,
                facet_ideal_normals,
            )

        facet._discrete_points = facet_discrete_points
        facet._discrete_points_ideal = facet_discrete_points_ideal
        facet._orig_world_points = facet._discrete_points_ideal.clone()
        facet._normals = facet_normals
        facet._ideal_normals = facet_ideal_normals

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

    @staticmethod
    def _facet_heliostat_config(
            heliostat_config: CfgNode,
            position: torch.Tensor,
    ) -> CfgNode:
        heliostat_config = heliostat_config.clone()
        heliostat_config.defrost()

        # We change the shape in order to speed up construction.
        # Later, we need to do adjust all loaded values to be the same
        # as the parent heliostat.
        heliostat_config.SHAPE = 'ideal'
        heliostat_config.ROWS = 2
        heliostat_config.COLS = 2

        heliostat_config.POSITION_ON_FIELD = position.tolist()
        return heliostat_config

    @staticmethod
    def _facet_nurbs_config(
            nurbs_config: CfgNode,
            span_x: torch.Tensor,
            span_y: torch.Tensor,
    ) -> CfgNode:
        height = (th.linalg.norm(span_y) * 2).item()
        width = (th.linalg.norm(span_x) * 2).item()

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
            position: torch.Tensor,
            span_x: torch.Tensor,
            span_y: torch.Tensor,
            orig_nurbs_config: CfgNode,
            nurbs_config: CfgNode,
    ) -> None:
        # "Load" values from parent heliostat.
        facet._discrete_points = self._discrete_points
        facet._discrete_points_ideal = self._discrete_points_ideal
        facet._normals = self._normals
        facet._ideal_normals = self._ideal_normals
        facet.params = self.params
        facet.h_rows = self.rows
        facet.h_cols = self.cols

        facet.height = nurbs_config.HEIGHT
        facet.width = nurbs_config.WIDTH
        # TODO initialize NURBS correctly
        # facet.position_on_field = self.position_on_field + position
        self._set_facet_points(facet, position, span_x, span_y)

        facet.nurbs_cfg.defrost()
        facet.nurbs_cfg.SET_UP_WITH_KNOWLEDGE = \
            orig_nurbs_config.SET_UP_WITH_KNOWLEDGE
        facet.nurbs_cfg.INITIALIZE_WITH_KNOWLEDGE = \
            orig_nurbs_config.INITIALIZE_WITH_KNOWLEDGE
        facet.nurbs_cfg.freeze()

        facet.initialize_control_points(facet.ctrl_points)
        # If we don't fit the control points, we instead cant them.
        if (
                facet.nurbs_cfg.SET_UP_WITH_KNOWLEDGE
                and not facet.nurbs_cfg.INITIALIZE_WITH_KNOWLEDGE
                and self.nurbs_cfg.FACETS.CANTING.ENABLED
                and not self.nurbs_cfg.FACETS.CANTING.ACTIVE
        ):
            facet.set_ctrl_points(
                self._apply_canting(  # type: ignore[union-attr]
                    position,
                    facet.ctrl_points.reshape(-1, facet.ctrl_points.shape[-1]),
                ).reshape(facet.ctrl_points.shape),
            )

        facet.initialize_eval_points()

    def _create_facet(
            self,
            position: torch.Tensor,
            span_x: torch.Tensor,
            span_y: torch.Tensor,
            heliostat_config: CfgNode,
            nurbs_config: CfgNode,
            setup_params: bool,
    ) -> NURBSHeliostat:
        orig_nurbs_config = nurbs_config
        heliostat_config = self._facet_heliostat_config(
            heliostat_config, position)
        nurbs_config = self._facet_nurbs_config(nurbs_config, span_x, span_y)

        facet = NURBSHeliostat(
            heliostat_config,
            nurbs_config,
            self.device,
            setup_params=False,
        )
        self._adjust_facet(
            facet,
            position,
            span_x,
            span_y,
            orig_nurbs_config,
            nurbs_config,
        )
        if setup_params:
            facet.setup_params()
        return facet

    @staticmethod
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

    def _create_facets(
            self,
            heliostat_config: CfgNode,
            nurbs_config: CfgNode,
            setup_params: bool,
    ) -> List[NURBSHeliostat]:
        positions = utils.with_outer_list(self.nurbs_cfg.FACETS.POSITIONS)
        spans_x: List[List[float]] = utils.with_outer_list(
            self.nurbs_cfg.FACETS.SPANS_X)
        spans_x = self._broadcast_spans(spans_x, len(positions))
        spans_y: List[List[float]] = utils.with_outer_list(
            self.nurbs_cfg.FACETS.SPANS_Y)
        spans_y = self._broadcast_spans(spans_y, len(positions))

        return [
            self._create_facet(
                position,
                span_x,
                span_y,
                heliostat_config,
                nurbs_config,
                setup_params,
            )
            for (position, span_x, span_y) in map(
                    lambda tup: map(
                        lambda vec: th.tensor(vec, device=self.device),
                        tup,
                    ),
                    zip(positions, spans_x, spans_y),
            )
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

    def align(  # type: ignore[override]
            self,
            sun_direction: torch.Tensor,
            receiver_center: torch.Tensor,
    ) -> 'AlignedMultiNURBSHeliostat':
        return AlignedMultiNURBSHeliostat(self, sun_direction, receiver_center)

    def _calc_normals_and_surface(
            self,
            reposition: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        total_size = len(self)
        surface_points = th.empty((total_size, 3), device=self.device)
        normals = th.empty((total_size, 3), device=self.device)

        i = 0
        for facet in self.facets:
            curr_surface_points, curr_normals = \
                facet.discrete_points_and_normals()
            offset = len(curr_surface_points)

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

            'receiver_center',
        })
        return keys

    @functools.lru_cache()
    def _fixed_dict(self) -> Dict[str, Any]:
        data = super()._fixed_dict()
        data['nurbs_config'] = self.nurbs_cfg
        data['receiver_center'] = self._receiver_center
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
            receiver_center: Optional[torch.Tensor] = None,
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

        if restore_strictly:
            self._receiver_center = data['receiver_center']


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
                self._heliostat.nurbs_cfg.FACETS.CANTING.ENABLED
                and self._heliostat.nurbs_cfg.FACETS.CANTING.ACTIVE
        ):
            self.facets = [
                facet.align(sun_direction, receiver_center)
                for facet in self._heliostat.facets
            ]
            self.device = self._heliostat.device

    def _calc_normals_and_surface(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
                self._heliostat.nurbs_cfg.FACETS.CANTING.ENABLED
                and self._heliostat.nurbs_cfg.FACETS.CANTING.ACTIVE
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
