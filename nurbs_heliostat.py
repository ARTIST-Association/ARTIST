import functools
from typing import (
    Any,
    cast,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from pytorch3d.transforms import Transform3d
import torch
import torch as th
from yacs.config import CfgNode

import canting
import facets
import heliostat_models
from heliostat_models import AbstractHeliostat, AlignedHeliostat, Heliostat
import nurbs
from nurbs_progressive_growing import ProgressiveGrowing
import utils

C = TypeVar('C', bound='NURBSHeliostat')


def _calc_normals_and_surface(
        eval_points: torch.Tensor,
        degree_x: int,
        degree_y: int,
        ctrl_points: torch.Tensor,
        ctrl_weights: torch.Tensor,
        knots_x: torch.Tensor,
        knots_y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return nurbs.calc_normals_and_surface_slow(
        eval_points[:, 0],
        eval_points[:, 1],
        degree_x,
        degree_y,
        ctrl_points,
        ctrl_weights,
        knots_x,
        knots_y,
    )


class AbstractNURBSHeliostat(AbstractHeliostat):
    def __len__(self) -> int:
        raise NotImplementedError('please override `__len__`')

    def _calc_normals_and_surface(
            self,
            do_canting: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            'please override `_calc_normals_and_surface`')

    def __call__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        discrete_points, normals = self.discrete_points_and_normals()
        return (discrete_points, self.get_ray_directions(normals))

    def discrete_points_and_normals(self) -> Tuple[torch.Tensor, torch.Tensor]:
        discrete_points, normals = self._calc_normals_and_surface()
        return discrete_points, normals

    def raw_discrete_points_and_normals(
            self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        discrete_points, normals = self._calc_normals_and_surface(
            do_canting=False)
        return discrete_points, normals

    @property
    def discrete_points(self) -> torch.Tensor:
        discrete_points, _ = self._calc_normals_and_surface()
        return discrete_points

    @property
    def normals(self) -> torch.Tensor:
        _, normals = self._calc_normals_and_surface()
        return normals

    def get_ray_directions(
            self,
            normals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError('please override `get_ray_directions`')


class NURBSHeliostat(AbstractNURBSHeliostat, Heliostat):
    _OPTIMIZABLE_NAMES = ['surface_0', 'surface_1', 'surface_2']

    def __init__(
            self,
            heliostat_config: CfgNode,
            nurbs_config: CfgNode,
            device: th.device,
            receiver_center: Union[torch.Tensor, List[float], None] = None,
            sun_directions: Union[
                torch.Tensor,
                List[List[float]],
                None,
            ] = None,
            setup_params: bool = True,
    ) -> None:
        super().__init__(
            heliostat_config,
            device,
            setup_params=False,
            receiver_center=receiver_center,
            sun_directions=sun_directions,
        )
        assert len(self.facets.positions) == 1, (
            'cannot handle multiple facets with `NURBSHeliostat`; '
            'please use `MultiNURBSHeliostat` instead.'
        )

        self.nurbs_cfg = nurbs_config
        if not self.nurbs_cfg.is_frozen():
            self.nurbs_cfg = self.nurbs_cfg.clone()
            self.nurbs_cfg.freeze()

        self._fix_spline_ctrl_weights: bool = \
            self.nurbs_cfg.FIX_SPLINE_CTRL_WEIGHTS
        self._fix_spline_knots: bool = self.nurbs_cfg.FIX_SPLINE_KNOTS
        self._recalc_eval_points: bool = self.nurbs_cfg.RECALCULATE_EVAL_POINTS

        spline_degree: int = self.nurbs_cfg.SPLINE_DEGREE
        self.degree_x = spline_degree
        self.degree_y = spline_degree
        self.h_rows: Optional[int] = self.rows
        self.h_cols: Optional[int] = self.cols
        self.rows: int = self.nurbs_cfg.ROWS
        self.cols: int = self.nurbs_cfg.COLS

        self._progressive_growing = ProgressiveGrowing(self)

        (
            ctrl_points,
            ctrl_weights,
            knots_x,
            knots_y,
        ) = nurbs.setup_nurbs_surface(
            self.degree_x, self.degree_y, self.rows, self.cols, self.device)

        self._orig_world_points = (
            facets.make_unfacetted(self.get_raw_discrete_points_ideal())
            + self.facets.positions[0]
        )

        utils.initialize_spline_knots(
            knots_x, knots_y, self.degree_x, self.degree_y)
        ctrl_weights[:] = 1

        self.split_nurbs_params(ctrl_weights, knots_x, knots_y)
        self.initialize_control_points(ctrl_points)
        with th.no_grad():
            self.set_ctrl_points(ctrl_points)
        self.initialize_eval_points()
        if setup_params:
            self.setup_params()

    def __len__(self) -> int:
        return len(self.eval_points)

    def get_ray_directions(
            self,
            normals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError('Heliostat has to be aligned first')

    @property
    def fix_spline_ctrl_weights(self) -> bool:
        return self._fix_spline_ctrl_weights

    @property
    def fix_spline_knots(self) -> bool:
        return self._fix_spline_knots

    @property
    def recalc_eval_points(self) -> bool:
        return self._recalc_eval_points

    def initialize_control_points(self, ctrl_points: torch.Tensor) -> None:
        nurbs_config = self.nurbs_cfg

        if nurbs_config.SET_UP_WITH_KNOWLEDGE:
            width = self.width
            height = self.height
        else:
            # Use perfect, unrotated heliostat at `position_on_field` as
            # starting point with width and height as initially guessed.
            width = nurbs_config.WIDTH
            height = nurbs_config.HEIGHT

        utils.initialize_spline_ctrl_points(
            ctrl_points,
            # We are only moved to `position_on_field` upon alignment,
            # so initialize at the origin where the heliostat's discrete
            # points are as well.
            th.zeros_like(self.position_on_field),
            self.rows,
            self.cols,
            width,
            height,
        )
        if nurbs_config.INITIALIZE_WITH_KNOWLEDGE:
            if self.h_rows is None or self.h_cols is None:
                # Higher edge factor creates more interpolated points
                # per control point, improving approximation accuracy.
                # This, of course, is slower and consumes more memory.
                edge_factor = 5
                world_points, rows, cols = utils.make_structured_points(
                    self._orig_world_points,
                    self.rows * edge_factor,
                    self.cols * edge_factor,
                )
            else:
                world_points = self._orig_world_points
                rows = self.h_rows
                cols = self.h_cols

            utils.initialize_spline_ctrl_points_perfectly(
                ctrl_points,
                world_points,
                rows,
                cols,
                self.degree_x,
                self.degree_y,
                self.knots_x,
                self.knots_y,
                self.nurbs_cfg.INITIALIZE_Z_ONLY,
                True,
            )

    def initialize_eval_points(self) -> None:
        if self.nurbs_cfg.SET_UP_WITH_KNOWLEDGE:
            if not self.recalc_eval_points:
                self._eval_points = \
                    utils.initialize_spline_eval_points_perfectly(
                        self._orig_world_points,
                        self.degree_x,
                        self.degree_y,
                        self.ctrl_points,
                        self.ctrl_weights,
                        self.knots_x,
                        self.knots_y,
                    )
        else:
            # Unless we change the knots, we don't need to recalculate
            # as we simply distribute the points uniformly.
            self._recalc_eval_points = False
            self._eval_points = utils.initialize_spline_eval_points(
                self.rows, self.cols, self.device)

    def _optimizables(self) -> Dict[str, List[torch.Tensor]]:
        ctrl_points_inner = self.ctrl_points_inner
        ctrl_points_edges = self.ctrl_points_edges
        ctrl_points_corners = self.ctrl_points_corners

        nurbs_optimizables_inner = ctrl_points_inner[2::3]
        nurbs_optimizables_edges = ctrl_points_edges[2::3]
        nurbs_optimizables_corners = ctrl_points_corners[2::3]

        if not self.nurbs_cfg.OPTIMIZE_Z_ONLY:
            nurbs_optimizables_inner.extend([
                tensor
                for tensors in zip(
                    ctrl_points_inner[::3],
                    ctrl_points_inner[1::3],
                )
                for tensor in tensors
            ])
            nurbs_optimizables_edges.extend([
                tensor
                for tensors in zip(
                    ctrl_points_edges[::3],
                    ctrl_points_edges[1::3],
                )
                for tensor in tensors
            ])
            nurbs_optimizables_corners.extend([
                tensor
                for tensors in zip(
                    ctrl_points_corners[::3],
                    ctrl_points_corners[1::3],
                )
                for tensor in tensors
            ])
        if not self.fix_spline_ctrl_weights:
            nurbs_optimizables_inner.extend(self.ctrl_weights_inner)
            nurbs_optimizables_edges.extend(self.ctrl_weights_edges)
            nurbs_optimizables_corners.extend(self.ctrl_weights_corners)

        if not self.fix_spline_knots:
            nurbs_optimizables_inner.extend(self.knots_x_inner)
            nurbs_optimizables_edges.extend(self.knots_x_edges)
            nurbs_optimizables_inner.extend(self.knots_y_inner)
            nurbs_optimizables_edges.extend(self.knots_y_edges)

        return {
            'surface_0': nurbs_optimizables_inner,
            'surface_1': nurbs_optimizables_edges,
            'surface_2': nurbs_optimizables_corners,
        }

    @property
    def ctrl_points_xy(self) -> List[torch.Tensor]:
        first_row = [
            row_slice
            for row_slices in self._ctrl_points_splits[0]
            for row_slice in row_slices[:-1]
        ]
        inner_rows = [
            rows_slice
            for rows_slices in self._ctrl_points_splits[1]
            for rows_slice in rows_slices[:-1]
        ]
        last_row = [
            row_slice
            for row_slices in self._ctrl_points_splits[2]
            for row_slice in row_slices[:-1]
        ]

        return first_row + inner_rows + last_row

    @property
    def ctrl_points_z(self) -> List[torch.Tensor]:
        first_row = [
            row_slice[-1:]
            for rows_slices in self._ctrl_points_splits[0]
            for row_slice in rows_slices
        ]
        inner_rows = [
            row_slice[-1:]
            for rows_slices in self._ctrl_points_splits[1]
            for row_slice in rows_slices
        ]
        last_row = [
            row_slice[-1:]
            for rows_slices in self._ctrl_points_splits[2]
            for row_slice in rows_slices
        ]

        return first_row + inner_rows + last_row

    @property
    def ctrl_points_inner(self) -> List[torch.Tensor]:
        return self._ctrl_points_splits[1][1]

    @property
    def ctrl_points_edges(self) -> List[torch.Tensor]:
        first_row_edge = self._ctrl_points_splits[0][1]
        inner_rows_edges = [
            row_slice
            for row_slices in self._ctrl_points_splits[1][::2]
            for row_slice in row_slices
        ]
        last_row_edge = self._ctrl_points_splits[2][1]

        return first_row_edge + inner_rows_edges + last_row_edge

    @property
    def ctrl_points_corners(self) -> List[torch.Tensor]:
        first_corners = [
            row_slice
            for row_slices in self._ctrl_points_splits[0][::2]
            for row_slice in row_slices
        ]
        last_corners = [
            row_slice
            for row_slices in self._ctrl_points_splits[2][::2]
            for row_slice in row_slices
        ]
        return first_corners + last_corners

    @property
    def ctrl_points(self) -> torch.Tensor:
        # self._ctrl_points_splits[0][0] per-dim
        # self._ctrl_points_splits[0] per-column
        # self._ctrl_points_splits per-row
        first_row = th.cat([
            th.cat(row_slice, dim=-1)
            for row_slice in self._ctrl_points_splits[0]
        ], dim=1)
        inner_rows = th.cat([
            th.cat(rows_slice, dim=-1)
            for rows_slice in self._ctrl_points_splits[1]
        ], dim=1)
        last_row = th.cat([
            th.cat(row_slice, dim=-1)
            for row_slice in self._ctrl_points_splits[2]
        ], dim=1)

        return th.cat([first_row, inner_rows, last_row], dim=0)

    @ctrl_points.setter
    def ctrl_points(self, new_ctrl_points: torch.Tensor) -> None:
        raise AttributeError(
            '`ctrl_points` is not a writable attribute; '
            'use `set_ctrl_points` instead'
        )

    @property
    def ctrl_weights_inner(self) -> List[torch.Tensor]:
        return [self._ctrl_weights_splits[1][1]]

    @property
    def ctrl_weights_edges(self) -> List[torch.Tensor]:
        first_row_edge = [self._ctrl_weights_splits[0][1]]
        inner_rows_edges = [
            row_slice
            for row_slices in self._ctrl_weights_splits[1][::2]
            for row_slice in row_slices
        ]
        last_row_edge = [self._ctrl_weights_splits[2][1]]

        return first_row_edge + inner_rows_edges + last_row_edge

    @property
    def ctrl_weights_corners(self) -> List[torch.Tensor]:
        first_corners = [
            row_slice
            for row_slice in self._ctrl_weights_splits[0][::2]
        ]
        last_corners = [
            row_slice
            for row_slice in self._ctrl_weights_splits[2][::2]
        ]
        return first_corners + last_corners

    @property
    def ctrl_weights(self) -> torch.Tensor:
        first_row = th.cat(self._ctrl_weights_splits[0], dim=1)
        inner_rows = th.cat(self._ctrl_weights_splits[1], dim=1)
        last_row = th.cat(self._ctrl_weights_splits[2], dim=1)
        return th.cat([first_row, inner_rows, last_row], dim=0)

    @ctrl_weights.setter
    def ctrl_weights(self, new_ctrl_weights: torch.Tensor) -> None:
        raise AttributeError(
            '`ctrl_weights` is not a writable attribute; '
            'use `set_ctrl_weights` instead'
        )

    @property
    def knots_x_inner(self) -> List[torch.Tensor]:
        return self._knots_x_splits[1:2]

    @property
    def knots_x_edges(self) -> List[torch.Tensor]:
        return self._knots_x_splits[::2]

    @property
    def knots_x(self) -> torch.Tensor:
        return th.cat(self._knots_x_splits)

    @knots_x.setter
    def knots_x(self, new_knots_x: torch.Tensor) -> None:
        raise AttributeError(
            '`knots_x` is not a writable attribute; '
            'use `set_knots_x` instead'
        )

    @property
    def knots_y_inner(self) -> List[torch.Tensor]:
        return self._knots_y_splits[1:2]

    @property
    def knots_y_edges(self) -> List[torch.Tensor]:
        return self._knots_y_splits[::2]

    @property
    def knots_y(self) -> torch.Tensor:
        return th.cat(self._knots_y_splits)

    @knots_y.setter
    def knots_y(self, new_knots_y: torch.Tensor) -> None:
        raise AttributeError(
            '`knots_y` is not a writable attribute; '
            'use `set_knots_y` instead'
        )

    def split_nurbs_params(
            self,
            ctrl_weights: torch.Tensor,
            knots_x: torch.Tensor,
            knots_y: torch.Tensor,
    ) -> None:
        with th.no_grad():
            self.set_ctrl_weights(ctrl_weights)
            self.set_knots_x(knots_x)
            self.set_knots_y(knots_y)

    def set_ctrl_points(self, ctrl_points: torch.Tensor) -> None:
        with th.no_grad():
            # Split in last dimension, then column dimension, then row
            # dimension.
            first_row = [
                [row_slice[:, :, dim].unsqueeze(-1) for dim in range(3)]
                for row_slice in [
                        ctrl_points[:1, :1],
                        ctrl_points[:1, 1:-1],
                        ctrl_points[:1, -1:],
                ]
            ]
            inner_rows = [
                [rows_slice[:, :, dim].unsqueeze(-1) for dim in range(3)]
                for rows_slice in [
                        ctrl_points[1:-1, :1],
                        ctrl_points[1:-1, 1:-1],
                        ctrl_points[1:-1, -1:],
                ]
            ]
            last_row = [
                [row_slice[:, :, dim].unsqueeze(-1) for dim in range(3)]
                for row_slice in [
                        ctrl_points[-1:, :1],
                        ctrl_points[-1:, 1:-1],
                        ctrl_points[-1:, -1:],
                ]
            ]

            self._ctrl_points_splits = [first_row, inner_rows, last_row]
            assert (self.ctrl_points == ctrl_points).all()

    def set_ctrl_weights(self, ctrl_weights: torch.Tensor) -> None:
        with th.no_grad():
            # Split in column dimension, then row dimension.
            first_row = [
                ctrl_weights[:1, :1],
                ctrl_weights[:1, 1:-1],
                ctrl_weights[:1, -1:],
            ]
            inner_rows = [
                ctrl_weights[1:-1, :1],
                ctrl_weights[1:-1, 1:-1],
                ctrl_weights[1:-1, -1:],
            ]
            last_row = [
                ctrl_weights[-1:, :1],
                ctrl_weights[-1:, 1:-1],
                ctrl_weights[-1:, -1:],
            ]

            self._ctrl_weights_splits = [first_row, inner_rows, last_row]
            assert (self.ctrl_weights == ctrl_weights).all()

    def set_knots_x(self, knots_x: torch.Tensor) -> None:
        with th.no_grad():
            self._knots_x_splits = self._split_knots(knots_x)
            assert (self.knots_x == knots_x).all()

    def set_knots_y(self, knots_y: torch.Tensor) -> None:
        with th.no_grad():
            self._knots_y_splits = self._split_knots(knots_y)
            assert (self.knots_y == knots_y).all()

    @staticmethod
    def _split_knots(knots: torch.Tensor) -> List[torch.Tensor]:
        with th.no_grad():
            return [knots[:1], knots[1:-1], knots[-1:]]

    def _invert_world_points(self) -> torch.Tensor:
        return utils.initialize_spline_eval_points_perfectly(
            self._orig_world_points,
            self.degree_x,
            self.degree_y,
            self.ctrl_points,
            self.ctrl_weights,
            self.knots_x,
            self.knots_y,
        )

    @property
    def eval_points(self) -> torch.Tensor:
        if self.recalc_eval_points:
            eval_points = self._invert_world_points()
        else:
            eval_points = self._eval_points
        return eval_points

    @property
    def shape(self) -> Tuple[int, int]:
        return self._progressive_growing.get_shape()

    def _calc_normals_and_surface(
            self,
            do_canting: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eval_points = self.eval_points
        ctrl_points, ctrl_weights, knots_x, knots_y = \
            self._progressive_growing.select()

        surface_points, normals = nurbs.calc_normals_and_surface_slow(
            eval_points[:, 0],
            eval_points[:, 1],
            self.degree_x,
            self.degree_y,
            ctrl_points,
            ctrl_weights,
            knots_x,
            knots_y,
        )

        if do_canting:
            position = self.facets.positions[0]
            surface_points = surface_points - position
            surface_points = canting.apply_rotation(
                self.facets.cant_rots[0], surface_points)
            normals = canting.apply_rotation(self.facets.cant_rots[0], normals)
            surface_points = surface_points + position

        return surface_points, normals

    def raw_discrete_points_and_normals(
            self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        discrete_points, normals = self._calc_normals_and_surface(
            do_canting=False)
        discrete_points = discrete_points - self.facets.positions[0]
        return discrete_points, normals

    @th.no_grad()
    def step(self, verbose: bool = False) -> None:  # type: ignore[override]
        self._progressive_growing.step(verbose)

    @property  # type: ignore[misc]
    @functools.lru_cache()
    def dict_keys(self) -> Set[str]:
        keys = super().dict_keys
        keys = cast(Set[str], keys).union({
            'degree_x',
            'degree_y',
            'control_points',
            'control_point_weights',
            'knots_x',
            'knots_y',

            'evaluation_points',
            'original_world_points',
            '_progressive_growing_step',

            'nurbs_config',
        })
        return keys

    @functools.lru_cache()
    def _fixed_dict(self) -> Dict[str, Any]:
        data = super()._fixed_dict()
        data.update({
            'degree_x': self.degree_x,
            'degree_y': self.degree_y,

            'nurbs_config': self.nurbs_cfg,
        })

        if self.fix_spline_ctrl_weights:
            data['control_point_weights'] = self.ctrl_weights

        if self.fix_spline_knots:
            data['knots_x'] = self.knots_x
            data['knots_y'] = self.knots_y

        if not self.recalc_eval_points:
            data['evaluation_points'] = self.eval_points
        return data

    def _to_dict(self) -> Dict[str, Any]:
        data = super()._to_dict()

        data.update({
            'control_points': self.ctrl_points.clone(),

            'original_world_points': self._orig_world_points,
            '_progressive_growing_step': self._progressive_growing.get_step(),
        })

        if not self.fix_spline_ctrl_weights:
            data['control_point_weights'] = self.ctrl_weights.clone()

        if not self.fix_spline_knots:
            data['knots_x'] = self.knots_x.clone()
            data['knots_y'] = self.knots_y.clone()

        if self.recalc_eval_points:
            data['evaluation_points'] = self.eval_points.clone()
        return data

    @classmethod
    def from_dict(  # type: ignore[override]
            cls: Type[C],
            data: Dict[str, Any],
            device: th.device,
            config: Optional[CfgNode] = None,
            nurbs_config: Optional[CfgNode] = None,
            receiver_center: Union[torch.Tensor, List[float], None] = None,
            sun_directions: Union[
                torch.Tensor,
                List[List[float]],
                None,
            ] = None,
            # Wether to disregard what standard initialization did and
            # load all data we have.
            restore_strictly: bool = False,
            setup_params: bool = True,
    ) -> C:
        if config is None:
            config = data['config']
        if nurbs_config is None:
            nurbs_config = data['nurbs_config']

        self = cls(
            config,
            nurbs_config,
            device,
            receiver_center=receiver_center,
            sun_directions=sun_directions,
            setup_params=False,
        )
        self._from_dict(data, restore_strictly)
        if setup_params:
            self.setup_params()
        return self

    def _from_dict(self, data: Dict[str, Any], restore_strictly: bool) -> None:
        # Keep normals from standard initialization here.
        normals = self._normals
        super()._from_dict(data, restore_strictly)
        self._normals = normals

        self.degree_x = data['degree_x']
        self.degree_y = data['degree_y']
        ctrl_points = data['control_points']
        ctrl_points.requires_grad_(False)
        self.set_ctrl_points(ctrl_points)
        ctrl_weights = data['control_point_weights']
        ctrl_weights.requires_grad_(False)
        self.set_ctrl_weights(ctrl_weights)
        knots_x = data['knots_x']
        knots_x.requires_grad_(False)
        self.set_knots_x(knots_x)
        knots_y = data['knots_y']
        knots_y.requires_grad_(False)
        self.set_knots_y(knots_y)

        self._progressive_growing.set_step(data['_progressive_growing_step'])

        if restore_strictly:
            self._eval_points = data['evaluation_points']
            self._orig_world_points = data['original_world_points']
            self._normals = data['_heliostat_normals']
        else:
            self.initialize_eval_points()


class AlignedNURBSHeliostat(AbstractNURBSHeliostat):
    _heliostat: NURBSHeliostat
    align_origin: List[Transform3d]
    from_sun: torch.Tensor

    def __init__(
            self,
            heliostat: NURBSHeliostat,
            sun_direction: torch.Tensor,
            aim_point: torch.Tensor,
    ) -> None:
        assert isinstance(heliostat, NURBSHeliostat), \
            'can only align NURBS heliostat'
        AlignedHeliostat.__init__(
            cast(AlignedHeliostat, self),
            heliostat,
            sun_direction,
            aim_point,
            align_points=False,
        )

    def __len__(self) -> int:
        return len(self._heliostat)

    def _calc_normals_and_surface(
            self,
            do_canting: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # print('calcing normals and surface')
        if canting.is_like_active(self._heliostat.canting_algo):
            hel_rotated, normal_vectors_rotated = \
                AlignedHeliostat.align_facets(
                    cast(AlignedHeliostat, self),
                    reposition=isinstance(
                        self._heliostat.canting_algo,
                        canting.ActiveCanting,
                    ),
                )
        else:
            hel_rotated, normal_vectors_rotated = heliostat_models.rotate(
                self._heliostat, self.align_origin[0])
        # TODO Remove if translation is added to `rotate` function.
        # Place in field
        hel_rotated = hel_rotated + self._heliostat.position_on_field
        normal_vectors_rotated = (
            normal_vectors_rotated
            / th.linalg.norm(normal_vectors_rotated, dim=-1).unsqueeze(-1)
        )
        return hel_rotated, normal_vectors_rotated

    def get_ray_directions(
            self,
            normals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if normals is None:
            normals = self.normals
        return heliostat_models.reflect_rays_(self.from_sun, normals)


NURBSHeliostat.aligned_cls = AlignedNURBSHeliostat
