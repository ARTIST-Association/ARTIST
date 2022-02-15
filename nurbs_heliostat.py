import functools
from typing import Tuple

import torch
import torch as th

import heliostat_models
from heliostat_models import AbstractHeliostat, AlignedHeliostat, Heliostat
import nurbs
from nurbs_progressive_growing import ProgressiveGrowing
import utils


def _calc_normals_and_surface(
        eval_points,
        degree_x: int,
        degree_y: int,
        ctrl_points,
        ctrl_weights,
        knots_x,
        knots_y,
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
    def __len__(self):
        raise NotImplementedError('please override `__len__`')

    def _calc_normals_and_surface(self):
        raise NotImplementedError(
            'please override `_calc_normals_and_surface`')

    def __call__(self):
        discrete_points, normals = self.discrete_points_and_normals()
        return (discrete_points, self.get_ray_directions(normals))

    def discrete_points_and_normals(self):
        discrete_points, normals = self._calc_normals_and_surface()
        return discrete_points, normals

    @property
    def discrete_points(self):
        discrete_points, _ = self._calc_normals_and_surface()
        return discrete_points

    @property
    def normals(self):
        _, normals = self._calc_normals_and_surface()
        return normals

    def get_ray_directions(self, normals=None):
        raise NotImplementedError('please override `get_ray_directions`')


class NURBSHeliostat(AbstractNURBSHeliostat, Heliostat):
    def __init__(
            self,
            heliostat_config,
            nurbs_config,
            device,
            setup_params=True,
    ):
        super().__init__(heliostat_config, device, setup_params=False)
        self.nurbs_cfg = nurbs_config
        if not self.nurbs_cfg.is_frozen():
            self.nurbs_cfg = self.nurbs_cfg.clone()
            self.nurbs_cfg.freeze()

        self._fix_spline_ctrl_weights = nurbs_config.FIX_SPLINE_CTRL_WEIGHTS
        self._fix_spline_knots = nurbs_config.FIX_SPLINE_KNOTS
        self._recalc_eval_points = self.nurbs_cfg.RECALCULATE_EVAL_POINTS

        spline_degree = nurbs_config.SPLINE_DEGREE
        self.degree_x = spline_degree
        self.degree_y = spline_degree
        self.h_rows = self.rows
        self.h_cols = self.cols
        self.rows = nurbs_config.ROWS
        self.cols = nurbs_config.COLS

        self._progressive_growing = ProgressiveGrowing(self)

        (
            ctrl_points,
            self.ctrl_weights,
            self.knots_x,
            self.knots_y,
        ) = nurbs.setup_nurbs_surface(
            self.degree_x, self.degree_y, self.rows, self.cols, self.device)

        self._orig_world_points = self._discrete_points.clone()

        utils.initialize_spline_knots(
            self.knots_x, self.knots_y, self.degree_x, self.degree_y)
        self.ctrl_weights[:] = 1

        self.initialize_control_points(ctrl_points)
        self.initialize_eval_points()
        if setup_params:
            self.setup_params()

    def __len__(self):
        return len(self.eval_points)

    def align(self, sun_direction, receiver_center):
        return AlignedNURBSHeliostat(self, sun_direction, receiver_center)

    def get_ray_directions(self, normals=None):
        raise NotImplementedError('Heliostat has to be aligned first')

    @property
    def fix_spline_ctrl_weights(self):
        return self._fix_spline_ctrl_weights

    @property
    def fix_spline_knots(self):
        return self._fix_spline_knots

    @property
    def recalc_eval_points(self):
        return self._recalc_eval_points

    def initialize_control_points(self, ctrl_points):
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
                # This, of course, is slower and consumer more memory.
                edge_factor = 5
                world_points, rows, cols = utils.make_structured_points(
                    self._discrete_points,
                    self.rows * edge_factor,
                    self.cols * edge_factor,
                )
            else:
                world_points = self._discrete_points
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

        self.set_ctrl_points(ctrl_points)

    def initialize_eval_points(self):
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

    def setup_params(self):
        self.ctrl_points_z.requires_grad_(True)
        optimize_xy = not self.nurbs_cfg.OPTIMIZE_Z_ONLY
        self.ctrl_points_xy.requires_grad_(optimize_xy)
        optimize_ctrl_weights = not self.fix_spline_ctrl_weights
        self.ctrl_weights.requires_grad_(optimize_ctrl_weights)

        optimize_knots = not self.fix_spline_knots
        self.knots_x.requires_grad_(optimize_knots)
        self.knots_y.requires_grad_(optimize_knots)

    def get_params(self):
        opt_params = [self.ctrl_points_z]
        if not self.nurbs_cfg.OPTIMIZE_Z_ONLY:
            opt_params.append(self.ctrl_points_xy)
        if not self.fix_spline_ctrl_weights:
            opt_params.append(self.ctrl_weights)

        if not self.fix_spline_knots:
            opt_params.append(self.knots_x)
            opt_params.append(self.knots_y)

        return opt_params

    @property
    def ctrl_points(self):
        return th.cat([
            self.ctrl_points_xy,
            self.ctrl_points_z,
        ], dim=-1)

    @ctrl_points.setter
    def ctrl_points(self):
        raise AttributeError(
            'ctrl_points is not a writable attribute; '
            'use `set_ctrl_points` instead'
        )

    def set_ctrl_points(self, ctrl_points):
        with th.no_grad():
            self.ctrl_points_xy = ctrl_points[:, :, :-1]
            self.ctrl_points_z = ctrl_points[:, :, -1:]

    def _invert_world_points(self):
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
    def eval_points(self):
        if self.recalc_eval_points:
            eval_points = self._invert_world_points()
        else:
            eval_points = self._eval_points
        return eval_points

    @property
    def shape(self):
        return self._progressive_growing.get_shape()

    def _calc_normals_and_surface(self):
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

        return surface_points, normals

    @th.no_grad()
    def step(self, verbose=False):
        self._progressive_growing.step(verbose)

    @property
    @functools.lru_cache()
    def dict_keys(self):
        keys = super().dict_keys
        keys = keys.union({
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
    def _fixed_dict(self):
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

    def _to_dict(self):
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
    def from_dict(
            cls,
            data,
            device,
            config=None,
            nurbs_config=None,
            # Wether to disregard what standard initialization did and
            # load all data we have.
            restore_strictly=False,
            setup_params=True,
    ):
        if config is None:
            config = data['config']
        if nurbs_config is None:
            nurbs_config = data['nurbs_config']

        self = cls(config, nurbs_config, device, setup_params=False)
        self._from_dict(data, restore_strictly)
        if setup_params:
            self.setup_params()
        return self

    def _from_dict(self, data, restore_strictly):
        # Keep normals from standard initialization here.
        normals = self._normals
        super()._from_dict(data, restore_strictly)
        self._normals = normals

        self.degree_x = data['degree_x']
        self.degree_y = data['degree_y']
        ctrl_points = data['control_points']
        ctrl_points.requires_grad_(False)
        self.set_ctrl_points(ctrl_points)
        self.ctrl_weights = data['control_point_weights']
        self.knots_x = data['knots_x']
        self.knots_y = data['knots_y']

        self._progressive_growing.set_step(data['_progressive_growing_step'])

        if restore_strictly:
            self._eval_points = data['evaluation_points']
            self._orig_world_points = data['original_world_points']
            self._normals = data['_heliostat_normals']
        else:
            self.initialize_eval_points()


class AlignedNURBSHeliostat(AbstractNURBSHeliostat):
    def __init__(self, heliostat, sun_direction, receiver_center):
        assert isinstance(heliostat, NURBSHeliostat), \
            'can only align NURBS heliostat'
        AlignedHeliostat.__init__(
            self,
            heliostat,
            sun_direction,
            receiver_center,
            align_points=False,
        )

    def __len__(self):
        return len(self._heliostat)

    def _calc_normals_and_surface(self):
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

    def get_ray_directions(self, normals=None):
        if normals is None:
            normals = self.normals
        return heliostat_models.reflect_rays_(self.from_sun, normals)
