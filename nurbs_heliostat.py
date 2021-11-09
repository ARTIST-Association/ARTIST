import functools
from typing import Tuple

import torch
import torch as th

import heliostat_models
from heliostat_models import AlignmentState, Heliostat
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


class NURBSHeliostat(Heliostat):
    DISABLE_CACHING = True

    def __init__(self, heliostat_config, nurbs_config, device):
        super().__init__(heliostat_config, device)
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
        self.rows = heliostat_config.NURBS.ROWS
        self.cols = heliostat_config.NURBS.COLS

        self.reset_cache()

        self._progressive_growing = ProgressiveGrowing(self)

        (
            ctrl_points,
            self.ctrl_weights,
            self.knots_x,
            self.knots_y,
        ) = nurbs.setup_nurbs_surface(
            self.degree_x, self.degree_y, self.rows, self.cols, self.device)

        self._orig_world_points = self._discrete_points_orig.clone()

        utils.initialize_spline_knots(
            self.knots_x, self.knots_y, self.degree_x, self.degree_y)
        self.ctrl_weights[:] = 1

        self.initialize_control_points(ctrl_points)
        self.initialize_eval_points()

    def reset_cache(self):
        self._ctrl_points_cache_valid = False
        self._eval_points_cache_valid = False
        self._orig_surface_cache_valid = False
        self._aligned_surface_cache_valid = False

    def invalidate_control_points_caches(self):
        self.reset_cache()
        # These can stay the same even after a training step;
        # unless we want to recalculate.
        self._eval_points_cache_valid = not self._recalc_eval_points

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
        heliostat_config = self.cfg
        nurbs_config = self.nurbs_cfg

        if nurbs_config.SET_UP_WITH_KNOWLEDGE:
            utils.initialize_spline_ctrl_points(
                ctrl_points,
                self.position_on_field,
                self.rows,
                self.cols,
                self.width,
                self.height,
            )
        else:
            # Use perfect, unrotated heliostat at `position_on_field` as
            # starting point with width and height as initially guessed.
            utils.initialize_spline_ctrl_points(
                ctrl_points,
                self.position_on_field,
                self.rows,
                self.cols,
                heliostat_config.NURBS.WIDTH,
                heliostat_config.NURBS.HEIGHT,
            )

        if nurbs_config.INITIALIZE_WITH_KNOWLEDGE:
            utils.adjust_spline_ctrl_points(
                ctrl_points,
                self._discrete_points_orig,
                self.nurbs_cfg.OPTIMIZE_Z_ONLY,
                k=4,
            )

        self.ctrl_points_xy = ctrl_points[:, :, :-1]
        self.ctrl_points_z = ctrl_points[:, :, -1:]

    def initialize_eval_points(self):
        if self.nurbs_cfg.SET_UP_WITH_KNOWLEDGE:
            if not self.recalc_eval_points:
                self._cached_eval_points = \
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
            self._cached_eval_points = utils.initialize_spline_eval_points(
                self.rows, self.cols, self.device)
        self._eval_points_cache_valid = True

    def setup_params(self):
        self.ctrl_points_z.requires_grad_(True)
        optimize_xy = not self.nurbs_cfg.OPTIMIZE_Z_ONLY
        self.ctrl_points_xy.requires_grad_(optimize_xy)
        optimize_ctrl_weights = not self.fix_spline_ctrl_weights
        self.ctrl_weights.requires_grad_(optimize_ctrl_weights)

        optimize_knots = not self.fix_spline_knots
        self.knots_x.requires_grad_(optimize_knots)
        self.knots_y.requires_grad_(optimize_knots)

        opt_params = [self.ctrl_points_z]
        if not self.nurbs_cfg.OPTIMIZE_Z_ONLY:
            opt_params.append(self.ctrl_points_xy)
        if not self.fix_spline_ctrl_weights:
            opt_params.append(self.ctrl_weights)

        if not self.fix_spline_knots:
            opt_params.append(self.knots_x)
            opt_params.append(self.knots_y)

        return opt_params

    def _needs_ctrl_points_update(self):
        return self.DISABLE_CACHING or not self._ctrl_points_cache_valid

    def _needs_eval_points_update(self):
        return self.DISABLE_CACHING or not self._eval_points_cache_valid

    def _needs_unaligned_surface_update(self):
        # FIXME does not work correctly
        return True
        return self.DISABLE_CACHING or not self._orig_surface_cache_valid

    def _needs_aligned_surface_update(self):
        return self.DISABLE_CACHING or not self._aligned_surface_cache_valid

    @property
    def ctrl_points(self):
        if self._needs_ctrl_points_update():
            self._cached_ctrl_points = th.cat([
                self.ctrl_points_xy,
                self.ctrl_points_z,
            ], dim=-1)
            self._ctrl_points_cache_valid = True
        return self._cached_ctrl_points

    @ctrl_points.setter
    def ctrl_points(self):
        raise AttributeError('ctrl_points is not a writable attribute')

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
        if self.recalc_eval_points and self._needs_eval_points_update():
            self._cached_eval_points = self._invert_world_points()
            self._eval_points_cache_valid = True
        return self._cached_eval_points

    @property
    def shape(self):
        return self._progressive_growing.get_shape()

    def _align(self):
        # No need to align points; we do it live in
        # `_update_aligned_surface_and_normals`.
        self._aligned_surface_cache_valid = False

    def _get_alignment(self):
        if self.state is AlignmentState.ON_GROUND:
            return None
        elif self.state is AlignmentState.ALIGNED:
            return self.alignment
        else:
            raise ValueError(f'unknown state {self.state}')

    def _update_aligned_surface_and_normals(self):
        needs_unaligned_update = self._needs_unaligned_surface_update()
        needs_aligned_update = self._needs_aligned_surface_update()
        needs_update = needs_unaligned_update or needs_aligned_update
        if not needs_update:
            return

        alignment = self._get_alignment()

        # Here we don't need to worry about the alignment yet.
        if needs_unaligned_update:
            eval_points = self.eval_points
            ctrl_points, ctrl_weights, knots_x, knots_y = \
                self._progressive_growing.select()

            surface_points, normals = _calc_normals_and_surface(
                eval_points,
                self.degree_x,
                self.degree_y,
                ctrl_points,
                ctrl_weights,
                knots_x,
                knots_y,
            )

            self._cached_discrete_points_unaligned = surface_points
            self._cached_normals_unaligned = normals
            self._orig_surface_cache_valid = True

        if alignment is None:
            self._cached_discrete_points = \
                self._cached_discrete_points_unaligned
            self._cached_normals = self._cached_normals_unaligned
            return

        if not needs_update:
            self._cached_discrete_points = self._cached_discrete_points_aligned
            self._cached_normals = self._cached_normals_aligned
            return

        hel_rotated = heliostat_models.rotate(
            self._cached_discrete_points_unaligned, alignment, clockwise=True)
        # Place in field
        hel_rotated = hel_rotated + self.position_on_field

        normal_vectors_rotated = heliostat_models.rotate(
            self._cached_normals_unaligned, alignment, clockwise=True)
        normal_vectors_rotated = (
            normal_vectors_rotated
            / th.linalg.norm(normal_vectors_rotated, dim=-1).unsqueeze(-1)
        )

        self._cached_discrete_points_aligned = hel_rotated
        self._cached_normals_aligned = normal_vectors_rotated
        self._cached_discrete_points = self._cached_discrete_points_aligned
        self._cached_normals = self._cached_normals_aligned
        self._aligned_surface_cache_valid = True

    @property
    def discrete_points(self):
        self._update_aligned_surface_and_normals()
        discrete_points = self._cached_discrete_points

        if self.state is AlignmentState.ON_GROUND:
            self._discrete_points_orig = discrete_points
            return self._discrete_points_orig
        elif self.state is AlignmentState.ALIGNED:
            self._discrete_points_aligned = discrete_points
            return self._discrete_points_aligned
        else:
            raise ValueError(f'unknown state {self.state}')

    @property
    def normals(self):
        self._update_aligned_surface_and_normals()
        normals = self._cached_normals

        if self.state is AlignmentState.ON_GROUND:
            self._normals_orig = normals
            return self._normals_orig
        elif self.state is AlignmentState.ALIGNED:
            self._normals_aligned = normals
            return self._normals_aligned
        else:
            raise ValueError(f'unknown state {self.state}')

    @th.no_grad()
    def step(self, verbose=False):
        self.invalidate_control_points_caches()

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
    ):
        if config is None:
            config = data['config']
        if nurbs_config is None:
            nurbs_config = data['nurbs_config']

        self = cls(config, nurbs_config, device)
        self._from_dict(data, restore_strictly)
        return self

    def _from_dict(self, data, restore_strictly):
        # Keep normals from standard initialization here.
        normals_orig = self._normals_orig
        super()._from_dict(data, restore_strictly)
        self._normals_orig = normals_orig

        self.degree_x = data['degree_x']
        self.degree_y = data['degree_y']
        ctrl_points = data['control_points']
        ctrl_points.requires_grad_(False)
        self.ctrl_points_xy = ctrl_points[:, :, :-1]
        self.ctrl_points_z = ctrl_points[:, :, -1:]
        self.ctrl_weights = data['control_point_weights']
        self.knots_x = data['knots_x']
        self.knots_y = data['knots_y']

        self._progressive_growing.set_step(data['_progressive_growing_step'])

        if restore_strictly:
            self._cached_eval_points = data['evaluation_points']
            self._orig_world_points = data['original_world_points']
            self._normals_orig = data['_heliostat_normals']
        else:
            self.initialize_eval_points()
