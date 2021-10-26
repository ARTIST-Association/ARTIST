import functools

import torch as th

import heliostat_models
import nurbs
import utils


class NURBSHeliostat(heliostat_models.Heliostat):
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
                # FIXME Get these for the current heliostat!
                heliostat_config.IDEAL.WIDTH,
                heliostat_config.IDEAL.HEIGHT,
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
        self.ctrl_points_xy = ctrl_points[:, :, :-1]
        self.ctrl_points_z = ctrl_points[:, :, -1:]

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
            self.recalc_eval_points = False
            self._eval_points = utils.initialize_spline_eval_points(
                self.rows, self.cols, self.device)

    def setup_params(self):
        self.ctrl_points_z.requires_grad_(True)
        if not self.nurbs_cfg.OPTIMIZE_Z_ONLY:
            self.ctrl_points_xy.requires_grad_(True)
        if not self.fix_spline_ctrl_weights:
            self.ctrl_weights.requires_grad_(True)

        if not self.fix_spline_knots:
            self.knots_x.requires_grad_(True)
            self.knots_y.requires_grad_(True)

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
        return th.cat([self.ctrl_points_xy, self.ctrl_points_z], dim=-1)

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _invert_world_points(
            eval_points,
            degree_x,
            degree_y,
            ctrl_points,
            ctrl_weights,
            knots_x,
            knots_y,
    ):
        return utils.initialize_spline_eval_points_perfectly(
            eval_points,
            degree_x,
            degree_y,
            ctrl_points,
            ctrl_weights,
            knots_x,
            knots_y,
        )

    @property
    def eval_points(self):
        if self.recalc_eval_points:
            self._eval_points = self._invert_world_points(
                self._orig_world_points,
                self.degree_x,
                self.degree_y,
                self.ctrl_points,
                self.ctrl_weights,
                self.knots_x,
                self.knots_y,
            )
        return self._eval_points

    def _get_alignment(self):
        if self.state == 'OnGround':
            return None
        elif self.state == 'Aligned':
            return self.alignment
        else:
            raise ValueError(f'unknown state {self.state}')

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _get_aligned_surface_and_normals(
            eval_points,
            degree_x,
            degree_y,
            ctrl_points,
            ctrl_weights,
            knots_x,
            knots_y,
            alignment,
            position_on_field,
    ):
        surface_points, normals = nurbs.calc_normals_and_surface_slow(
            eval_points[:, 0],
            eval_points[:, 1],
            degree_x,
            degree_y,
            ctrl_points,
            ctrl_weights,
            knots_x,
            knots_y,
        )
        if alignment is None:
            return surface_points, normals

        hel_rotated = heliostat_models.rotate(
            surface_points, alignment, clockwise=True)
        # Place in field
        hel_rotated = hel_rotated + position_on_field

        normal_vectors_rotated = heliostat_models.rotate(
            normals, alignment, clockwise=True)
        normal_vectors_rotated = (
            normal_vectors_rotated
            / th.linalg.norm(normal_vectors_rotated, dim=-1).unsqueeze(-1)
        )
        return hel_rotated, normal_vectors_rotated

    @property
    def discrete_points(self):
        alignment = self._get_alignment()
        discrete_points, _ = self._get_aligned_surface_and_normals(
            self.eval_points,
            self.degree_x,
            self.degree_y,
            self.ctrl_points,
            self.ctrl_weights,
            self.knots_x,
            self.knots_y,
            alignment,
            self.position_on_field,
        )

        if self.state == 'OnGround':
            self._discrete_points_orig = discrete_points
            return self._discrete_points_orig
        elif self.state == 'Aligned':
            self._discrete_points_aligned = discrete_points
            return self._discrete_points_aligned
        else:
            raise ValueError(f'unknown state {self.state}')

    @property
    def normals(self):
        alignment = self._get_alignment()
        _, normals = self._get_aligned_surface_and_normals(
            self.eval_points,
            self.degree_x,
            self.degree_y,
            self.ctrl_points,
            self.ctrl_weights,
            self.knots_x,
            self.knots_y,
            alignment,
            self.position_on_field,
        )

        if self.state == 'OnGround':
            self._normals_orig = normals
            return self._normals_orig
        elif self.state == 'Aligned':
            self._normals_aligned = normals
            return self._normals_aligned
        else:
            raise ValueError(f'unknown state {self.state}')

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

        ctrl_points = self.ctrl_points.clone()
        data.update({
            'control_points': self.ctrl_points.clone(),

            'original_world_points': self._orig_world_points,
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

        if restore_strictly:
            self._eval_points = data['evaluation_points']
            self._orig_world_points = data['original_world_points']
            self._normals_orig = data['_heliostat_normals']
        else:
            self.initialize_eval_points()
