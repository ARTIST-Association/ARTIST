import functools

import torch as th

import heliostat_models
import nurbs
import utils


class NURBSHeliostat(heliostat_models.Heliostat):
    def __init__(self, heliostat_config, nurbs_config, device):
        super().__init__(heliostat_config, device)
        self.nurbs_cfg = nurbs_config

        self.fix_spline_ctrl_weights = nurbs_config.FIX_SPLINE_CTRL_WEIGHTS

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

        utils.initialize_spline_knots(
            self.knots_x, self.knots_y, self.degree_x, self.degree_y)
        self.ctrl_weights[:] = 1

        if nurbs_config.SET_UP_WITH_KNOWLEDGE:
            utils.initialize_spline_ctrl_points(
                ctrl_points,
                self.position_on_field,
                self.rows,
                self.cols,
                heliostat_config.IDEAL.WIDTH,
                heliostat_config.IDEAL.HEIGHT,
            )
            self.eval_points = utils.initialize_spline_eval_points_perfectly(
                self._discrete_points_orig,
                self.degree_x,
                self.degree_y,
                ctrl_points,
                self.ctrl_weights,
                self.knots_x,
                self.knots_y,
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
            self.eval_points = utils.initialize_spline_eval_points(
                self.rows, self.cols, self.device)

        self.ctrl_points_xy = ctrl_points[:, :-1]
        self.ctrl_points_z = ctrl_points[:, -1:]

    def setup_params(self):
        self.ctrl_points_z.requires_grad_(True)
        if not self.fix_spline_ctrl_weights:
            self.ctrl_weights.requires_grad_(True)

        # knots_x.requires_grad_(True)
        # knots_y.requires_grad_(True)

        opt_params = [self.ctrl_points_z]
        if not self.fix_spline_ctrl_weights:
            opt_params.append(self.ctrl_weights)

        if self.knots_x.requires_grad:
            opt_params.append(self.knots_x)
        if self.knots_y.requires_grad:
            opt_params.append(self.knots_y)

        return opt_params

    @property
    def ctrl_points(self):
        return th.hstack([self.ctrl_points_xy, self.ctrl_points_z])

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
            / normal_vectors_rotated.norm(dim=-1).unsqueeze(-1)
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

    def to_dict(self):
        return {
            'degree_x': self.degree_x,
            'degree_y': self.degree_y,
            'control_points': self.ctrl_points,
            'control_point_weights': self.ctrl_weights,
            'knots_x': self.knots_x,
            'knots_y': self.knots_y,
        }
