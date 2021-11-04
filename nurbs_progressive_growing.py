import torch as th

import utils


class ProgressiveGrowing:
    def __init__(self, heliostat):
        self.heliostat = heliostat
        self.cfg = self.heliostat.nurbs_cfg.GROWING

        self._interval = self.cfg.INTERVAL
        self._step = 0

        if self._no_progressive_growing():
            self.row_indices = None
            self.col_indices = None
            return

        self.device = self.heliostat.device

        self.row_indices = self._calc_start_indices(
            self.cfg.START_ROWS,
            self.heliostat.rows,
            self.heliostat.degree_x,
        )
        self.col_indices = self._calc_start_indices(
            self.cfg.START_COLS,
            self.heliostat.cols,
            self.heliostat.degree_y,
        )

    def get_step(self):
        return self._step

    def set_step(self, step):
        self._step = step

    def _no_progressive_growing(self):
        return self._interval < 1

    def _calc_start_indices(self, start_size, final_size, degree):
        if start_size < 1:
            start_size = degree + 1

        assert start_size > degree, (
            f'NURBS growing start size is too small; '
            f'must be at least {degree + 1}'
        )
        assert final_size > degree, (
            f'NURBS growing final size is too small; '
            f'must be at least {degree + 1}'
        )

        indices = th.linspace(
            0,
            final_size - 1,
            start_size,
            device=self.device,
        )
        indices = utils.round_positionally(indices)
        return indices

    def _done_growing(self):
        return self.row_indices is None and self.col_indices is None

    def _grow_indices(self, indices, final_size):
        if indices is None or len(indices) == final_size:
            return None
        elif len(indices) > final_size:
            raise ValueError('overshot goal size')

        between_indices = indices[:-1] + (indices[1:] - indices[:-1]) / 2
        between_indices = utils.round_positionally(between_indices)

        grown_indices = th.cat([indices, between_indices])
        grown_indices = grown_indices.sort()[0]
        grown_indices = grown_indices.unique_consecutive()

        if len(grown_indices) == final_size:
            return None
        return grown_indices

    def _find_new_indices(self, old_indices, curr_indices, final_size):
        dtype = old_indices.dtype

        if curr_indices is None:
            curr_indices = th.arange(
                final_size,
                device=self.device,
                dtype=dtype,
            )

        old_indices_set = set(map(int, old_indices))
        curr_indices_set = set(map(int, curr_indices))
        new_indices = curr_indices_set.difference(old_indices_set)
        if not new_indices:
            # No new indices; quit early.
            return None

        new_indices = th.tensor(
            list(new_indices),
            device=self.device,
            dtype=dtype,
        )
        new_indices = new_indices.sort()[0]
        return new_indices

    def _calc_grown_control_points_per_dim(
            self,
            old_ctrl_points,
            k,
    ):
        old_ctrl_points_shape = old_ctrl_points.shape
        world_points = self.heliostat.discrete_points

        old_ctrl_points = old_ctrl_points.reshape(
            -1, self.heliostat.ctrl_points.shape[-1])

        new_control_points = utils.calc_knn_averages(
            old_ctrl_points,
            world_points,
            k,
        )
        new_control_points = new_control_points.reshape(old_ctrl_points_shape)

        return new_control_points

    def _set_grown_control_points(self, old_row_indices, old_col_indices, k=4):
        new_row_indices = self._find_new_indices(
            old_row_indices,
            self.row_indices,
            self.heliostat.rows,
        )
        if new_row_indices is not None:
            new_row_control_points = self._calc_grown_control_points_per_dim(
                self.heliostat.ctrl_points[new_row_indices, :], k)

            if not self.heliostat.nurbs_cfg.OPTIMIZE_Z_ONLY:
                self.heliostat.ctrl_points_xy[new_row_indices, :] = \
                    new_row_control_points[..., :-1]
            self.heliostat.ctrl_points_z[new_row_indices, :] = \
                new_row_control_points[..., -1:]

        new_col_indices = self._find_new_indices(
            old_col_indices,
            self.col_indices,
            self.heliostat.cols,
        )
        if new_col_indices is not None:
            new_col_control_points = self._calc_grown_control_points_per_dim(
                self.heliostat.ctrl_points[:, new_col_indices], k)

            if not self.heliostat.nurbs_cfg.OPTIMIZE_Z_ONLY:
                self.heliostat.ctrl_points_xy[:, new_col_indices] = \
                    new_col_control_points[..., :-1]
            self.heliostat.ctrl_points_z[:, new_col_indices] = \
                new_col_control_points[..., -1:]

    def _grow_nurbs(self, verbose=False):
        already_done = self._done_growing()
        if already_done:
            return

        old_row_indices = self.row_indices
        old_col_indices = self.col_indices

        self.row_indices = self._grow_indices(
            self.row_indices, self.heliostat.rows)
        self.col_indices = self._grow_indices(
            self.col_indices, self.heliostat.cols)

        self._set_grown_control_points(old_row_indices, old_col_indices, k=4)

        if verbose:
            if not already_done and self._done_growing():
                print('Finished growing NURBS.')
            else:
                print(
                    f'Grew NURBS to '
                    f'{len(self.row_indices)}'
                    f'×{len(self.col_indices)}.'
                )

    def select(self):
        degree_x = self.heliostat.degree_x
        degree_y = self.heliostat.degree_y
        ctrl_points = self.heliostat.ctrl_points
        ctrl_weights = self.heliostat.ctrl_weights
        knots_x = self.heliostat.knots_x
        knots_y = self.heliostat.knots_y
        indices_x = self.row_indices
        indices_y = self.col_indices

        if indices_x is not None:
            ctrl_points = ctrl_points[indices_x, :]
            ctrl_weights = ctrl_weights[indices_x, :]

            # FIXME can we make this more dynamic, basing it on the
            #       available knot points?
            knots_x = th.empty(
                len(indices_x) + degree_x + 1,
                device=self.device,
                dtype=knots_x.dtype,
            )
            utils.initialize_spline_knots_(knots_x, degree_x)

        if indices_y is not None:
            ctrl_points = ctrl_points[:, indices_y]
            ctrl_weights = ctrl_weights[:, indices_y]

            # FIXME can we make this more dynamic, basing it on the
            #       available knot points?
            knots_y = th.empty(
                len(indices_y) + degree_y + 1,
                device=self.device,
                dtype=knots_y.dtype,
            )
            utils.initialize_spline_knots_(knots_y, degree_y)

        return ctrl_points, ctrl_weights, knots_x, knots_y

    def get_shape(self):
        rows = self.row_indices
        if rows is None:
            rows = self.rows
        else:
            rows = len(rows)

        cols = self.col_indices
        if cols is None:
            cols = self.cols
        else:
            cols = len(cols)

        return (rows, cols)

    def step(self, verbose):
        self._step += 1
        if (
                self._no_progressive_growing()
                or (
                    (
                        self._step
                        % self._interval
                    )
                    != 0
                )
        ):
            return

        self._grow_nurbs(verbose=verbose)
