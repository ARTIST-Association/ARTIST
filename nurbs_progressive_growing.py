import torch as th


class ProgressiveGrowing:
    def __init__(self, heliostat):
        self.heliostat = heliostat

        self._interval = self.heliostat.nurbs_cfg.PROGRESSIVE_GROWING_INTERVAL
        self._step = 0

        if self._no_progressive_growing():
            self.row_indices = None
            self.col_indices = None
            return

        self.device = self.heliostat.device

        self.row_indices = \
            self._calc_start_indices(
                self.heliostat.rows,
                self.heliostat.degree_x,
                self.heliostat.device,
            )
        self.col_indices = \
            self._calc_start_indices(
                self.heliostat.cols,
                self.heliostat.degree_y,
                self.heliostat.device,
            )

    def get_step(self):
        return self._step

    def set_step(self, step):
        self._step = step

    def _no_progressive_growing(self):
        return self._interval < 1

    @staticmethod
    def _calc_start_indices(final_size, degree, device):
        assert final_size > degree, \
            'the NURBS does not have enough control points'
        return th.linspace(0, final_size - 1, degree + 1).round().long()

    def _done_growing(self):
        return self.row_indices is None and self.col_indices is None

    def _grow_indices(self, indices, final_size):
        if indices is None or len(indices) == final_size:
            return None
        elif len(indices) > final_size:
            raise ValueError('overshot goal size')

        between_indices = indices[:-1] + (indices[1:] - indices[:-1]) / 2
        between_indices_middle = th.tensor(
            len(between_indices) / 2,
            device=self.device,
        ).round().long()

        # Round lower values down, upper values up.
        # This makes the indices become mirrored around the middle
        # index.
        between_indices = th.cat([
            between_indices[:between_indices_middle].long(),
            between_indices[between_indices_middle:].round().long(),
        ])

        grown_indices = th.cat([indices, between_indices])
        grown_indices = grown_indices.sort()[0]
        grown_indices = grown_indices.unique_consecutive()

        if len(grown_indices) == final_size:
            return None
        return grown_indices

    @staticmethod
    def _horizontal_dist(a, b, ord=2):
        return th.linalg.norm(b[..., :-1] - a[..., :-1], dim=-1, ord=ord)

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
        distances = self._horizontal_dist(
            old_ctrl_points.unsqueeze(1),
            world_points.unsqueeze(0),
        )
        closest_indices = distances.argsort(dim=-1)
        closest_indices = closest_indices[..., :k]

        new_control_points = world_points[closest_indices].mean(dim=-2)
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
                print('finished growing NURBS')
            else:
                print(
                    f'grew NURBS to '
                    f'{len(self.row_indices)}'
                    f'×{len(self.col_indices)}'
                )

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
