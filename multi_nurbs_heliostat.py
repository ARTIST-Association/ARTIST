import functools

import torch as th

import heliostat_models
from heliostat_models import AlignedHeliostat, Heliostat
from nurbs_heliostat import (
    AbstractNURBSHeliostat,
    AlignedNURBSHeliostat,
    NURBSHeliostat,
)


def _with_outer_list(values):
    if isinstance(values[0], list):
        return values
    return [values]


def _indices_between(points, from_, to):
    indices = (
        (from_ <= points) & (points < to)
    ).all(dim=-1)
    return indices


class MultiNURBSHeliostat(AbstractNURBSHeliostat, Heliostat):
    def __init__(
            self,
            heliostat_config,
            nurbs_config,
            device,
            setup_params=True,
            receiver_center=None,
    ):
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
            self._receiver_center = th.tensor(
                receiver_center,
                dtype=self.position_on_field.dtype,
                device=device,
            )
        else:
            self._receiver_center = None

        self.facets = self._create_facets(
            heliostat_config, nurbs_config, setup_params=setup_params)

        self._normals_ideal = th.cat(
            [facet._normals_ideal for facet in self.facets])

    @staticmethod
    def angle(a, b):
        return th.acos(
            th.dot(a, b)
            / (th.linalg.norm(a) * th.linalg.norm(b))
        )

    @staticmethod
    def rot_x_mat(angle, dtype, device):
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
    def rot_y_mat(angle, dtype, device):
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
    def rot_z_mat(angle, dtype, device):
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
            position_on_field,
            discrete_points,
            normals=None,
            normals_ideal=None,
    ):
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

        if normals is None:
            return hel_rotated

        normals_rotated = look_at_receiver(normals)
        normals_rotated = (
            normals_rotated
            / th.linalg.norm(normals_rotated, dim=-1).unsqueeze(-1)
        )

        normals_ideal_rotated = look_at_receiver(normals_ideal)
        normals_ideal_rotated = (
            normals_ideal_rotated
            / th.linalg.norm(normals_ideal_rotated, dim=-1).unsqueeze(-1)
        )
        return hel_rotated, normals_rotated, normals_ideal_rotated

    def _set_facet_points(self, facet, position, span_x, span_y):
        from_xyz = position - span_y - span_x
        to_xyz = position + span_y + span_x
        # We ignore the z-axis here.
        indices = _indices_between(
            self._discrete_points[:, :-1],
            from_xyz[:-1],
            to_xyz[:-1],
        )

        facet_discrete_points = self._discrete_points[indices] - position
        facet_normals = self._normals[indices]
        facet_normals_ideal = self._normals_ideal[indices]

        if (
                self.nurbs_cfg.FACETS.CANTING.ENABLED
                and not self.nurbs_cfg.FACETS.CANTING.ACTIVE
        ):
            (
                facet_discrete_points,
                facet_normals,
                facet_normals_ideal,
            ) = self._apply_canting(
                position,
                facet_discrete_points,
                facet_normals,
                facet_normals_ideal,
            )

        facet._discrete_points = facet_discrete_points
        facet._orig_world_points = facet._discrete_points.clone()
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

    @staticmethod
    def _facet_heliostat_config(heliostat_config, position):
        heliostat_config = heliostat_config.clone()
        heliostat_config.defrost()
        heliostat_config.POSITION_ON_FIELD = position.tolist()
        return heliostat_config

    @staticmethod
    def _facet_nurbs_config(nurbs_config, span_x, span_y):
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
            facet,
            position,
            span_x,
            span_y,
            orig_nurbs_config,
            nurbs_config,
    ):
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
                self._apply_canting(
                    position,
                    facet.ctrl_points.reshape(-1, facet.ctrl_points.shape[-1]),
                ).reshape(facet.ctrl_points.shape),
            )

        facet.initialize_eval_points()

    def _create_facet(
            self,
            position,
            span_x,
            span_y,
            heliostat_config,
            nurbs_config,
            setup_params,
    ):
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
    def _broadcast_spans(spans, to_length):
        if len(spans) == to_length:
            return spans

        assert len(spans) == 1, (
            'will only broadcast spans of length 1. If you did not intend '
            'to broadcast, make sure there is the same amount of facet '
            'positions and spans.'
        )
        return spans * to_length

    def _create_facets(self, heliostat_config, nurbs_config, setup_params):
        positions = _with_outer_list(self.nurbs_cfg.FACETS.POSITIONS)
        spans_x = _with_outer_list(self.nurbs_cfg.FACETS.SPANS_X)
        spans_x = self._broadcast_spans(spans_x, len(positions))
        spans_y = _with_outer_list(self.nurbs_cfg.FACETS.SPANS_Y)
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

    def __len__(self):
        return sum(len(facet) for facet in self.facets)

    def setup_params(self):
        for facet in self.facets:
            facet.setup_params()

    def get_params(self):
        return [
            param
            for facet in self.facets
            for param in facet.get_params()
        ]

    def align(self, sun_direction, receiver_center):
        return AlignedMultiNURBSHeliostat(self, sun_direction, receiver_center)

    def _calc_normals_and_surface(self, reposition=True):
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

    def discrete_points_and_normals(self, reposition=True):
        discrete_points, normals = self._calc_normals_and_surface(
            reposition=reposition)
        return discrete_points, normals

    @property
    @functools.lru_cache()
    def dict_keys(self):
        """All keys we assume in the dictionary returned by `_to_dict`."""
        keys = super().dict_keys
        keys = keys.union({
            'nurbs_config',
            'facets',

            'receiver_center',
        })
        return keys

    @functools.lru_cache()
    def _fixed_dict(self):
        data = super()._fixed_dict()
        data['nurbs_config'] = self.nurbs_cfg
        data['receiver_center'] = self._receiver_center
        return data

    def _to_dict(self):
        data = super()._to_dict()
        data['facets'] = [
            facet._to_dict()
            for facet in self.facets
        ]
        return data

    @classmethod
    def from_dict(
            cls,
            data,
            device,
            config=None,
            nurbs_config=None,
            receiver_center=None,
            # Wether to disregard what standard initialization did and
            # load all data we have.
            restore_strictly=False,
            setup_params=True,
    ):
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

        if restore_strictly:
            self.facets = self._create_facets(
                config,
                nurbs_config,
                setup_params=setup_params,
            )

            for (facet, facet_data) in zip(self.facets, data['facets']):
                facet._from_dict(facet_data, restore_strictly)

        return self

    def _from_dict(self, data, restore_strictly):
        super()._from_dict(data, restore_strictly)

        if restore_strictly:
            self._receiver_center = data['receiver_center']


class AlignedMultiNURBSHeliostat(AlignedNURBSHeliostat):
    def __init__(self, heliostat, sun_direction, receiver_center):
        assert isinstance(heliostat, MultiNURBSHeliostat), \
            'can only align multi-NURBS heliostat'
        AlignedHeliostat.__init__(
            self,
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

    def _calc_normals_and_surface(self):
        if (
                self._heliostat.nurbs_cfg.FACETS.CANTING.ENABLED
                and self._heliostat.nurbs_cfg.FACETS.CANTING.ACTIVE
        ):
            hel_rotated, normal_vectors_rotated = \
                MultiNURBSHeliostat.discrete_points_and_normals(
                    self, reposition=False)
            hel_rotated = hel_rotated + self._heliostat.position_on_field
        else:
            surface_points, normals = \
                MultiNURBSHeliostat.discrete_points_and_normals(
                    self._heliostat)

            hel_rotated = heliostat_models.rotate_multi_nurbs(
                surface_points, self.alignment)
            # Place in field
            hel_rotated = hel_rotated + self._heliostat.position_on_field

            normal_vectors_rotated = heliostat_models.rotate_multi_nurbs(
                normals, self.alignment)
            normal_vectors_rotated = (
                normal_vectors_rotated
                / th.linalg.norm(normal_vectors_rotated, dim=-1).unsqueeze(-1)
            )

        return hel_rotated, normal_vectors_rotated
