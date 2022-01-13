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
    ):
        super().__init__(heliostat_config, device, setup_params=False)
        self.nurbs_cfg = nurbs_config
        if not self.nurbs_cfg.is_frozen():
            self.nurbs_cfg = self.nurbs_cfg.clone()
            self.nurbs_cfg.freeze()

        self.facets = self._create_facets(
            heliostat_config, nurbs_config, setup_params=setup_params)

    def _set_facet_points(self, facet, position, span_x, span_y):
        from_xyz = position - span_y - span_x
        to_xyz = position + span_y + span_x
        # We ignore the z-axis here.
        indices = _indices_between(
            self._discrete_points[:, :-1],
            from_xyz[:-1],
            to_xyz[:-1],
        )

        facet._discrete_points = (
            self._discrete_points[indices]
            - self.position_on_field
        )
        facet._orig_world_points = facet._discrete_points.clone()
        facet._normals = self._normals[indices]
        facet._normals_ideal = self._normals_ideal[indices]

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

    def align(self, sun_origin, receiver_center):
        return AlignedMultiNURBSHeliostat(self, sun_origin, receiver_center)

    def _calc_normals_and_surface(self):
        total_size = len(self)
        surface_points = th.empty((total_size, 3), device=self.device)
        normals = th.empty((total_size, 3), device=self.device)

        i = 0
        for facet in self.facets:
            curr_surface_points, curr_normals = \
                facet.discrete_points_and_normals()
            offset = len(curr_surface_points)

            surface_points[i:i + offset] = curr_surface_points
            normals[i:i + offset] = curr_normals
            i += offset

        return surface_points, normals

    @property
    @functools.lru_cache()
    def dict_keys(self):
        """All keys we assume in the dictionary returned by `_to_dict`."""
        keys = super().dict_keys
        keys = keys.union({
            'nurbs_config',
            'facets',
        })
        return keys

    @functools.lru_cache()
    def _fixed_dict(self):
        data = super()._fixed_dict()
        data['nurbs_config'] = self.nurbs_cfg
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

        if restore_strictly:
            self.facets = self._create_facets(
                config,
                nurbs_config,
                setup_params=setup_params,
            )

            for (facet, facet_data) in zip(self.facets, data['facets']):
                facet._from_dict(facet_data, restore_strictly)

        return self


class AlignedMultiNURBSHeliostat(AlignedNURBSHeliostat):
    def __init__(self, heliostat, sun_origin, receiver_center):
        assert isinstance(heliostat, MultiNURBSHeliostat), \
            'can only align multi-NURBS heliostat'
        AlignedHeliostat.__init__(
            self, heliostat, sun_origin, receiver_center, align_points=False)
        if self._heliostat.nurbs_cfg.FACETS.USE_CANTING:
            self.facets = [
                facet.align(sun_origin, receiver_center)
                for facet in self._heliostat.facets
            ]
            self.device = self._heliostat.device

    def _calc_normals_and_surface(self):
        if self._heliostat.nurbs_cfg.FACETS.USE_CANTING:
            hel_rotated, normal_vectors_rotated = \
                MultiNURBSHeliostat._calc_normals_and_surface(self)
            hel_rotated = hel_rotated + self._heliostat.position_on_field
        else:
            surface_points, normals = \
                MultiNURBSHeliostat._calc_normals_and_surface(self._heliostat)

            hel_rotated = heliostat_models.rotate(
                surface_points, self.alignment, clockwise=True)
            # Place in field
            hel_rotated = hel_rotated + self._heliostat.position_on_field

            normal_vectors_rotated = heliostat_models.rotate(
                normals, self.alignment, clockwise=True)
            normal_vectors_rotated = (
                normal_vectors_rotated
                / th.linalg.norm(normal_vectors_rotated, dim=-1).unsqueeze(-1)
            )

        return hel_rotated, normal_vectors_rotated
