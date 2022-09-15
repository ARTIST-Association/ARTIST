import functools
from typing import (
    Any,
    cast,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch
import torch as th
from yacs.config import CfgNode

import canting
from canting import CantingAlgorithm
import facets
import heliostat_models
from heliostat_models import AlignedHeliostat, Heliostat
from nurbs_heliostat import (
    AbstractNURBSHeliostat,
    AlignedNURBSHeliostat,
    NURBSHeliostat,
)

C = TypeVar('C', bound='MultiNURBSHeliostat')


class NURBSFacets(facets.AbstractFacets):
    def __init__(
            self,
            heliostat: 'MultiNURBSHeliostat',
            nurbs_heliostats: List[AbstractNURBSHeliostat],
            cant_rots: torch.Tensor,
    ):
        self._facets = nurbs_heliostats
        self.positions = heliostat.facets.positions
        self.spans_n = heliostat.facets.spans_n
        self.spans_e = heliostat.facets.spans_e

        self.offsets = self._make_offsets(
            self._facets)  # type: ignore[arg-type]
        self._canting_algo = heliostat.canting_algo
        self.cant_rots = cant_rots

    def __len__(self) -> int:
        return sum(len(facet) for facet in self._facets)

    def __iter__(self) -> Iterator[NURBSHeliostat]:
        assert self._facets and isinstance(self._facets[0], NURBSHeliostat)
        return iter(cast(List[NURBSHeliostat], self._facets))

    @property
    def raw_discrete_points(self) -> List[torch.Tensor]:
        return super().raw_discrete_points

    @raw_discrete_points.setter
    def raw_discrete_points(
            self,
            new_discrete_points: List[torch.Tensor],
    ) -> None:
        raise ValueError('NURBS facet does not allow setting discrete points')

    @property
    def _discrete_points(self) -> List[torch.Tensor]:  # type: ignore[override]
        discrete_points = [facet.discrete_points for facet in self._facets]
        return discrete_points

    @property
    def _discrete_points_ideal(  # type: ignore[override]
            self,
    ) -> List[torch.Tensor]:
        return [
            facets.make_unfacetted(facet.get_raw_discrete_points_ideal())
            for facet in self._facets
        ]

    @property
    def raw_normals(self) -> List[torch.Tensor]:
        return super().raw_normals

    @raw_normals.setter
    def raw_normals(
            self,
            new_normals: List[torch.Tensor],
    ) -> None:
        raise ValueError('NURBS facet does not allow setting normals')

    @property
    def _normals(self) -> List[torch.Tensor]:  # type: ignore[override]
        return [facet.normals for facet in self._facets]

    @property
    def _normals_ideal(self) -> List[torch.Tensor]:  # type: ignore[override]
        return [facet._normals_ideal for facet in self._facets]

    def align_discrete_points_and_normals(
            self,
            reposition: bool = True,
            force_canting: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        discrete_points_and_normals = [
            facet.discrete_points_and_normals()
            for facet in self._facets
        ]
        discrete_points = [tup[0] for tup in discrete_points_and_normals]
        normals = [tup[1] for tup in discrete_points_and_normals]
        del discrete_points_and_normals

        merged_discrete_points = facets.cant_and_merge_facet_vectors(
            self,
            discrete_points,
            reposition=reposition,
            force_canting=force_canting,
        )
        merged_normals = facets.cant_and_merge_facet_vectors(
            self,
            normals,
            reposition=False,
            force_canting=force_canting,
        )
        return merged_discrete_points, merged_normals


class MultiNURBSHeliostat(AbstractNURBSHeliostat, Heliostat):
    # Map from optimizable name to per-facet optimizable name so that
    # individual facet optimizables combine to the complete named
    # optimizable.
    _FACET_OPTIMIZABLES = {
        'surface': 'surface',
    }

    def __init__(
            self,
            heliostat_config: CfgNode,
            nurbs_config: CfgNode,
            device: th.device,
            setup_params: bool = True,
            receiver_center: Union[torch.Tensor, List[float], None] = None,
    ) -> None:
        super().__init__(
            heliostat_config,
            device,
            setup_params=False,
            receiver_center=receiver_center,
        )
        self.nurbs_cfg = nurbs_config
        if not self.nurbs_cfg.is_frozen():
            self.nurbs_cfg = self.nurbs_cfg.clone()
            self.nurbs_cfg.freeze()

        cfg_width: float = self.nurbs_cfg.WIDTH
        if isinstance(cfg_width, str):
            if cfg_width != 'inherit':
                raise ValueError(f'unknown width config "{cfg_width}"')
        else:
            self.width = cfg_width

        cfg_height: float = self.nurbs_cfg.HEIGHT
        if isinstance(cfg_height, str):
            if cfg_height != 'inherit':
                raise ValueError(f'unknown height config "{cfg_height}"')
        else:
            self.height = cfg_height

        cfg_position_on_field: Union[List[float], str] = \
            self.nurbs_cfg.POSITION_ON_FIELD
        if isinstance(cfg_position_on_field, str):
            if cfg_position_on_field != 'inherit':
                raise ValueError(
                    f'unknown position on field config '
                    f'"{cfg_position_on_field}"'
                )
        else:
            self.position_on_field = heliostat_models.get_position(
                self.nurbs_cfg,
                dtype=self.position_on_field.dtype,
                device=self.device,
            )

        cfg_aim_point: Union[List[float], str, None] = self.nurbs_cfg.AIM_POINT
        if isinstance(cfg_aim_point, str):
            if cfg_aim_point != 'inherit':
                raise ValueError(f'unknown aim point config "{cfg_aim_point}"')
            _, aim_point_cfg = self.select_heliostat_builder(self.cfg)
            maybe_aim_point: Optional[torch.Tensor] = self.aim_point
        else:
            if (
                    receiver_center is not None
                    and not isinstance(receiver_center, th.Tensor)
            ):
                receiver_center = th.tensor(
                    receiver_center,
                    dtype=self.position_on_field.dtype,
                    device=self.device,
                )
            aim_point_cfg = self.nurbs_cfg
            maybe_aim_point = receiver_center
        self.aim_point = self._get_aim_point(aim_point_cfg, maybe_aim_point)

        cfg_disturbance_angles: Union[List[float], str] = \
            self.nurbs_cfg.DISTURBANCE_ROT_ANGLES
        if isinstance(cfg_disturbance_angles, str):
            if cfg_disturbance_angles != 'inherit':
                raise ValueError(
                    f'unknown disturbance angles config '
                    f'"{cfg_disturbance_angles}"'
                )
        else:
            # Radians
            self.disturbance_angles = self._get_disturbance_angles(
                self.nurbs_cfg)

        self._inherit_canting()

        facets_and_rots = self._create_facets(
            self.cfg, self.nurbs_cfg)
        self.facets = NURBSFacets(
            self,
            [tup[0] for tup in facets_and_rots],
            th.stack([tup[1] for tup in facets_and_rots]),
        )

        if setup_params:
            self.setup_params()

    def _inherit_canting(self) -> None:
        old_canting_cfg = self._canting_cfg
        self._canting_cfg = self.nurbs_cfg.FACETS.CANTING.clone()
        self._canting_cfg.defrost()

        cfg_focus_point: Union[None, List[float], float, str] = \
            self._canting_cfg.FOCUS_POINT
        if isinstance(cfg_focus_point, str):
            if cfg_focus_point != 'inherit':
                raise ValueError(
                    f'unknown focus point config "{cfg_focus_point}"')
            self._canting_cfg.FOCUS_POINT = old_canting_cfg.FOCUS_POINT
        focus_point = canting.get_focus_point(
            self._canting_cfg,
            self.aim_point,
            self.cfg.IDEAL.NORMAL_VECS,
            dtype=self.position_on_field.dtype,
            device=self.device,
        )
        self._set_deconstructed_focus_point(focus_point)

        cfg_canting_algo: str = self._canting_cfg.ALGORITHM
        if cfg_canting_algo == 'inherit':
            self._canting_cfg.ALGORITHM = old_canting_cfg.ALGORITHM
        self.canting_algo = canting.get_algorithm(self._canting_cfg)

        self._canting_cfg.freeze()

    def _set_facet_points(
            self,
            facet: NURBSHeliostat,
            facet_index: int,
            position: torch.Tensor,
    ) -> torch.Tensor:
        assert not isinstance(self.facets, NURBSFacets)
        # We calculate everything anew again here because the canting
        # settings may have changed.
        # TODO Could probably be done smarter with less work. Because we
        #      already have most of this.
        facet_slice = slice(
            self.facets.offsets[facet_index],
            (
                self.facets.offsets[facet_index + 1]
                if facet_index + 1 < len(self.facets.offsets)
                else None
            ),
        )
        facet_discrete_points = self.facets.align_discrete_points(
            force_canting=True)[facet_slice]
        facet_discrete_points_ideal = self.facets.align_discrete_points_ideal(
            force_canting=True)[facet_slice]
        facet_normals = self.facets.align_normals(
            force_canting=True)[facet_slice]
        facet_normals_ideal = self.facets.align_normals_ideal(
            force_canting=True)[facet_slice]

        if (
                self.canting_enabled
                and self.canting_algo is not CantingAlgorithm.ACTIVE
        ):
            canting_params: Optional[Tuple[
                Optional[torch.Tensor],
                torch.Tensor,
            ]] = (self.focus_point, self.position_on_field)
        else:
            canting_params = None

        (
            facet_discrete_points,
            facet_discrete_points_ideal,
            facet_normals,
            facet_normals_ideal,
            cant_rot,
        ) = canting.decant_facet(
            position,
            facet_discrete_points,
            facet_discrete_points_ideal,
            facet_normals,
            facet_normals_ideal,
            self.cfg.IDEAL.NORMAL_VECS,
            canting_params,
        )

        # Re-center facet around zero.
        facet_discrete_points -= position
        facet_discrete_points_ideal -= position

        facet.facets = facets.Facets(
            facet,
            th.zeros_like(position).unsqueeze(0),
            facet.facets.spans_n,
            facet.facets.spans_e,
            [facet_discrete_points],
            [facet_discrete_points_ideal],
            [facet_normals],
            [facet_normals_ideal],
            th.eye(3).unsqueeze(0),
        )
        facet._orig_world_points = facet._discrete_points_ideal.clone()

        added_dims = facet.height + facet.width
        height_ratio = facet.height / added_dims
        width_ratio = facet.width / added_dims

        # FIXME Not perfectly accurate.
        # Only way to really do this is by comparing values for
        # equality over an axis.
        num_discrete_points = sum(
            len(points)
            for points in facet._discrete_points
        )
        if facet.h_rows is not None:
            facet.h_rows = int(th.pow(
                th.tensor(num_discrete_points),
                height_ratio,
            ))
        if facet.h_cols is not None:
            facet.h_cols = int(th.ceil(th.pow(
                th.tensor(num_discrete_points),
                width_ratio,
            )))

        # Handle non-rectangular facet points.
        if (
                facet.h_rows is not None and facet.h_cols is not None
                and facet.h_rows * facet.h_cols != num_discrete_points
        ):
            facet.h_rows = None
            facet.h_cols = None

        return cant_rot

    @staticmethod
    def _facet_optimizable_to_optimizable(
            facet_optimizable: str,
    ) -> Optional[str]:
        return next(
            (
                optimizable
                for (
                        optimizable,
                        comp_facet_optimizable,
                ) in MultiNURBSHeliostat._FACET_OPTIMIZABLES.items()
                if comp_facet_optimizable == facet_optimizable
            ),
            None,
        )

    def _facet_heliostat_config(
            self,
            heliostat_config: CfgNode,
            position: torch.Tensor,
            span_n: torch.Tensor,
            span_e: torch.Tensor,
    ) -> CfgNode:
        heliostat_config = heliostat_config.clone()
        heliostat_config.defrost()

        # We change the shape in order to speed up construction.
        # Later, we need to do adjust all loaded values to be the same
        # as the parent heliostat.
        heliostat_config.SHAPE = 'ideal'
        heliostat_config.IDEAL.ROWS = 2
        heliostat_config.IDEAL.COLS = 2

        heliostat_config.TO_OPTIMIZE = [
            self._FACET_OPTIMIZABLES[name]
            for name in self.get_to_optimize()
            if name in self._FACET_OPTIMIZABLES
        ]
        # We use `self.facets.positions` to position the heliostat's
        # values.
        heliostat_config.IDEAL.POSITION_ON_FIELD = [0.0, 0.0, 0.0]
        # Give any aim point so it doesn't complain.
        heliostat_config.IDEAL.AIM_POINT = [0.0, 0.0, 0.0]
        # We don't want to optimize the rotation for each facet, only
        # for the whole heliostat. So do not disturb facets.
        heliostat_config.IDEAL.DISTURBANCE_ROT_ANGLES = [0.0, 0.0, 0.0]
        # Even though we use `self.facets.positions` to position the
        # heliostat's values, we need this for correct initialization of
        # the single NURBS heliostat facet.
        heliostat_config.IDEAL.FACETS.POSITIONS = [position.tolist()]
        heliostat_config.IDEAL.FACETS.SPANS_N = [span_n.tolist()]
        heliostat_config.IDEAL.FACETS.SPANS_E = [span_e.tolist()]
        # Do not cant facet NURBS in their constructor; we do it
        # manually.
        heliostat_config.IDEAL.FACETS.CANTING.FOCUS_POINT = 0
        return heliostat_config

    @staticmethod
    def _facet_nurbs_config(
            nurbs_config: CfgNode,
            span_n: torch.Tensor,
            span_e: torch.Tensor,
    ) -> CfgNode:
        height = (th.linalg.norm(span_n) * 2).item()
        width = (th.linalg.norm(span_e) * 2).item()

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
            facet_index: int,
            position: torch.Tensor,
            span_n: torch.Tensor,
            span_e: torch.Tensor,
            orig_nurbs_config: CfgNode,
            nurbs_config: CfgNode,
    ) -> torch.Tensor:
        # "Load" values from parent heliostat.
        facet._discrete_points = self._discrete_points
        facet.set_raw_discrete_points_ideal(
            self.get_raw_discrete_points_ideal())
        facet._normals = self._normals
        facet.set_raw_normals_ideal(self.get_raw_normals_ideal())
        facet.params = self.params
        facet.h_rows = self.rows
        facet.h_cols = self.cols

        facet.height = nurbs_config.HEIGHT
        facet.width = nurbs_config.WIDTH
        # TODO initialize NURBS correctly
        # facet.position_on_field = self.position_on_field + position
        cant_rot = self._set_facet_points(facet, facet_index, position)

        facet.nurbs_cfg.defrost()
        facet.nurbs_cfg.SET_UP_WITH_KNOWLEDGE = \
            orig_nurbs_config.SET_UP_WITH_KNOWLEDGE
        facet.nurbs_cfg.INITIALIZE_WITH_KNOWLEDGE = \
            orig_nurbs_config.INITIALIZE_WITH_KNOWLEDGE
        facet.nurbs_cfg.freeze()

        facet.initialize_control_points(facet.ctrl_points)
        facet.initialize_eval_points()
        return cant_rot

    def _create_facet(
            self,
            facet_index: int,
            position: torch.Tensor,
            span_n: torch.Tensor,
            span_e: torch.Tensor,
            heliostat_config: CfgNode,
            nurbs_config: CfgNode,
    ) -> Tuple[NURBSHeliostat, torch.Tensor]:
        orig_nurbs_config = nurbs_config
        heliostat_config = self._facet_heliostat_config(
            heliostat_config,
            position,
            span_n,
            span_e,
        )
        nurbs_config = self._facet_nurbs_config(nurbs_config, span_n, span_e)

        facet = NURBSHeliostat(
            heliostat_config,
            nurbs_config,
            self.device,
            setup_params=False,
        )
        cant_rot = self._adjust_facet(
            facet,
            facet_index,
            position,
            span_n,
            span_e,
            orig_nurbs_config,
            nurbs_config,
        )
        return facet, cant_rot

    def _create_facets(
            self,
            heliostat_config: CfgNode,
            nurbs_config: CfgNode,
    ) -> List[Tuple[NURBSHeliostat, torch.Tensor]]:
        return [
            self._create_facet(
                i,
                position,
                span_n,
                span_e,
                heliostat_config,
                nurbs_config,
            )
            for (i, (position, span_n, span_e)) in enumerate(zip(
                    self.facets.positions,
                    self.facets.spans_n,
                    self.facets.spans_e,
            ))
        ]

    def __len__(self) -> int:
        return len(self.facets)

    def _optimizables(self) -> Dict[str, List[torch.Tensor]]:
        assert isinstance(self.facets, NURBSFacets)
        optimizables: Dict[str, List[torch.Tensor]] = {}
        for facet in self.facets:
            for (facet_name, params) in facet.optimizables().items():
                name = self._facet_optimizable_to_optimizable(facet_name)
                if name is None:
                    # This should never happen as we already filter
                    # these out during creation.
                    continue
                optimizables.setdefault(name, []).extend(params)
        return optimizables

    def _calc_normals_and_surface(
            self,
            reposition: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(self.facets, NURBSFacets)
        return self.facets.align_discrete_points_and_normals(
            reposition=reposition)

    def discrete_points_and_normals(
            self,
            reposition: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # We do this awkward call so that `AlignedMultiNURBSHeliostat`
        # can call this method but avoid using its own
        # `_calc_normals_and_surface` method.
        discrete_points, normals = \
            MultiNURBSHeliostat._calc_normals_and_surface(
                self, reposition=reposition)
        return discrete_points, normals

    def step(self, verbose: bool = False) -> None:  # type: ignore[override]
        assert isinstance(self.facets, NURBSFacets)
        facets = iter(self.facets)
        next(facets).step(verbose)
        for facet in facets:
            facet.step(False)

    @property  # type: ignore[misc]
    @functools.lru_cache()
    def dict_keys(self) -> Set[str]:
        """All keys we assume in the dictionary returned by `_to_dict`."""
        keys = super().dict_keys
        keys = cast(Set[str], keys).union({
            'nurbs_config',
            'facets',
        })
        return keys

    @functools.lru_cache()
    def _fixed_dict(self) -> Dict[str, Any]:
        data = super()._fixed_dict()
        data['nurbs_config'] = self.nurbs_cfg
        return data

    def _to_dict(self) -> Dict[str, Any]:
        assert isinstance(self.facets, NURBSFacets)
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
            receiver_center: Union[torch.Tensor, List[float], None] = None,
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
            setup_params=False,
        )
        self._from_dict(data, restore_strictly)

        assert isinstance(self.facets, NURBSFacets)
        for (facet, facet_data) in zip(self.facets, data['facets']):
            facet._from_dict(facet_data, restore_strictly)

        if setup_params:
            self.setup_params()
        return self

    def _from_dict(self, data: Dict[str, Any], restore_strictly: bool) -> None:
        super()._from_dict(data, restore_strictly)


class AlignedMultiNURBSHeliostat(AlignedNURBSHeliostat):
    _heliostat: MultiNURBSHeliostat  # type: ignore[assignment]

    def __init__(
            self,
            heliostat: MultiNURBSHeliostat,
            sun_direction: torch.Tensor,
            aim_point: torch.Tensor,
    ) -> None:
        assert isinstance(heliostat, MultiNURBSHeliostat), \
            'can only align multi-NURBS heliostat'
        AlignedHeliostat.__init__(
            cast(AlignedHeliostat, self),
            heliostat,
            sun_direction,
            aim_point,
            align_points=False,
        )

        if (
                self._heliostat.canting_enabled
                and self._heliostat.canting_algo is CantingAlgorithm.ACTIVE
        ):
            assert isinstance(self._heliostat.facets, NURBSFacets)
            self.facets = NURBSFacets(
                self._heliostat,
                [
                    cast(
                        AlignedNURBSHeliostat,
                        facet.align(sun_direction, aim_point),
                    )
                    for facet in self._heliostat.facets
                ],
                self._heliostat.facets.cant_rots,
            )
            self.device = self._heliostat.device

    def _calc_normals_and_surface(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
                self._heliostat.canting_enabled
                and self._heliostat.canting_algo is CantingAlgorithm.ACTIVE
        ):
            hel_rotated, normal_vectors_rotated = \
                MultiNURBSHeliostat.discrete_points_and_normals(
                    cast(MultiNURBSHeliostat, self), reposition=False)
            hel_rotated = hel_rotated + self._heliostat.position_on_field
        else:
            hel_rotated, normal_vectors_rotated = \
                super()._calc_normals_and_surface()

        return hel_rotated, normal_vectors_rotated


MultiNURBSHeliostat.aligned_cls = AlignedMultiNURBSHeliostat
