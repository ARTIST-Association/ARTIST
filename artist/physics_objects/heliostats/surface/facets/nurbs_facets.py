import itertools
import struct
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
import torch
from artist.physics_objects.heliostats.surface.facets import canting, facets_copied
from artist.physics_objects.heliostats.surface.facets.canting import ActiveCanting
from artist.physics_objects.heliostats.surface.facets.facets import AFacetModule
from yacs.config import CfgNode

from artist.physics_objects.heliostats.surface.nurbs import bpro_loader, nurbs
from artist.util import utils

HeliostatParams = Tuple[
    torch.Tensor,  # surface position on field
    torch.Tensor,  # facet positions
    torch.Tensor,  # facet spans N
    torch.Tensor,  # facet spans E
    torch.Tensor,  # discrete points
    torch.Tensor,  # ideal discrete points
    torch.Tensor,  # normals
    torch.Tensor,  # ideal normals
    float,  # height
    float,  # width
    Optional[int],  # rows
    Optional[int],  # cols
    Optional[Dict[str, Any]],  # params
]


def real_surface(
    real_configs: CfgNode,
    device: torch.device,
) -> HeliostatParams:
    """
    Compute a surface loaded from deflectometric data.

    Parameters
    ----------
    real_config : CfgNode
        The config file containing Information about the real surface.
    device : torch.device
        Specifies the device type responsible to load tensors into memory.

    Returns
    -------
    HeliostatParams
        Tuple of all heliostat parameters.

    """
    cfg = real_configs
    dtype = torch.get_default_dtype()

    concentratorHeader_struct = struct.Struct(cfg.CONCENTRATORHEADER_STRUCT_FMT)
    facetHeader_struct = struct.Struct(cfg.FACETHEADER_STRUCT_FMT)
    ray_struct = struct.Struct(cfg.RAY_STRUCT_FMT)

    (
        surface_position,
        facet_positions,
        facet_spans_n,
        facet_spans_e,
        ideal_positions,
        directions,
        ideal_normal_vecs,
        width,
        height,
    ) = bpro_loader.load_bpro(
        cfg.FILENAME,
        concentratorHeader_struct,
        facetHeader_struct,
        ray_struct,
        cfg.VERBOSE,
    )

    surface_position: torch.Tensor = (
        torch.tensor(surface_position, dtype=dtype, device=device)
        if cfg.POSITION_ON_FIELD is None
        else get_position(cfg, dtype, device)
    )

    h_normal_vecs = []
    h_ideal_vecs = []
    h = []
    h_ideal = []

    step_size = sum(map(len, directions)) // cfg.TAKE_N_VECTORS
    for f in range(len(directions)):
        h_normal_vecs.append(
            torch.tensor(
                directions[f][::step_size],
                dtype=dtype,
                device=device,
            )
        )
        h_ideal_vecs.append(
            torch.tensor(
                ideal_normal_vecs[f][::step_size],
                dtype=dtype,
                device=device,
            )
        )
        h.append(
            torch.tensor(
                ideal_positions[f][::step_size],
                dtype=dtype,
                device=device,
            )
        )

        h_ideal.append(
            torch.tensor(
                ideal_positions[f][::step_size],
                dtype=dtype,
                device=device,
            )
        )

    h_normal_vecs: torch.Tensor = torch.cat(h_normal_vecs, dim=0)
    h_ideal_vecs: torch.Tensor = torch.cat(h_ideal_vecs, dim=0)
    h: torch.Tensor = torch.cat(h, dim=0)

    h_ideal: torch.Tensor = torch.cat(h_ideal, dim=0)
    if cfg.VERBOSE:
        print("Done")

    rows = None
    cols = None
    params = None
    return (
        surface_position,
        torch.tensor(facet_positions, dtype=dtype, device=device),
        torch.tensor(facet_spans_n, dtype=dtype, device=device),
        torch.tensor(facet_spans_e, dtype=dtype, device=device),
        h,
        h_ideal,
        h_normal_vecs,
        h_ideal_vecs,
        height,
        width,
        rows,
        cols,
        params,
    )


def get_position(
    cfg: CfgNode,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Retrieve the position of the heliostat in the field as tensor.

    Parameters
    ----------
    cfg : CfgNode
        The config file containing the information about the heliostat.
    dtype : torch.dtype
        The type and size of the tensor.
    device : torch.device
        Specifies the device type responsible to load tensors into memory.

    Returns
    -------
    torch.tensor
        The position of the heliostat in the field.
    """
    position_on_field: List[float] = cfg.POSITION_ON_FIELD
    return torch.tensor(position_on_field, dtype=dtype, device=device)

class NURBS:
    """
    Implementation of the NURBS surface.
    """

    def __init__(
        self,
        heliostat_config: CfgNode,
        nurbs_config: CfgNode,
        device: torch.device,
        rows: int,
        cols: int,
        discrete_points: torch.Tensor,
        width: int,
        height: int,
        # receiver_center: Union[torch.Tensor, List[float], None] = None,
        # sun_directions: Union[
        #     torch.Tensor,
        #     List[List[float]],
        #     None,
        # ] = None,
        # setup_params: bool = True,
    ) -> None:
        """
        Initialize the NURBS surface.

        Parameters
        ----------
        nurbs_config : CfgNode
            Config Node containing information about the NURBS.
        device : torch.device
            Specifies the device type responsible to load tensors into memory.
        rows : int
            Number of rows (control points).
        cols : int
            Number of columns (control points).
        discrete_points : torch.Tensor
            Discrete points of the surface
        width : int
            Width of the surface
        height: int
            Height of the surface.
        """
        self.device = device
        self.width = width
        self.height = height
        self.nurbs_cfg = nurbs_config
        if not self.nurbs_cfg.is_frozen():
            self.nurbs_cfg = self.nurbs_cfg.clone()
            self.nurbs_cfg.freeze()

        self._fix_spline_ctrl_weights: bool = self.nurbs_cfg.FIX_SPLINE_CTRL_WEIGHTS
        self._fix_spline_knots: bool = self.nurbs_cfg.FIX_SPLINE_KNOTS
        self._recalc_eval_points: bool = self.nurbs_cfg.RECALCULATE_EVAL_POINTS

        spline_degree: int = self.nurbs_cfg.SPLINE_DEGREE
        self.degree_x = spline_degree
        self.degree_y = spline_degree
        self.h_rows: Optional[int] = rows
        self.h_cols: Optional[int] = cols
        self.rows: int = self.nurbs_cfg.ROWS
        self.cols: int = self.nurbs_cfg.COLS

        (
            self.ctrl_points,
            self.ctrl_weights,
            self.knots_x,
            self.knots_y,
        ) = nurbs.setup_nurbs_surface(
            self.degree_x, self.degree_y, self.rows, self.cols, self.device
        )

        self._discrete_points_ideal = discrete_points
        self._orig_world_points = self._discrete_points_ideal.clone()

        utils.initialize_spline_knots(
            self.knots_x, self.knots_y, self.degree_x, self.degree_y
        )
        self.ctrl_weights[:] = 1

        self.initialize_control_points(self.ctrl_points)
        self.split_nurbs_params(
            self.ctrl_points, self.ctrl_weights, self.knots_x, self.knots_y
        )
        self.initialize_eval_points()

        # nurbs.plot_surface(
        #     self.degree_x,
        #     self.degree_y,
        #     self.ctrl_points,
        #     self.ctrl_weights,
        #     self.knots_x,
        #     self.knots_y,
        # )

        # self.surface_points, self.surface_normals = self._calc_normals_and_surface()
        # print(self.surface_points.size())
        # print(self.surface_points)

        # z_values = torch.zeros(self._eval_points.size(0), 1)
        # self.surface_points = torch.cat((self._eval_points, z_values), dim=1)
        # self.surface_normals = torch.tensor([[0.0, 0.0, 1.0]] * self.surface_points.size(0))

    @property
    def recalc_eval_points(self) -> bool:
        return self._recalc_eval_points

    def _invert_world_points(self) -> torch.Tensor:
        """
        Initialize spline evaluation points.

        Returns
        -------
        torch.Tensor
            The spline evaluation points.
        """
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
        """
        Compute the evaluation points of the NURBS surface.

        Returns
        -------
        torch.Tensor
            The evaluation points.
        """
        if self.recalc_eval_points:
            eval_points = self._invert_world_points()
        else:
            eval_points = self._eval_points
        return eval_points

    def initialize_control_points(self, ctrl_points: torch.Tensor) -> None:
        """
        Initialize the control points of the NURBS surface.

        Parameters
        ----------
        ctrl_points : torch.Tensor
            The control points of the NURBS surface.
        """
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
            torch.tensor([0, 0, 0]),
            # torch.zeros_like(self.position_on_field),
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
                    self._discrete_points_ideal,
                    self.rows * edge_factor,
                    self.cols * edge_factor,
                )
            else:
                world_points = self._discrete_points_ideal
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

    def split_nurbs_params(
        self,
        ctrl_points: torch.Tensor,
        ctrl_weights: torch.Tensor,
        knots_x: torch.Tensor,
        knots_y: torch.Tensor,
    ) -> None:
        """
        Split up the NURBS parameters.

        Parameters
        ----------
        ctrl_points : torch.Tensor
            The control points of the NURBS surface.
        ctrl_weights : torch.Tensor
            The corresponding weights to the control points.
        knots_x : torch.Tensor
            List of numbers representing the knots in x dimension.
        knots_y : torch.Tensor
            List of numbers representing the knots in y dimension.
        """
        with torch.no_grad():
            self.set_ctrl_points(ctrl_points)
            self.set_ctrl_weights(ctrl_weights)
            self.set_knots_x(knots_x)
            self.set_knots_y(knots_y)

    def set_ctrl_points(self, ctrl_points: torch.Tensor) -> None:
        """
        Setter function for the NURBS control points.

        Parameters
        ----------
        ctrl_points : torch.Tensor
            The control points of the NURBS surface.
        """
        with torch.no_grad():
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
            # assert (self.ctrl_points == ctrl_points).all()

    def set_ctrl_weights(self, ctrl_weights: torch.Tensor) -> None:
        """
        Setter function for the control weights.

        Parameters
        ----------
        ctrl_weights : torch.Tensor
            The control weights to the control points of the NURBS surface.
        """
        with torch.no_grad():
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
            # assert (self.ctrl_weights == ctrl_weights).all()

    def set_knots_x(self, knots_x: torch.Tensor) -> None:
        """
        Setter function for the knots in the x dimension.

        Parameters
        ----------
        knots_x : torch.Tensor
            The knots in x dimension.
        """
        with torch.no_grad():
            self._knots_x_splits = self._split_knots(knots_x)
            # assert (self.knots_x == knots_x).all()

    def set_knots_y(self, knots_y: torch.Tensor) -> None:
        """
        Setter function for the knots in the y dimension.

        Parameters
        ----------
        knots_y : torch.Tensor
            The knots in y dimension.
        """
        with torch.no_grad():
            self._knots_y_splits = self._split_knots(knots_y)
            # assert (self.knots_y == knots_y).all()

    @staticmethod
    def _split_knots(knots: torch.Tensor) -> List[torch.Tensor]:
        """
        Split the knots along their dimensions.

        Parameters
        ----------
        knots : torch.Tensor
            The knots of the NURBS surface.
        """
        with torch.no_grad():
            return [knots[:1], knots[1:-1], knots[-1:]]

    def initialize_eval_points(self) -> None:
        """
        Initialize the evaluation points of the NURBS surface.
        """
        if self.nurbs_cfg.SET_UP_WITH_KNOWLEDGE:
            if not self.recalc_eval_points:
                self._eval_points = utils.initialize_spline_eval_points_perfectly(
                    self._orig_world_points,
                    self.degree_x,
                    self.degree_y,
                    self.ctrl_points,
                    self.ctrl_weights,
                    self.knots_x,
                    self.knots_y,
                )
            # print(self._eval_points.size())
        else:
            # Unless we change the knots, we don't need to recalculate
            # as we simply distribute the points uniformly.
            self._recalc_eval_points = False
            self._eval_points = utils.initialize_spline_eval_points(
                self.rows, self.cols, self.device
            )
            # print(self._eval_points.size())

    @property
    def discrete_points(self) -> torch.Tensor:
        discrete_points, _ = self._calc_normals_and_surface()
        return discrete_points

    @property
    def normals(self) -> torch.Tensor:
        _, normals = self._calc_normals_and_surface()
        return normals
    
    def _calc_normals_and_surface(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # eval_points = self.eval_points
        # ctrl_points, ctrl_weights, knots_x, knots_y = \
        #     self._progressive_growing.select()

        surface_points, normals = nurbs.calc_normals_and_surface_slow(
            self.eval_points[:, 0],
            self.eval_points[:, 1],
            self.degree_x,
            self.degree_y,
            self.ctrl_points,
            self.ctrl_weights,
            self.knots_x,
            self.knots_y,
        )

        # do_canting = True
        # if do_canting:
        #     position = self.facets.positions[0]
        #     surface_points = surface_points - position
        #     surface_points = canting.apply_rotation(
        #         self.facets.cant_rots[0], surface_points)
        #     normals = canting.apply_rotation(self.facets.cant_rots[0], normals)
        #     surface_points = surface_points + position

        return surface_points, normals

class NURBSFacets():
    def __init__(
            self,
            heliostat: 'NURBSFacetsModule',
            nurbs_heliostats: List[NURBS],
    ):
        self._facets = nurbs_heliostats
        self.positions = heliostat.positions
        self.spans_n = heliostat.spans_n
        self.spans_e = heliostat.spans_e

        # self.offsets = self._make_offsets(
        #     cast(List['NURBSFacets'], self._facets))
        self._canting_algo = heliostat.canting_algo
        if (
                self._canting_algo is None
                or isinstance(self._canting_algo, ActiveCanting)
        ):
            self.cant_rots = torch.stack([
                torch.eye(3, dtype=self.spans_n.dtype, device=self.spans_n.device)
                for _ in range(len(self._facets))
            ])
        else:
            self.cant_rots = torch.stack(
                [facet.facets.cant_rots[0] for facet in self._facets])
            for facet in self._facets:
                facet.facets.cant_rots[0] = torch.eye(
                    3, dtype=self.spans_n.dtype, device=self.spans_n.device)

    @property
    def _discrete_points(self) -> List[torch.Tensor]:  # type: ignore[override]
        discrete_points = [facet.discrete_points for facet in self._facets]
        return discrete_points
    
    @property
    def _normals(self) -> List[torch.Tensor]:  # type: ignore[override]
        return [facet.normals for facet in self._facets]


class NURBSFacetsModule(AFacetModule):
    """
    Implementation of the NURBS Facets.

    See also
    --------
    :class:AFacetModule : The parent class.
    """

    def __init__(
        self,
        surface_config: CfgNode,
        nurbs_config: CfgNode,
        device: torch.device,
        # setup_params: bool = True,
        receiver_center: Union[torch.Tensor, List[float], None] = None,
        sun_directions: Union[
            torch.Tensor,
            List[List[float]],
            None,
        ] = None,
    ) -> None:
        """
        Initialize the NURBS facets.

        Parameters
        ----------
        surface_config : CfgNode
            Config node containing information about the surface of the concentrator.
        nurbs_config : CfgNode
            Config node containing information about the NURBS.
        device : torch.device
            Specifies the device type responsible to load tensors into memory.
        receiver_center : Union[torch.Tensor, List[float], None] = None
            The center of the receiver.
        """
        super().__init__()

        self.device = device
        self.nurbs_cfg = nurbs_config
        self.surface_config = surface_config
        self.sun_directions = sun_directions

        if not self.nurbs_cfg.is_frozen():
            self.nurbs_cfg = self.nurbs_cfg.clone()
            self.nurbs_cfg.freeze()

        cfg_width: float = self.nurbs_cfg.WIDTH
        if isinstance(cfg_width, str):
            if cfg_width != "inherit":
                raise ValueError(f'unknown width config "{cfg_width}"')
        else:
            self.width = cfg_width

        cfg_height: float = self.nurbs_cfg.HEIGHT
        if isinstance(cfg_height, str):
            if cfg_height != "inherit":
                raise ValueError(f'unknown height config "{cfg_height}"')
        else:
            self.height = cfg_height

        cfg_position_on_field: Union[
            List[float], str
        ] = self.nurbs_cfg.POSITION_ON_FIELD
        if isinstance(cfg_position_on_field, str):
            if cfg_position_on_field != "inherit":
                raise ValueError(
                    f"unknown position on field config " f'"{cfg_position_on_field}"'
                )
        else:
            self.position_on_field = torch.tensor(
                self.nurbs_cfg.POSITION_ON_FIELD, device=self.device
            )

        cfg_aim_point: Union[List[float], str, None] = self.nurbs_cfg.AIM_POINT
        if isinstance(cfg_aim_point, str):
            if cfg_aim_point != "inherit":
                raise ValueError(f'unknown aim point config "{cfg_aim_point}"')
            builder_fn, aim_point_cfg = self.select_surface_builder(self.surface_config)
            maybe_aim_point: Optional[torch.Tensor] = receiver_center

        else:
            if receiver_center is not None and not isinstance(
                receiver_center, torch.Tensor
            ):
                receiver_center = torch.tensor(
                    receiver_center,
                    dtype=self.position_on_field.dtype,
                    device=self.device,
                )
            aim_point_cfg = self.nurbs_cfg
            maybe_aim_point = receiver_center
        self.aim_point = self._get_aim_point(aim_point_cfg, maybe_aim_point)

        cfg_disturbance_angles: Union[
            List[float], str
        ] = self.nurbs_cfg.DISTURBANCE_ROT_ANGLES
        if isinstance(cfg_disturbance_angles, str):
            if cfg_disturbance_angles != "inherit":
                raise ValueError(
                    f"unknown disturbance angles config " f'"{cfg_disturbance_angles}"'
                )
        else:
            # Radians
            self.disturbance_angles = self._get_disturbance_angles(self.nurbs_cfg)

        self._canting_cfg: CfgNode = nurbs_config.FACETS.CANTING
        (
            heliostat_position,
            self.positions,
            self.spans_n,
            self.spans_e,
            self.heliostat,
            self.heliostat_ideal,
            self.heliostat_normals,
            self.heliostat_ideal_vecs,
            self.height,
            self.width,
            self.rows,
            self.cols,
            params,
        ) = builder_fn(aim_point_cfg, self.device)

        self.heliostat_ideal = torch.tensor_split(self.heliostat_ideal, 4)
        self.heliostat_normals = torch.tensor_split(self.heliostat_normals, 4)
        
        self._inherit_canting()
        
        facets = self._create_facets(
            self.surface_config,
            self.nurbs_cfg,
            sun_directions,
        )
        
        self.facets = NURBSFacets(
            self,
            facets)
        
        self.facetted_discrete_points = torch.stack(self.facets._discrete_points, dim=0)
        self.facetted_normals = torch.stack(self.facets._normals, dim=0)


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

        cfg_canting_algo: str = self._canting_cfg.ALGORITHM
        if cfg_canting_algo == 'inherit':
            self._canting_cfg.ALGORITHM = old_canting_cfg.ALGORITHM
        self.canting_algo = canting.get_algorithm(self._canting_cfg)
        
        self.canting_enabled = True
        if self.canting_enabled:
            focus_point = canting.get_focus_point(
                self._canting_cfg,
                self.aim_point,
                self.surface_config.IDEAL.NORMAL_VECS,
                device=self.device,
            )
        else:
            focus_point = None
        self._set_deconstructed_focus_point(focus_point)

        self._canting_cfg.freeze()

    @staticmethod
    def _deconstruct_focus_point(
            focus_point: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if focus_point is not None:
            focus_distance = torch.linalg.norm(focus_point)
            focus_normal = focus_point / focus_distance
        else:
            focus_normal = None
            focus_distance = None
        return focus_normal, focus_distance

    def _set_deconstructed_focus_point(
            self,
            focus_point: Optional[torch.Tensor],
    ) -> None:
        self._focus_normal, self._focus_distance = \
            self._deconstruct_focus_point(focus_point)
        

    def _get_aim_point(
        self,
        cfg: CfgNode,
        maybe_aim_point: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Getter-function for the aimpoint.

        Parameters
        ----------
        cfg : CfgNode
            Config Node containing information about the aimpoint.
        maybe_aim_point : Optional[torch.Tensor]
            The aimpoint.

        Returns
        -------
        torch.Tensor
            The aimpoint.
        """
        cfg_aim_point: Optional[List[float]] = cfg.AIM_POINT
        if cfg_aim_point is not None:
            aim_point = torch.tensor(
                cfg_aim_point,
                dtype=torch.get_default_dtype(),
                device=self.device,
            )
        elif maybe_aim_point is not None:
            aim_point = maybe_aim_point
        else:
            raise ValueError("no aim point was supplied")
        return aim_point

    def _get_disturbance_angles(self, cfg: CfgNode) -> List[torch.Tensor]:
        """
        Getter function for the disturbance angles.

        Parameters
        ----------
        cfg : CfgNode
            The Config Node containing information about the disturbance angles.

        Returns
        -------
        List[torch.Tensor]
            The disturbance rotation angles.
        """
        angles: List[float] = cfg.DISTURBANCE_ROT_ANGLES
        return [
            torch.deg2rad(
                torch.tensor(
                    angle,
                    dtype=self.aim_point.dtype,
                    device=self.device,
                )
            )
            for angle in angles
        ]

    def select_surface_builder(
        self, cfg: CfgNode
    ) -> Tuple[Callable[[CfgNode, torch.device], HeliostatParams], CfgNode,]:
        """
        Select which kind of surface is to be loaded.

        At the moment only real surfaces can be loaded.

        Parameters
        ----------
        cfg : CfgNode
            Contains the information about the shape/ kind of surface to be loaded.

        Returns
        -------
        Tuple[Callable[[CfgNode, torch.device], HeliostatParams], CfgNode,]
            Returns the loaded surface, the heliostat parameters and deflectometry data.
        """
        shape = cfg.SHAPE.lower()
        if shape == "real":
            return real_surface, cfg.DEFLECT_DATA
        raise ValueError("unknown surface shape")

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
            # self._FACET_OPTIMIZABLES[name]
            # for name in self.get_to_optimize()
            # if name in self._FACET_OPTIMIZABLES
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
        """
        Build the configuration file for the NURBS surface.

        Parameters
        ----------
        nurbs_config : CfgNode
            The config node containing the information about the NURBS.
        span_n : torch.Tensor

        span_e : torch.Tensor


        Returns
        -------
        CfgNode
            The config Node containing information abou the NURBS.
        """
        height = (torch.linalg.norm(span_n) * 2).item()
        width = (torch.linalg.norm(span_e) * 2).item()

        nurbs_config = nurbs_config.clone()
        nurbs_config.defrost()

        nurbs_config.HEIGHT = height
        nurbs_config.WIDTH = width

        #nurbs_config.SET_UP_WITH_KNOWLEDGE = False
        nurbs_config.INITIALIZE_WITH_KNOWLEDGE = False
        return nurbs_config
    
    def _create_facet(
            self,
            facet_index: int,
            position: torch.Tensor,
            span_n: torch.Tensor,
            span_e: torch.Tensor,
            heliostat_config: CfgNode,
            nurbs_config: CfgNode,
            sun_direction: Optional[torch.Tensor],
    ) -> NURBS:
        heliostat_config = self._facet_heliostat_config(
            heliostat_config,
            position,
            span_n,
            span_e,
        )
        nurbs_config = self._facet_nurbs_config(nurbs_config, span_n, span_e)

        facet = NURBS(
            heliostat_config,
            nurbs_config,
            self.device,
            rows=self.rows,
            cols=self.cols,
            discrete_points=self.heliostat_ideal[facet_index],
            width=self.width,
            height=self.height,
            # setup_params=False,
        )
        return facet
    
    def _create_facets(
            self,
            heliostat_config: CfgNode,
            nurbs_config: CfgNode,
            sun_direction: Optional[torch.Tensor],
    ) -> List[Tuple[NURBS, torch.Tensor]]:
        return [
            self._create_facet(
                i,
                position,
                span_n,
                span_e,
                heliostat_config,
                nurbs_config,
                sun_direction,
            )
            for (i, (position, span_n, span_e)) in enumerate(zip(
                    self.positions,
                    self.spans_n,
                    self.spans_e,
            ))
        ]
    

    def facetted_discrete_points_and_normals(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the facetted surface points and facetted surface normals.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The facetted surface points and facetted surface normals.
        """
        return self.facetted_discrete_points, self.facetted_normals

    def make_facets_list(self) -> list[list[torch.Tensor], list[torch.Tensor]]:
        """
        Create a list of facets.

        Returns
        List[List[torch.Tensor], List[torch.Tensor]]
            The list of all facets with their points and normals.
        """
        facets = []
        (
            facetted_discrete_points,
            facetted_normals,
        ) = self.facetted_discrete_points_and_normals()
        # for i in range(len(facetted_discrete_points)):
        #     if(i == 0 or i == 2):
        #         facets.append([facetted_discrete_points[i], facetted_normals[i]])
        for points, normals in zip(facetted_discrete_points, facetted_normals):
            facets.append([points, normals])

        return facets
