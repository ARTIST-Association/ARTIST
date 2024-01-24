import copy
import json
import os
import struct
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from yacs.config import CfgNode
from artist.physics_objects.heliostats.surface.facets.facets import AFacetModule
from artist.physics_objects.heliostats.surface.nurbs import bpro_loader, nurbs
from artist.util import utils

HeliostatParams = Tuple[
    torch.Tensor,  # heliostat position on field
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

def get_position(
        cfg: CfgNode,
        dtype: torch.dtype,
        device: torch.device,
) -> torch.Tensor:
    position_on_field: List[float] = cfg.POSITION_ON_FIELD
    return torch.tensor(position_on_field, dtype=dtype, device=device)


def load_heliostat_position_file(
        json_file_path: str,
        heliostat_name: str,
) -> Tuple[
    List[float],
    List[List[float]],
    List[List[float]],
    List[List[float]],
]:
    with open(json_file_path, 'r') as f:
        data = json.load(f)['data']
    values = next(filter(lambda x: x['Name'] == heliostat_name, data), None)
    assert values is not None, \
        f'heliostat {heliostat_name} not found in {json_file_path}'
    # name = values["Name"]
    position = values["Position"].item()
    facet_positions = values["FacetPositions"].item()
    facet_spans_n = values["FacetSpansN"].item()
    facet_spans_e = values["FacetSpansE"].item()
    return position, facet_positions, facet_spans_n, facet_spans_e

def _broadcast_spans(
        spans: List[List[float]],
        to_length: int,
) -> List[List[float]]:
    if len(spans) == to_length:
        return spans

    assert len(spans) == 1, (
        'will only broadcast spans of length 1. If you did not intend '
        'to broadcast, make sure there is the same amount of facet '
        'positions and spans.'
    )
    return spans * to_length


def get_facet_params(
        cfg: CfgNode,
        dtype: torch.dtype,
        device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions: List[List[float]] = utils.with_outer_list(cfg.FACETS.POSITIONS)
    spans_n: List[List[float]] = utils.with_outer_list(cfg.FACETS.SPANS_N)
    spans_n = _broadcast_spans(spans_n, len(positions))
    spans_e: List[List[float]] = utils.with_outer_list(cfg.FACETS.SPANS_E)
    spans_e = _broadcast_spans(spans_e, len(positions))
    position, spans_n, spans_e = map(
        lambda l: torch.tensor(l, dtype=dtype, device=device),
        [positions, spans_n, spans_e],
    )
    return position, spans_n, spans_e

def ideal_surface(
        ideal_configs: CfgNode,
        device: torch.device,
) -> HeliostatParams:
    """Return an ideally shaped heliostat lying flat on the ground."""
    cfg = ideal_configs

    columns: int = cfg.COLS
    column = torch.arange(columns + 1, device=device)
    row = torch.arange(cfg.ROWS + 1, device=device)

    h_x = (row/cfg.ROWS * cfg.HEIGHT) - (cfg.HEIGHT / 2)
    # Use points at centers of grid squares.
    h_x = h_x[:-1] + (h_x[1:] - h_x[:-1]) / 2
    h_x = torch.tile(h_x, (columns,))
    # heliostat y position
    h_y = (column/columns * cfg.WIDTH) - (cfg.WIDTH / 2)
    # Use points at centers of grid squares.
    h_y = h_y[:-1] + (h_y[1:] - h_y[:-1]) / 2
    h_y = torch.tile(h_y.unsqueeze(-1), (1, cfg.ROWS)).ravel()
    h_z = torch.zeros_like(h_x)

    h = torch.stack(
        [h_x, h_y, h_z],
        -1,
    ).reshape(len(h_x), -1)

    normal_vector_direction = torch.tensor(
        ideal_configs.NORMAL_VECS,
        dtype=h.dtype,
        device=device,
    )
    h_normal_vectors = torch.tile(normal_vector_direction, (len(h), 1))

    (facet_positions, facet_spans_n, facet_spans_e) = get_facet_params(
        cfg,
        dtype=h.dtype,
        device=device,
    )
    params = None
    return (
        get_position(cfg, h.dtype, device),
        facet_positions,
        facet_spans_n,
        facet_spans_e,
        h,
        h,  # h_ideal
        h_normal_vectors,
        h_normal_vectors,  # h_ideal_normal_vecs
        cfg.HEIGHT,
        cfg.WIDTH,
        cfg.ROWS,
        cfg.COLS,
        params,
    )

def real_surface(
        real_configs: CfgNode, device: torch.device,
) -> HeliostatParams:
    """Return a heliostat loaded from deflectometric data."""
    cfg = real_configs
    dtype = torch.get_default_dtype()

    heliostat_position: Union[bpro_loader.Vector3d, List[float]]
    facet_positions: Union[List[bpro_loader.Vector3d], List[List[float]]]
    facet_spans_n: Union[List[bpro_loader.Vector3d], List[List[float]]]
    facet_spans_e: Union[List[bpro_loader.Vector3d], List[List[float]]]

    concentratorHeader_struct = struct.Struct(cfg.CONCENTRATORHEADER_STRUCT_FMT)
    facetHeader_struct = struct.Struct(cfg.FACETHEADER_STRUCT_FMT)
    ray_struct = struct.Struct(cfg.RAY_STRUCT_FMT)
    (
        heliostat_position,
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

    # if cfg.LOAD_OTHER_HELIOSTAT_PROPS:
    #     (
    #         heliostat_position,
    #         facet_positions,
    #         facet_spans_n,
    #         facet_spans_e,
    #     ) = load_heliostat_position_file(
    #         os.path.join(cfg.DIRECTORY, cfg.JSON_FILE_NAME),
    #         cfg.OTHER_HELIOSTAT_NAME,
    #     )
    #     heliostat_position_new = torch.tensor(
    #         heliostat_position, dtype=dtype, device=device)
    # else:
    #     heliostat_position_new = (
    #         torch.tensor(heliostat_position, dtype=dtype, device=device)
    #         if cfg.POSITION_ON_FIELD is None
    #         else get_position(cfg, dtype, device)
    #     )
    # heliostat_position: torch.Tensor = heliostat_position_new
    # del heliostat_position_new
    # if cfg.ZS_PATH:
    #     if cfg.VERBOSE:
    #         print("Path to heliostat surface values found. Load values...")
    #     positions = copy.deepcopy(ideal_positions)
    #     integrated = bpro_loader.load_csv(
    #         cfg.ZS_PATH, len(positions))
    #     pos_type = type(positions[0][0][0])

    #     for (
    #             facet_index,
    #             (integrated_facet, pos_facet),
    #     ) in enumerate(zip(integrated, positions)):
    #         integrated_facet_iter = iter(integrated_facet)
    #         in_facet_index = 0
    #         while in_facet_index < len(pos_facet):
    #             curr_integrated = next(integrated_facet_iter)
    #             pos = pos_facet[in_facet_index]

    #             # Remove positions without matching integrated.
    #             rounded_pos = [round(val, 4) for val in pos[:-1]]
    #             rounded_integrated = [
    #                 round(val, 4)
    #                 for val in curr_integrated[:-1]
    #             ]
    #             while not all(map(
    #                     lambda tup: tup[0] == tup[1],
    #                     zip(rounded_pos, rounded_integrated),
    #             )):
    #                 pos_facet.pop(in_facet_index)
    #                 directions[facet_index].pop(in_facet_index)
    #                 ideal_normal_vecs[facet_index].pop(in_facet_index)
    #                 if in_facet_index >= len(pos_facet):
    #                     break

    #                 pos = pos_facet[in_facet_index]
    #                 rounded_pos = [round(val, 4) for val in pos[:-1]]
    #             else:
    #                 pos[-1] = pos_type(curr_integrated[-1])
    #                 in_facet_index += 1
    #     del integrated
    # else:
    #     positions = ideal_positions

    h_normal_vecs = []
    h_ideal_vecs = []
    h = []
    h_ideal = []
    if not cfg.ZS_PATH:
        if cfg.VERBOSE:
            print(
                "No path to heliostat surface values found. "
                "Calculate values..."
            )
        zs_list = []
    
    step_size = sum(map(len, directions)) // cfg.TAKE_N_VECTORS
    for f in range(len(directions)):
        h_normal_vecs.append(torch.tensor(
            directions[f][::step_size],
            dtype=dtype,
            device=device,
        ))
        h_ideal_vecs.append(torch.tensor(
            ideal_normal_vecs[f][::step_size],
            dtype=dtype,
            device=device,
        ))
        h.append(torch.tensor(
            ideal_positions[f][::step_size],
            dtype=dtype,
            device=device,
        ))
        if not cfg.ZS_PATH:
            zs_list.append(utils.deflec_facet_zs_many(
                h[-1],
                h_normal_vecs[-1],
                h_ideal_vecs[-1],
                num_samples=16,
            ))
        h_ideal.append(torch.tensor(
            ideal_positions[f][::step_size],
            dtype=dtype,
            device=device,
        ))

    h_normal_vecs: torch.Tensor = torch.cat(h_normal_vecs, dim=0)
    h_ideal_vecs: torch.Tensor = torch.cat(h_ideal_vecs, dim=0)
    h: torch.Tensor = torch.cat(h, dim=0)
    if not cfg.ZS_PATH:
        zs = torch.cat(zs_list, dim=0)
        h[:, -1] += zs

    h_ideal: torch.Tensor = torch.cat(h_ideal, dim=0)
    if cfg.VERBOSE:
        print("Done")
    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize =(14, 9))
    # ax = plt.axes(projection ='3d')
    # h[:,2] = h[:,2]-h_ideal[:,2]
    # h = h.detach().cpu()
    # # im3 = ax.scatter(h[:,0],h[:,1], c=h[:,2], cmap="magma")
    # # h.detach().cpus()
    # my_cmap = plt.get_cmap('hot')
    # ax.plot_trisurf(h[:,0],h[:,1],h[:,2], cmap =my_cmap)
    # plt.show()
    # exit()
    # plt.savefig("test.png", dpi=fig.dpi)
    # plt.close(fig)
    # exit()

    # print(h_ideal_vecs)
    rows = None
    cols = None
    params = None

    # Overwrite facet parameters if we **really** want to.
    if hasattr(cfg.FACETS, '_POSITIONS'):
        facet_positions = cfg.FACETS._POSITIONS
    if hasattr(cfg.FACETS, '_SPANS_N'):
        facet_spans_n = cfg.FACETS._SPANS_N
    if hasattr(cfg.FACETS, '_SPANS_E'):
        facet_spans_e = cfg.FACETS._SPANS_E

    return (
        heliostat_position,
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
        # powers,
    )

class NURBS:
    def __init__(
        self,
        nurbs_config: CfgNode,
        device: torch.device,
        rows: int,
        cols: int,
        discrete_points: torch.Tensor,
        width: int,
        height: int,
    ) -> None:
        
        self.device = device
        self.width = width
        self.height = height
        
        self.nurbs_cfg = nurbs_config
        if not self.nurbs_cfg.is_frozen():
            self.nurbs_cfg = self.nurbs_cfg.clone()
            self.nurbs_cfg.freeze()

        self._fix_spline_ctrl_weights: bool = \
            self.nurbs_cfg.FIX_SPLINE_CTRL_WEIGHTS
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

        #  diff
        self._discrete_points_ideal = discrete_points
        self._orig_world_points = self._discrete_points_ideal.clone()

        utils.initialize_spline_knots(
            self.knots_x, self.knots_y, self.degree_x, self.degree_y
        )
        self.ctrl_weights[:] = 1

        self.split_nurbs_params(self.ctrl_weights, self.knots_x, self.knots_y)
        self.initialize_control_points(self.ctrl_points)
        with torch.no_grad():
            self.set_ctrl_points(self.ctrl_points)
        self.initialize_eval_points()

        # nurbs.plot_surface(
        #     self.degree_x,
        #     self.degree_y,
        #     self.ctrl_points,
        #     self.ctrl_weights,
        #     self.knots_x,
        #     self.knots_y,
        # )
    
    def split_nurbs_params(
            self,
            ctrl_weights: torch.Tensor,
            knots_x: torch.Tensor,
            knots_y: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            self.set_ctrl_weights(ctrl_weights)
            self.set_knots_x(knots_x)
            self.set_knots_y(knots_y)


    def set_ctrl_points(self, ctrl_points: torch.Tensor) -> None:
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
            #assert (self.ctrl_points == ctrl_points).all()

    def set_ctrl_weights(self, ctrl_weights: torch.Tensor) -> None:
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
            #assert (ctrl_weights == ctrl_weights).all()

    def set_knots_x(self, knots_x: torch.Tensor) -> None:
        with torch.no_grad():
            self._knots_x_splits = self._split_knots(knots_x)
            #assert (self.knots_x == knots_x).all()

    def set_knots_y(self, knots_y: torch.Tensor) -> None:
        with torch.no_grad():
            self._knots_y_splits = self._split_knots(knots_y)
            #assert (self.knots_y == knots_y).all()

    @staticmethod
    def _split_knots(knots: torch.Tensor) -> List[torch.Tensor]:
        with torch.no_grad():
            return [knots[:1], knots[1:-1], knots[-1:]]
        
    def initialize_control_points(self, ctrl_points: torch.Tensor) -> None:
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
            #torch.zeros_like(self.position_on_field),
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
                    self._orig_world_points,
                    self.rows * edge_factor,
                    self.cols * edge_factor,
                )
            else:
                world_points = self._orig_world_points
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

    def initialize_eval_points(self) -> None:
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
    
    @property
    def recalc_eval_points(self) -> bool:
        return self._recalc_eval_points

    def _invert_world_points(self) -> torch.Tensor:
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
        if self.recalc_eval_points:
            eval_points = self._invert_world_points()
        else:
            eval_points = self._eval_points
        return eval_points

class NURBSFacetsModule(AFacetModule):
    def __init__(
            self,
            surface_config: CfgNode,
            nurbs_config: CfgNode,
            device: torch.device,
            setup_params: bool = True,
            receiver_center: Union[torch.Tensor, List[float], None] = None,
            sun_directions: Union[
                torch.Tensor,
                List[List[float]],
                None,
            ] = None,
    ) -> None:
        super().__init__()

        self.device = device
        self.nurbs_cfg = nurbs_config
        self.cfg = surface_config

        # maybe_sun_direction = heliostat_models.to_sun_direction(
        #     sun_directions, self.device)

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

        #diff
        cfg_position_on_field: Union[List[float], str] = \
            self.nurbs_cfg.POSITION_ON_FIELD
        if isinstance(cfg_position_on_field, str):
            if cfg_position_on_field != 'inherit':
                raise ValueError(
                    f'unknown position on field config '
                    f'"{cfg_position_on_field}"'
                )
        else:
            self.position_on_field = torch.tensor(
                self.nurbs_cfg.POSITION_ON_FIELD, device=self.device
            )

        cfg_aim_point: Union[List[float], str, None] = self.nurbs_cfg.AIM_POINT
        if isinstance(cfg_aim_point, str):
            if cfg_aim_point != 'inherit':
                raise ValueError(f'unknown aim point config "{cfg_aim_point}"')
            builder_fn, aim_point_cfg = self.select_surface_builder(self, self.cfg)
            maybe_aim_point: Optional[torch.Tensor] = receiver_center
        else:
            if (
                    receiver_center is not None
                    and not isinstance(receiver_center, torch.Tensor)
            ):
                receiver_center = torch.tensor(
                    receiver_center,
                    dtype=self.position_on_field.dtype,
                    device=self.device,
                )
            aim_point_cfg = self.nurbs_cfg
            maybe_aim_point = receiver_center
        self.aim_point = self._get_aim_point(aim_point_cfg, maybe_aim_point)


        #diff
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
            
    
        (
            heliostat_position,
            positions,
            self.spans_n,
            self.spans_e,
            self.heliostat,
            self.heliostat_ideal,
            heliostat_normals,
            heliostat_ideal_vecs,
            self.height,
            self.width,
            self.rows,
            self.cols,
            params,
        ) = builder_fn(aim_point_cfg, self.device)

        #print(positions)

        #diff
        # facets = self._create_facets(
        #     self.cfg,
        #     self.nurbs_cfg,
        #     maybe_sun_direction,
        # )
        # self.facets = NURBSFacets(
        #     self, cast(List[AbstractNURBSHeliostat], facets))
            
        self.facets = self._create_nurbs_facets(
            self.nurbs_cfg
        )

        (
            self.facetted_discrete_points,
            self.facetted_normals,
        ) = self._calc_normals_and_surface(
            self.facets.eval_points,
            self.facets.degree_x,
            self.facets.degree_y,
            self.facets.ctrl_points,
            self.facets.ctrl_weights,
            self.facets.knots_x,
            self.facets.knots_y,
        )

        self.facetted_discrete_points = torch.tensor_split(
            self.facetted_discrete_points, 4
        )
        self.facetted_normals = torch.tensor_split(self.facetted_normals, 4)


    def _calc_normals_and_surface(
        self,
        eval_points: torch.Tensor,
        degree_x: int,
        degree_y: int,
        ctrl_points: torch.Tensor,
        ctrl_weights: torch.Tensor,
        knots_x: torch.Tensor,
        knots_y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return surface_points, normals


    @staticmethod
    def _select_surface_builder(cfg: CfgNode) -> Tuple[
            Callable[[CfgNode, torch.device], HeliostatParams],
            str,
    ]:
        shape = cfg.SHAPE.lower()
        if shape == 'ideal' or shape == 'nurbs':
            return ideal_surface, 'IDEAL'
        elif shape == 'real':
            return real_surface, 'DEFLECT_DATA'
        raise ValueError('unknown surface shape')

    @staticmethod
    def select_surface_builder(self, cfg: CfgNode) -> Tuple[
            Callable[[CfgNode, torch.device], HeliostatParams],
            CfgNode,
    ]:
        builder_fn, cfg_key = self._select_surface_builder(cfg)
        h_cfg = getattr(cfg, cfg_key)
        return builder_fn, h_cfg
    

    def _get_aim_point(
            self,
            cfg: CfgNode,
            maybe_aim_point: Optional[torch.Tensor],
    ) -> torch.Tensor:
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
            raise ValueError('no aim point was supplied')
        return aim_point
    

    def _get_disturbance_angles(self, h_cfg: CfgNode) -> List[torch.Tensor]:
        angles: List[float] = h_cfg.DISTURBANCE_ROT_ANGLES
        return [
            torch.deg2rad(torch.tensor(
                angle,
                dtype=self.aim_point.dtype,
                device=self.device,
            ))
            for angle in angles
        ]
    
    @staticmethod
    def _facet_nurbs_config(
            nurbs_config: CfgNode,
            span_n: torch.Tensor,
            span_e: torch.Tensor,
    ) -> CfgNode:
        height = (torch.linalg.norm(span_n) * 2).item()
        width = (torch.linalg.norm(span_e) * 2).item()

        nurbs_config = nurbs_config.clone()
        nurbs_config.defrost()

        nurbs_config.HEIGHT = height
        nurbs_config.WIDTH = width
        return nurbs_config
    

    def _create_facet(
        self,
        span_n: torch.Tensor,
        span_e: torch.Tensor,
        nurbs_config: CfgNode,
    ) -> NURBS:
        nurbs_config = self._facet_nurbs_config(nurbs_config, span_n, span_e)

        facets = NURBS(
            nurbs_config,
            self.device,
            rows=self.rows,
            cols=self.cols,
            discrete_points=self.heliostat,
            width=self.width,
            height=self.height,
        )
        return facets

    def _create_nurbs_facets(
        self,
        nurbs_config: CfgNode,
    ) -> NURBS:
        # span_e = torch.tensor([1, 0, 0], dtype=torch.float64)
        # span_n = torch.tensor([0, 1, 0], dtype=torch.float64)
        return self._create_facet(
            self.spans_n,
            self.spans_e,
            nurbs_config,
        )

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
