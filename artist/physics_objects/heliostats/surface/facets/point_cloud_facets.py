import copy
import struct
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar
from matplotlib import pyplot as plt

import torch
from artist.physics_objects.heliostats.surface.concentrator import ConcentratorModule
from artist.physics_objects.heliostats.surface.facets.facets import AFacetModule

from yacs.config import CfgNode

from artist.physics_objects.heliostats.surface.nurbs import bpro_loader, canting
from artist.util import utils


C = TypeVar('C', bound='PointCloudFacetModule')

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
        real_configs: CfgNode, device: torch.device,
) -> HeliostatParams:
    """Return a surface loaded from deflectometric data."""
    cfg = real_configs
    dtype = torch.get_default_dtype()

    concentratorHeader_struct = struct.Struct(
        cfg.CONCENTRATORHEADER_STRUCT_FMT)
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

    if cfg.ZS_PATH:
        if cfg.VERBOSE:
            print("Path to surface values found. Load values...")
        positions = copy.deepcopy(ideal_positions)
        integrated = bpro_loader.load_csv(cfg.ZS_PATH, len(positions))
        pos_type = type(positions[0][0][0])

        for (
                facet_index,
                (integrated_facet, pos_facet),
        ) in enumerate(zip(integrated, positions)):
            integrated_facet_iter = iter(integrated_facet)
            in_facet_index = 0
            while in_facet_index < len(pos_facet):
                curr_integrated = next(integrated_facet_iter)
                pos = pos_facet[in_facet_index]

                # Remove positions without matching integrated.
                rounded_pos = [round(val, 4) for val in pos[:-1]]
                rounded_integrated = [
                    round(val, 4)
                    for val in curr_integrated[:-1]
                ]
                while not all(map(
                        lambda tup: tup[0] == tup[1],
                        zip(rounded_pos, rounded_integrated),
                )):
                    pos_facet.pop(in_facet_index)
                    directions[facet_index].pop(in_facet_index)
                    ideal_normal_vecs[facet_index].pop(in_facet_index)
                    if in_facet_index >= len(pos_facet):
                        break

                    pos = pos_facet[in_facet_index]
                    rounded_pos = [round(val, 4) for val in pos[:-1]]
                else:
                    pos[-1] = pos_type(curr_integrated[-1])
                    in_facet_index += 1
        del integrated
    else:
        positions = ideal_positions

    h_normal_vecs = []
    h_ideal_vecs = []
    h = []
    h_ideal = []
    if not cfg.ZS_PATH:
        if cfg.VERBOSE:
            print(
                "No path to surface surface values found. "
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
            positions[f][::step_size],
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
    
    ###################################################################
    # fig = plt.figure(figsize =(14, 9))
    # ax = plt.axes(projection ='3d')
    # h[:,2] = h[:,2]-h_ideal[:,2]
    # h = h.detach().cpu()
    # # im3 = ax.scatter(h[:,0],h[:,1], c=h[:,2], cmap="magma")
    # # h.detach().cpus()
    # my_cmap = plt.get_cmap('hot')
    # ax.plot_trisurf(h[:,0],h[:,1],h[:,2], cmap =my_cmap)
    # plt.show()
    ####################################################################

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
    position_on_field: List[float] = cfg.POSITION_ON_FIELD
    return torch.tensor(position_on_field, dtype=dtype, device=device)

def _indices_between(
        points: torch.Tensor,
        from_: torch.Tensor,
        to: torch.Tensor,
) -> torch.Tensor:
    indices = (
        (from_ <= points) & (points < to)
    ).all(dim=-1)
    return indices

def facet_point_indices(
        points: torch.Tensor,
        position: torch.Tensor,
        span_n: torch.Tensor,
        span_e: torch.Tensor,
) -> torch.Tensor:
    from_xyz = position + span_e - span_n
    to_xyz = position - span_e + span_n
    # We ignore the z-axis here.
    return _indices_between(
        points[:, :-1],
        from_xyz[:-1],
        to_xyz[:-1],
    )

class PointCloudFacetModule(AFacetModule):
    def __init__(self,
                 config : CfgNode,
                 aimpoint : torch.Tensor,
                 sun_direction : torch.Tensor,
                 device : torch.device
        ) -> None:
        super().__init__()
        
        self.cfg = config
        self.device = device
        self.canting_enabled = True

        self.load(aimpoint, sun_direction)
        

    def load(
            self,
            maybe_aim_point: Optional[torch.Tensor],
            maybe_sun_direction: Optional[torch.Tensor],
    ) -> None:
        builder_fn, h_cfg = self.select_surface_builder(self.cfg)
        self._canting_cfg: CfgNode = h_cfg.FACETS.CANTING

        self.canting_algo = canting.get_algorithm(self._canting_cfg)

        self.aim_point = self._get_aim_point(
            h_cfg,
            maybe_aim_point,
        )
        # Radians
        self.disturbance_angles = self._get_disturbance_angles(h_cfg)

        (
            surface_position,
            facet_positions,
            facet_spans_n,
            facet_spans_e,
            surface_points,
            surface_ideal,
            surface_normals,
            surface_ideal_vecs,
            height,
            width,
            rows,
            cols,
            params,
        ) = builder_fn(h_cfg, self.device)

        self.position_on_field = surface_position
        self.set_up_facets(
            facet_positions,
            facet_spans_n,
            facet_spans_e,
            surface_points,
            surface_ideal,
            surface_normals,
            surface_ideal_vecs,
            maybe_sun_direction,
        )
        self.params = params
        self.height = height
        self.width = width
        self.rows = rows
        self.cols = cols

        self.surface_points = surface_points
        self.surface_normals = surface_normals

    def select_surface_builder(self, cfg: CfgNode) -> Tuple[
            Callable[[CfgNode, torch.device], HeliostatParams],
            CfgNode,
    ]:
        shape = cfg.SHAPE.lower()
        if shape == "real":
            return real_surface, cfg.DEFLECT_DATA
        raise ValueError('unknown surface shape')
    
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
    
    def set_up_facets(
            self,
            facet_positions: torch.Tensor,
            facet_spans_n: torch.Tensor,
            facet_spans_e: torch.Tensor,
            discrete_points: torch.Tensor,
            discrete_points_ideal: torch.Tensor,
            normals: torch.Tensor,
            normals_ideal: torch.Tensor,
            maybe_sun_direction: Optional[torch.Tensor],
    ) -> None:
        if self.canting_enabled:
            self.focus_point = canting.get_focus_point(
                self._canting_cfg,
                self.aim_point,
                self.cfg.IDEAL.NORMAL_VECS,
                facet_positions.dtype,
                self.device,
            )
        else:
            self.focus_point = None
        self._set_deconstructed_focus_point(self.focus_point)

        self.facetted_discrete_points, self.facetted_normals= self.find_facets(
            facet_positions,
            facet_spans_n,
            facet_spans_e,
            discrete_points,
            discrete_points_ideal,
            normals,
            normals_ideal,
            maybe_sun_direction,
        )

    def _set_deconstructed_focus_point(
            self,
            focus_point: Optional[torch.Tensor],
    ) -> None:
        self._focus_normal, self._focus_distance = \
            self._deconstruct_focus_point(focus_point)
        
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
    
    def find_facets(
            surface: 'ConcentratorModule',
            positions: torch.Tensor,
            spans_n: torch.Tensor,
            spans_e: torch.Tensor,
            discrete_points: torch.Tensor,
            discrete_points_ideal: torch.Tensor,
            normals: torch.Tensor,
            normals_ideal: torch.Tensor,
            sun_direction: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        facetted_discrete_points: List[torch.Tensor] = []
        facetted_discrete_points_ideal: List[torch.Tensor] = []
        facetted_normals: List[torch.Tensor] = []
        facetted_normals_ideal: List[torch.Tensor] = []
        cant_rots: torch.Tensor = torch.empty(
            (len(positions), 3, 3),
            dtype=positions.dtype,
            device=positions.device,
        )

        canting_params = canting.get_canting_params(surface, sun_direction)

        for (i, (position, span_n, span_e)) in enumerate(zip(
                positions,
                spans_n,
                spans_e,
        )):
            # Select points on facet based on positions of ideal points.
            indices = facet_point_indices(
                discrete_points_ideal, position, span_n, span_e)
            facet_discrete_points = discrete_points[indices]
            facet_discrete_points_ideal = discrete_points_ideal[indices]
            facet_normals = normals[indices]
            facet_normals_ideal = normals_ideal[indices]

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
                surface.cfg.IDEAL.NORMAL_VECS,
                canting_params,
            )
            # Re-center facet around zero.
            facet_discrete_points -= position
            facet_discrete_points_ideal -= position

            facetted_discrete_points.append(facet_discrete_points)
            facetted_discrete_points_ideal.append(facet_discrete_points_ideal)
            facetted_normals.append(facet_normals)
            facetted_normals_ideal.append(facet_normals_ideal)
            cant_rots[i] = cant_rot

        return facetted_discrete_points, facetted_normals
    
    def discrete_points_and_normals(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.surface_points, self.surface_normals
    
    def facetted_discrete_points_and_normals(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.facetted_discrete_points, self.facetted_normals
    
    def make_facets_list(self):
        facets = []
        facetted_discrete_points, facetted_normals = self.facetted_discrete_points_and_normals()
        for i in range(len(facetted_discrete_points)):
            facets.append([facetted_discrete_points[i], facetted_normals[i]])
        return facets
        