import copy
import functools
import struct
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import pytorch3d.transforms as throt
import torch
import torch as th
from yacs.config import CfgNode

import bpro_loader
import canting
from canting import CantingAlgorithm
import utils

ParamGroups = Iterable[Dict[str, Any]]

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
A = TypeVar('A', bound='AbstractHeliostat')
C = TypeVar('C', bound='Heliostat')


def reflect_rays_(rays: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    return rays - 2 * utils.batch_dot(rays, normals) * normals


def reflect_rays(rays: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    normals = normals / th.linalg.norm(normals, dim=-1).unsqueeze(-1)
    return reflect_rays_(rays, normals)


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


def get_position(
        cfg: CfgNode,
        dtype: th.dtype,
        device: th.device,
) -> torch.Tensor:
    position_on_field: List[float] = cfg.POSITION_ON_FIELD
    return th.tensor(position_on_field, dtype=dtype, device=device)


def get_facet_params(
        cfg: CfgNode,
        dtype: th.dtype,
        device: th.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions: List[List[float]] = utils.with_outer_list(cfg.FACETS.POSITIONS)
    spans_n: List[List[float]] = utils.with_outer_list(cfg.FACETS.SPANS_N)
    spans_n = _broadcast_spans(spans_n, len(positions))
    spans_e: List[List[float]] = utils.with_outer_list(cfg.FACETS.SPANS_E)
    spans_e = _broadcast_spans(spans_e, len(positions))
    position, spans_n, spans_e = map(
        lambda l: th.tensor(l, dtype=dtype, device=device),
        [positions, spans_n, spans_e],
    )
    return position, spans_n, spans_e


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


def _sole_facet(
        height: float,
        width: float,
        dtype: th.dtype,
        device: th.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        th.zeros((1, 3), dtype=dtype, device=device),
        th.tensor([[0, height / 2, 0]], dtype=dtype, device=device),
        th.tensor([[-width / 2, 0, 0]], dtype=dtype, device=device),
    )


# Heliostat Models
# ================

def real_heliostat(
        real_configs: CfgNode, device: th.device,
) -> HeliostatParams:
    """Return a heliostat loaded from deflectometric data."""
    cfg = real_configs
    dtype = th.get_default_dtype()

    concentratorHeader_struct = struct.Struct(
        cfg.CONCENTRATORHEADER_STRUCT_FMT)
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
    heliostat_position: torch.Tensor = (
        th.tensor(heliostat_position, dtype=dtype, device=device)
        if cfg.POSITION_ON_FIELD is None
        else get_position(cfg, dtype, device)
    )

    if cfg.ZS_PATH:
        if cfg.VERBOSE:
            print("Path to heliostat surface values found. Load values...")
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
                "No path to heliostat surface values found. "
                "Calculate values..."
            )
        zs_list = []
    step_size = sum(map(len, directions)) // cfg.TAKE_N_VECTORS
    for f in range(len(directions)):
        h_normal_vecs.append(th.tensor(
            directions[f][::step_size],
            dtype=dtype,
            device=device,
        ))
        h_ideal_vecs.append(th.tensor(
            ideal_normal_vecs[f][::step_size],
            dtype=dtype,
            device=device,
        ))
        h.append(th.tensor(
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
        h_ideal.append(th.tensor(
            ideal_positions[f][::step_size],
            dtype=dtype,
            device=device,
        ))

    h_normal_vecs: torch.Tensor = th.cat(h_normal_vecs, dim=0)
    h_ideal_vecs: torch.Tensor = th.cat(h_ideal_vecs, dim=0)
    h: torch.Tensor = th.cat(h, dim=0)
    if not cfg.ZS_PATH:
        zs = th.cat(zs_list, dim=0)
        h[:, -1] += zs

    h_ideal: torch.Tensor = th.cat(h_ideal, dim=0)
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
    return (
        heliostat_position,
        th.tensor(facet_positions, dtype=dtype, device=device),
        th.tensor(facet_spans_n, dtype=dtype, device=device),
        th.tensor(facet_spans_e, dtype=dtype, device=device),
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


def heliostat_by_function(
        heliostat_function_cfg: CfgNode,
        device: th.device,
) -> HeliostatParams:
    cfg = heliostat_function_cfg

    # width = cfg.WIDTH / 2
    # height = cfg.HEIGHT / 2

    # X = th.linspace(-width, width, cfg.ROWS)
    # Y = th.linspace(-height, height, cfg.COLS)
    # X, Y = th.meshgrid(X, Y)

    columns: int = cfg.COLS
    column = th.arange(columns + 1, device=device)
    row = th.arange(cfg.ROWS + 1, device=device)

    X = (row/cfg.ROWS * cfg.HEIGHT) - (cfg.HEIGHT / 2)
    # Use points at centers of grid squares.
    X = X[:-1] + (X[1:] - X[:-1]) / 2
    X = th.tile(X, (columns,))
    X = X.reshape(cfg.ROWS, cfg.COLS)
    # heliostat y position
    Y = (column/columns * cfg.WIDTH) - (cfg.WIDTH / 2)
    # Use points at centers of grid squares.
    Y = Y[:-1] + (Y[1:] - Y[:-1]) / 2
    Y = th.tile(Y.unsqueeze(-1), (1, cfg.ROWS)).ravel()

    Y = Y.reshape(cfg.ROWS, cfg.COLS)

    reduction: float = cfg.REDUCTION_FACTOR
    fr: float = cfg.FREQUENCY
    if cfg.NAME == "sin":
        Z = th.sin(fr * X + fr * Y) / reduction  # + np.cos(Y)
    elif cfg.NAME == "sin+cos":
        Z = th.sin(X) / reduction + th.cos(Y) / reduction
    elif cfg.NAME == "random":
        Z = (
            th.sin(X) + th.sin(2 * Y) + th.sin(3 * X) + th.sin(4 * Y)
            + th.cos(Y) + th.cos(2 * X) + th.cos(3 * Y) + th.cos(4 * Y)
        ) / reduction
    else:
        raise ValueError("Z-Function not implemented in heliostat_models.py")

    Z_ideal = th.zeros_like(Z)
    h_ideal = th.stack((X, Y, Z_ideal)).T
    h_ideal = h_ideal.reshape(-1, h_ideal.shape[-1])
    h = th.stack((X, Y, Z)).T

    normal_vecs = th.zeros_like(h)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                origin = th.tensor([X[i, j], Y[i, j], Z[i, j]], device=device)
                next_row_vec = th.tensor(
                    [X[i, j + 1], Y[i, j + 1], Z[i, j + 1]], device=device)
                next_col_vec = th.tensor(
                    [X[i + 1, j], Y[i + 1, j], Z[i + 1, j]], device=device)
            except Exception:
                origin = th.tensor([X[i, j], Y[i, j], Z[i, j]], device=device)
                next_row_vec = th.tensor(
                    [X[i, j - 1], Y[i, j - 1], Z[i, j - 1]], device=device)
                next_col_vec = th.tensor(
                    [X[i - 1, j], Y[i - 1, j], Z[i - 1, j]], device=device)

            vec_1 = next_row_vec - origin

            vec_2 = next_col_vec - origin

            vec_1 = vec_1 / th.linalg.norm(vec_1)
            vec_2 = vec_2 / th.linalg.norm(vec_2)

            n = th.cross(vec_1, vec_2)
            # print(f"{n[2]:.10}")
            n = n / th.linalg.norm(n)
            # print(n)
            if n[2] < 0:
                n = -n
                # print("hello")
            normal_vecs[i, j] = n
            # print(normal_vecs)
            # exit()
    h = h.reshape(X.shape[0] * X.shape[1], -1)
    h_normal_vecs = normal_vecs.reshape(X.shape[0] * X.shape[1], -1)
    h_ideal_vecs = th.tile(
        th.tensor([0, 0, 1], dtype=h.dtype, device=device),
        (h_normal_vecs.shape[0], 1),
    )
    # print(h.shape)

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
        h_ideal,
        h_normal_vecs,
        h_ideal_vecs,
        cfg.HEIGHT,
        cfg.WIDTH,
        cfg.ROWS,
        cfg.COLS,
        params,
    )


def ideal_heliostat(
        ideal_configs: CfgNode,
        device: th.device,
) -> HeliostatParams:
    """Return an ideally shaped heliostat lying flat on the ground."""
    cfg = ideal_configs

    columns: int = cfg.COLS
    column = th.arange(columns + 1, device=device)
    row = th.arange(cfg.ROWS + 1, device=device)

    h_x = (row/cfg.ROWS * cfg.HEIGHT) - (cfg.HEIGHT / 2)
    # Use points at centers of grid squares.
    h_x = h_x[:-1] + (h_x[1:] - h_x[:-1]) / 2
    h_x = th.tile(h_x, (columns,))
    # heliostat y position
    h_y = (column/columns * cfg.WIDTH) - (cfg.WIDTH / 2)
    # Use points at centers of grid squares.
    h_y = h_y[:-1] + (h_y[1:] - h_y[:-1]) / 2
    h_y = th.tile(h_y.unsqueeze(-1), (1, cfg.ROWS)).ravel()
    h_z = th.zeros_like(h_x)

    h = th.stack(
        [h_x, h_y, h_z],
        -1,
    ).reshape(len(h_x), -1)

    normal_vector_direction = th.tensor(
        ideal_configs.NORMAL_VECS,
        dtype=h.dtype,
        device=device,
    )
    h_normal_vectors = th.tile(normal_vector_direction, (len(h), 1))

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


def _read_wavefront(
        filename: str,
        device: th.device,
) -> Tuple[
    Optional[str],
    torch.Tensor,
    List[torch.Tensor],
    List[torch.Tensor],
]:
    dtype = th.get_default_dtype()
    name = None
    vertices = []
    weights = []
    face_indices = []
    with open(filename, 'r') as obj_file:
        for line in obj_file:
            contents = line.split()

            if not contents:
                continue
            elif contents[0] == 'v':
                vertices.append(th.tensor(
                    list(map(float, contents[1:4])),
                    dtype=dtype,
                    device=device,
                ))
                weights.append(
                    th.tensor(
                        float(contents[4]),
                        dtype=dtype,
                        device=device,
                    )
                    if len(contents) > 4
                    else th.ones((0,), device=device)
                )
            elif contents[0] == 'f':
                if len(contents) > 4:
                    raise ValueError('can only load triangular faces')
                # face_indices.append(list(map(
                #     lambda x: int(x) - 1,
                #     contents[1:4],
                # )))
                indices = th.tensor(
                    list(map(
                        lambda x: int(x),
                        contents[1:4],
                    )),
                    dtype=th.long,
                    device=device,
                )
                indices -= 1
                assert indices.unique().numel() == indices.numel()
                face_indices.append(indices)
            elif contents[0] == 'o':
                if name is not None:
                    raise ValueError(
                        f'found multiple objects in {filename}; '
                        f'this is not supported'
                    )
                name = contents[1]
    return (name, th.stack(vertices), weights, face_indices)


# Read Wavefront OBJ files.
def other_objects(config: CfgNode, device: th.device) -> HeliostatParams:
    (name, vertices, weights, face_indices) = _read_wavefront(
        config.FILENAME, device)
    use_weighted_avg: bool = config.USE_WEIGHTED_AVG

    if use_weighted_avg:
        adjacent_surface_normals: List[List[torch.Tensor]] = [
            [] for i in range(len(vertices))
        ]
        face_areas: List[List[torch.Tensor]] = [
            [] for i in range(len(vertices))
        ]
    else:
        vertex_normals = th.zeros_like(vertices)
        num_adjacent_faces = th.zeros(
            vertices.shape[:-1] + (1,),
            device=device,
        )

    # face_centers = []
    # face_normals = []

    for triangle_indices in face_indices:
        a, b, c = vertices[triangle_indices]

        bma = b - a
        cma = c - a
        normal = th.cross(bma, cma)
        magnitude = th.linalg.norm(normal)
        normal /= magnitude
        # face_centers.append(a + bma / 2 + cma / 2)
        # face_normals.append(normal)

        if use_weighted_avg:
            for i in triangle_indices:
                adjacent_surface_normals[i].append(normal)
                area = magnitude / 2
                face_areas[i].append(area)
        else:
            vertex_normals[triangle_indices] += normal
            num_adjacent_faces[triangle_indices] += 1
    # import plotter
    # plotter.plot_heliostat(th.stack(face_centers), th.stack(face_normals))

    if use_weighted_avg:
        vertex_normals = th.empty_like(vertices)
        for (i, (normals_list, areas_list)) in enumerate(zip(
                adjacent_surface_normals,
                face_areas,
        )):
            normals = th.stack(normals_list)
            areas = th.stack(areas_list).unsqueeze(-1)
            weighted_avg = (normals * areas).sum(0) / areas.sum()
            vertex_normals[i] = weighted_avg
    else:
        vertex_normals /= num_adjacent_faces

    vertex_normals /= th.linalg.norm(vertex_normals)

    # FIXME Remove when `cfg.POSITION_ON_FIELD` is fixed.
    # Manually center the vertices as `cfg.POSITION_ON_FIELD` does not
    # work.
    x_min = vertices[:, 0].min()
    x_max = vertices[:, 0].max()
    y_min = vertices[:, 1].min()
    y_max = vertices[:, 1].max()
    height = y_max - y_min
    width = x_max - x_min

    vertices[:, 0] -= x_min + width / 2
    vertices[:, 1] -= y_min + height / 2

    # We add a bit more so we don't evaluate at the edges.
    height += 2e-6
    width += 2e-6

    height: float = float(height)
    width: float = float(width)

    # plotter.plot_heliostat(vertices, vertex_normals)
    rows = None
    cols = None
    params = {'name': name}
    return (
        get_position(config, vertices.dtype, device),
        *_sole_facet(height, width, vertices.dtype, device),
        vertices,
        vertices,
        vertex_normals,
        # TODO Implement Ideal Vecs
        vertex_normals,
        height,
        width,
        rows,
        cols,
        params,
    )


# Heliostat-specific functions
# ============================

# def rotate(h, hel_coordsystem, clockwise: bool):
#     r = rot_from_matrix(hel_coordsystem)
#     euler = rot_as_euler(r, 'xyx', degrees=True)
#     ele_degrees = 270-euler[2]

#     ele_radians = th.deg2rad(ele_degrees)
#     ele_axis = th.tensor([0, 1, 0], dtype=h.dtype, device=h.device)
#     ele_vector = ele_radians * ele_axis
#     if not clockwise:
#         ele_vector = -ele_vector
#     ele = rot_from_rotvec(ele_vector)

#     # TODO Max: re-add ax-offsets
#     azi_degrees = euler[1]-90
#     azi_radians = th.deg2rad(azi_degrees)
#     azi_axis = th.tensor([0, 0, 1], dtype=h.dtype, device=h.device)
#     azi_vector = azi_radians * azi_axis
#     if not clockwise:
#         azi_vector = -azi_vector
#     azi = rot_from_rotvec(azi_vector)

#     # darray with all heliostats (#heliostats, 3 coords)
#     h_rotated = rot_apply(azi, rot_apply(ele, h.unsqueeze(-1)))
#     return h_rotated.squeeze(-1)


def rotate(
        h: 'AbstractHeliostat',
        align_origin: throt.Transform3d,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # r = rot_from_matrix(hel_coordsystem)
    if hasattr(h, 'discrete_points_and_normals'):
        discrete_points, normals = \
            h.discrete_points_and_normals()  # type: ignore[attr-defined]
    else:
        discrete_points = h.discrete_points
        normals = h.normals

    rotated_points: torch.Tensor = align_origin.transform_points(
        discrete_points)
    rotated_normals: torch.Tensor = align_origin.transform_normals(normals)
    return rotated_points, rotated_normals


def heliostat_coord_system(
        Position: torch.Tensor,
        Sun: torch.Tensor,
        Aimpoint: torch.Tensor,
        disturbance_angles: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = Position.dtype
    device = Position.device

    pSun = Sun
    pPosition = Position
    pAimpoint = Aimpoint
    # print(pSun,pPosition,pAimpoint)
    # Berechnung Idealer Heliostat
    # 0. Iteration
    z = pAimpoint - pPosition
    z = z / th.linalg.norm(z)
    z = pSun + z
    z = z / th.linalg.norm(z)

    # Add heliostat rotation error/disturbance.
    disturbance_rot = (
        utils.rot_z_mat(disturbance_angles[2], dtype=dtype, device=device)
        @ utils.rot_y_mat(disturbance_angles[1], dtype=dtype, device=device)
        @ utils.rot_x_mat(disturbance_angles[0], dtype=dtype, device=device)
    )
    z = disturbance_rot @ z

    x = th.stack([
        -z[1],
        z[0],
        th.tensor(0, dtype=dtype, device=device),
    ])
    x = x / th.linalg.norm(x)
    y = th.cross(z, x)
    return x, y, z


class AbstractHeliostat:
    device: th.device
    position_on_field: torch.Tensor
    aim_point: torch.Tensor
    focus_point: Optional[torch.Tensor]
    disturbance_angles: List[torch.Tensor]
    cfg: CfgNode
    aligned_cls: Type['AbstractHeliostat']

    _facet_offsets: torch.Tensor
    _discrete_points: torch.Tensor
    _normals: torch.Tensor
    _discrete_points_ideal: torch.Tensor
    _normals_ideal: torch.Tensor

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError('do not construct an abstract class')

    def __len__(self) -> int:
        return len(self._discrete_points)

    def __call__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.discrete_points, self.get_ray_directions())

    @property
    def discrete_points(self) -> torch.Tensor:
        return self._discrete_points

    @property
    def normals(self) -> torch.Tensor:
        return self._normals

    def _make_facetted(self, values: torch.Tensor) -> List[torch.Tensor]:
        return list(th.tensor_split(values, self._facet_offsets[1:].cpu()))

    @property
    def facetted_discrete_points(self) -> List[torch.Tensor]:
        return self._make_facetted(self.discrete_points)

    @property
    def _facetted_discrete_points(self) -> List[torch.Tensor]:
        return self._make_facetted(self._discrete_points)

    @property
    def _facetted_discrete_points_ideal(self) -> List[torch.Tensor]:
        return self._make_facetted(self._discrete_points_ideal)

    @property
    def facetted_normals(self) -> List[torch.Tensor]:
        return self._make_facetted(self.normals)

    @property
    def _facetted_normals(self) -> List[torch.Tensor]:
        return self._make_facetted(self._normals)

    @property
    def _facetted_normals_ideal(self) -> List[torch.Tensor]:
        return self._make_facetted(self._normals_ideal)

    def get_ray_directions(self) -> torch.Tensor:
        raise NotImplementedError('please override `get_ray_directions`')

    def _optimizables(self) -> Dict[str, List[torch.Tensor]]:
        raise TypeError(
            self.__class__.__name__ + ' has no trainable parameters')

    def get_to_optimize(self) -> List[str]:
        raise TypeError(
            self.__class__.__name__ + ' has no trainable parameters')

    def set_to_optimize(
            self,
            new_to_optimize: List[str],
            *args: Any,
            **kwargs: Any,
    ) -> None:
        raise TypeError(
            self.__class__.__name__ + ' has no trainable parameters')

    def get_params(self) -> ParamGroups:
        raise TypeError(
            self.__class__.__name__ + ' has no trainable parameters')

    def to_dict(self) -> Dict[str, Any]:
        raise TypeError(
            'cannot convert ' + self.__class__.__name__ + ' to dictionary')

    @classmethod
    def from_dict(
            cls: Type[A],
            data: Dict[str, Any],
            *args: Any,
            **kwargs: Any,
    ) -> A:
        raise TypeError(
            'cannot construct ' + cls.__name__ + ' from dictionary')

    def align(
            self,
            sun_direction: torch.Tensor,
            aim_point: Optional[torch.Tensor] = None,
    ) -> 'AbstractHeliostat':
        assert hasattr(self, 'aligned_cls'), (
            'please assign the type of the aligned version of this '
            'heliostat to `aligned_cls`'
        )
        if aim_point is None:
            aim_point = self.aim_point
        return self.aligned_cls(self, sun_direction, aim_point)


class Heliostat(AbstractHeliostat):
    def __init__(
            self,
            heliostat_config: CfgNode,
            device: th.device,
            setup_params: bool = True,
            receiver_center: Union[torch.Tensor, List[float], None] = None,
    ) -> None:
        self.cfg = heliostat_config
        if not self.cfg.is_frozen():
            self.cfg = self.cfg.clone()
            self.cfg.freeze()
        self.device = device

        if (
                receiver_center is not None
                and not isinstance(receiver_center, th.Tensor)
        ):
            receiver_center = th.tensor(
                receiver_center,
                dtype=th.get_default_dtype(),
                device=device,
            )
        self._to_optimize: List[str] = self.cfg.TO_OPTIMIZE

        self._checked_dict = False
        self.params: Union[Dict[str, Any], CfgNode, None] = None

        self.load(receiver_center)
        if setup_params:
            self.setup_params()

    @staticmethod
    def select_heliostat_builder(cfg: CfgNode) -> Tuple[
            Callable[[CfgNode, th.device], HeliostatParams],
            CfgNode,
    ]:
        shape = cfg.SHAPE.lower()
        if shape == "ideal" or shape == "nurbs":
            return ideal_heliostat, cfg.IDEAL
        elif shape == "real":
            return real_heliostat, cfg.DEFLECT_DATA
        elif shape == "function":
            return heliostat_by_function, cfg.FUNCTION
        elif shape == "other":
            return other_objects, cfg.OTHER
        raise ValueError('unknown heliostat shape')

    def set_up_facets(
            self,
            facet_positions: torch.Tensor,
            facet_spans_n: torch.Tensor,
            facet_spans_e: torch.Tensor,
            discrete_points: torch.Tensor,
            discrete_points_ideal: torch.Tensor,
            normals: torch.Tensor,
            normals_ideal: torch.Tensor,
    ) -> None:
        if self._canting_enabled:
            self.focus_point = canting.get_focus_point(
                self._canting_cfg,
                self.aim_point,
                self.cfg.IDEAL.NORMAL_VECS,
                facet_positions.dtype,
                self.device,
            )
        else:
            self.focus_point = None

        facet_offsets: List[int] = []
        offset = 0
        facetted_discrete_points: List[torch.Tensor] = []
        facetted_discrete_points_ideal: List[torch.Tensor] = []
        facetted_normals: List[torch.Tensor] = []
        facetted_normals_ideal: List[torch.Tensor] = []

        for (position, span_n, span_e) in zip(
                facet_positions,
                facet_spans_n,
                facet_spans_e,
        ):
            facet_offsets.append(offset)

            # Select points on facet based on positions of ideal points.
            indices = facet_point_indices(
                discrete_points_ideal, position, span_n, span_e)
            facet_discrete_points = discrete_points[indices]
            facet_discrete_points_ideal = discrete_points_ideal[indices]
            facet_normals = normals[indices]
            facet_normals_ideal = normals_ideal[indices]

            if (
                    self._canting_enabled
                    and self._canting_algo is not CantingAlgorithm.ACTIVE
            ):
                (
                    facet_discrete_points,
                    facet_discrete_points_ideal,
                    facet_normals,
                    facet_normals_ideal,
                ) = canting.cant_facet_to_point(
                    self.position_on_field,
                    position,
                    self.focus_point,
                    facet_discrete_points,
                    facet_discrete_points_ideal,
                    facet_normals,
                    facet_normals_ideal,
                    self.cfg.IDEAL.NORMAL_VECS,
                )

            facetted_discrete_points.append(facet_discrete_points)
            facetted_discrete_points_ideal.append(facet_discrete_points_ideal)
            facetted_normals.append(facet_normals)
            facetted_normals_ideal.append(facet_normals_ideal)

            offset += len(facet_discrete_points)

        self._facet_offsets = th.tensor(facet_offsets, device=self.device)
        self.facet_positions = facet_positions
        self.facet_spans_n = facet_spans_n
        self.facet_spans_e = facet_spans_e
        self._discrete_points = th.cat(facetted_discrete_points, dim=0)
        self._discrete_points_ideal = th.cat(
            facetted_discrete_points_ideal, dim=0)
        self._normals = th.cat(facetted_normals, dim=0)
        self._normals_ideal = th.cat(facetted_normals_ideal, dim=0)

    def _get_aim_point(
            self,
            cfg: CfgNode,
            maybe_aim_point: Optional[torch.Tensor],
    ) -> torch.Tensor:
        cfg_aim_point: Optional[List[float]] = cfg.AIM_POINT
        if cfg_aim_point is not None:
            aim_point = th.tensor(
                cfg_aim_point,
                dtype=th.get_default_dtype(),
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
            th.deg2rad(th.tensor(
                angle,
                dtype=self.aim_point.dtype,
                device=self.device,
            ))
            for angle in angles
        ]

    def load(self, maybe_aim_point: Optional[torch.Tensor]) -> None:
        builder_fn, h_cfg = self.select_heliostat_builder(self.cfg)
        self._canting_cfg: CfgNode = h_cfg.FACETS.CANTING
        self._canting_enabled = canting.canting_enabled(self._canting_cfg)

        self._canting_algo = canting.get_algorithm(self._canting_cfg)
        self.aim_point = self._get_aim_point(
            h_cfg,
            maybe_aim_point,
        )
        # Radians
        self.disturbance_angles = self._get_disturbance_angles(h_cfg)

        (
            heliostat_position,
            facet_positions,
            facet_spans_n,
            facet_spans_e,
            heliostat,
            heliostat_ideal,
            heliostat_normals,
            heliostat_ideal_vecs,
            height,
            width,
            rows,
            cols,
            params,
        ) = builder_fn(h_cfg, self.device)

        self.position_on_field = heliostat_position
        self.set_up_facets(
            facet_positions,
            facet_spans_n,
            facet_spans_e,
            heliostat,
            heliostat_ideal,
            heliostat_normals,
            heliostat_ideal_vecs,
        )
        self.params = params
        self.height = height
        self.width = width
        self.rows = rows
        self.cols = cols

    @property
    def shape(self) -> Tuple[Optional[int], Optional[int]]:
        return (self.rows, self.cols)

    def _optimizables(self) -> Dict[str, List[torch.Tensor]]:
        return {'surface': [self._normals]}

    def optimizables(self) -> Dict[str, List[torch.Tensor]]:
        params = {
            'position': [self.position_on_field],
            'rotation_x': [self.disturbance_angles[0]],
            'rotation_y': [self.disturbance_angles[1]],
            'rotation_z': [self.disturbance_angles[2]],
        }
        params.update(self._optimizables())
        return params

    def get_to_optimize(self) -> List[str]:
        return self._to_optimize

    def set_to_optimize(  # type: ignore[override]
            self,
            new_to_optimize: List[str],
            setup_params: bool = True,
    ) -> None:
        # Reset old parameters.
        for group in self.optimizables().values():
            for param in group:
                param.requires_grad_(False)
                param.grad = None

        self._to_optimize = new_to_optimize
        if setup_params:
            self.setup_params()

    def setup_params(self) -> None:
        optimizables = self.optimizables()
        for name in self._to_optimize:
            if name not in optimizables:
                raise KeyError(f'{name} is not an optimizable variable')

            for param in optimizables[name]:
                param.requires_grad_(True)

    def get_params(self) -> ParamGroups:
        opt_params = []
        optimizables = self.optimizables()
        for name in self._to_optimize:
            if name not in optimizables:
                raise KeyError(f'{name} is not an optimizable variable')
            opt_params.append({'params': optimizables[name], 'name': name})
        return opt_params

    def get_ray_directions(self) -> torch.Tensor:
        raise NotImplementedError('Heliostat has to be aligned first')

    def step(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property  # type: ignore[misc]
    @functools.lru_cache()
    def dict_keys(self) -> Set[str]:
        """All keys we assume in the dictionary returned by `_to_dict`."""
        return {
            'heliostat_points',
            'heliostat_normals',

            'position_on_field',
            'disturbance_rotation_angles_rad',

            'config',
            'params',
            'to_optimize',
        }

    def _check_dict(self, data: Dict[str, Any]) -> None:
        """Check whether the given data dictionary has the correct keys;
        only do this once to save some cycles.

        The reason we can safely do it just once is that all keys must
        always be the same due to read-only constraints of the
        heliostat's properties.
        """
        if self._checked_dict:
            return
        assert data.keys() == self.dict_keys
        self._checked_dict = True

    @functools.lru_cache()
    def _fixed_dict(self) -> Dict[str, Any]:
        """The part of the heliostat's configuration that does not change."""
        data = {
            'heliostat_points': self._discrete_points,

            'config': self.cfg,
            'params': copy.deepcopy(self.params),
        }
        return data

    def to_dict(self) -> Dict[str, Any]:
        data = self._to_dict()
        self._check_dict(data)
        return data

    def _to_dict(self) -> Dict[str, Any]:
        data = self._fixed_dict()
        data.update({
            'heliostat_normals': self._normals.clone(),

            'position_on_field': self.position_on_field.clone(),
            'disturbance_rotation_angles_rad': [
                angle.clone()
                for angle in self.disturbance_angles
            ],

            'to_optimize': self._to_optimize.copy(),
        })
        return data

    @classmethod
    def from_dict(  # type: ignore[override]
            cls: Type[C],
            data: Dict[str, Any],
            device: th.device,
            config: Optional[CfgNode] = None,
            receiver_center: Union[torch.Tensor, List[float], None] = None,
            # Wether to disregard what standard initialization did and
            # load all data we have.
            restore_strictly: bool = True,
            setup_params: bool = True,
    ) -> C:
        if config is None:
            config = data['config']

        self = cls(
            config,
            device,
            receiver_center=receiver_center,
            setup_params=False,
        )
        self._from_dict(data, restore_strictly)
        if setup_params:
            self.setup_params()
        return self

    def _from_dict(self, data: Dict[str, Any], restore_strictly: bool) -> None:
        self._normals = data['heliostat_normals']
        self.position_on_field = data['position_on_field']
        self.disturbance_angles = data['disturbance_rotation_angles_rad']

        if restore_strictly:
            self._discrete_points = data['heliostat_points']
            self.params = data['params']
            self._to_optimize = data['to_optimize']


class AlignedHeliostat(AbstractHeliostat):
    def __init__(
            self,
            heliostat: Heliostat,
            sun_direction: torch.Tensor,
            aim_point: torch.Tensor,
            align_points: bool = True,
    ) -> None:
        assert type(self) == heliostat.aligned_cls, \
            'aligned heliostat class does not match'
        if not hasattr(heliostat, '_discrete_points'):
            raise ValueError('Heliostat has to be loaded first')

        self._heliostat = heliostat

        from_sun = -sun_direction
        self.from_sun = from_sun.unsqueeze(0)

        self.alignment = th.stack(heliostat_coord_system(
            self._heliostat.position_on_field,
            sun_direction,
            aim_point,
            self._heliostat.disturbance_angles,
        ))
        self.align_origin = throt.Rotate(
            self.alignment, dtype=self.alignment.dtype)

        if align_points:
            self._align()

    def _align(self) -> None:
        hel_rotated, normal_vectors_rotated = rotate(
            self._heliostat, self.align_origin)
        # TODO Add Translation in rotate function
        hel_rotated_in_field = hel_rotated + self._heliostat.position_on_field
        normal_vectors_rotated = (
            normal_vectors_rotated
            / th.linalg.norm(normal_vectors_rotated, dim=-1).unsqueeze(-1)
        )

        self._discrete_points = hel_rotated_in_field
        self._normals = normal_vectors_rotated

    def get_ray_directions(self) -> torch.Tensor:
        return reflect_rays_(self.from_sun, self.normals)


Heliostat.aligned_cls = AlignedHeliostat
