import copy
import functools
import struct
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import pytorch3d.transforms as throt
import torch
import torch as th
from yacs.config import CfgNode

import bpro_loader
import utils

HeliostatParams = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
    float,
    Optional[int],
    Optional[int],
    Optional[Dict[str, Any]],
]
A = TypeVar('A', bound='AbstractHeliostat')
C = TypeVar('C', bound='Heliostat')


def reflect_rays_(rays: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    return rays - 2 * utils.batch_dot(rays, normals) * normals


def reflect_rays(rays: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    normals = normals / th.linalg.norm(normals, dim=-1).unsqueeze(-1)
    return reflect_rays_(rays, normals)


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
        positions,
        directions,
        ideal_normal_vecs,
        width,
        height,
    ) = bpro_loader.load_bpro(
        cfg.FILENAME,
        concentratorHeader_struct,
        facetHeader_struct,
        ray_struct,
    )

    h_normal_vecs = []
    h_ideal_vecs = []
    h = []
    zs = []
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
        zs.append(utils.deflec_facet_zs_many(
            h[-1],
            h_normal_vecs[-1],
            num_samples=4,
        ))

    h_normal_vecs: torch.Tensor = th.cat(h_normal_vecs, dim=0)
    h_ideal_vecs: torch.Tensor = th.cat(h_ideal_vecs, dim=0)
    h_ideal = th.cat(h, dim=0)

    zs: torch.Tensor = th.cat(zs, dim=0)
    h: torch.Tensor = h_ideal.clone()
    h[:, -1] += zs
    
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize =(14, 9))
    ax = plt.axes(projection ='3d') 
    h[:,2] = h[:,2]-h_ideal[:,2]
    h = h.detach().cpu()
    my_cmap = plt.get_cmap('hot')
    ax.plot_trisurf(h[:,0],h[:,1],h[:,2], cmap =my_cmap)
    plt.show()
    exit()

    # print(h_ideal_vecs)
    rows = None
    cols = None
    params = None
    return (
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
    h = th.stack((X, Y, Z)).T

    normal_vecs = th.zeros_like(h)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                origin = th.tensor([X[i, j], Y[i, j], Z[i, j]])
                next_row_vec = th.tensor(
                    [X[i, j + 1], Y[i, j + 1], Z[i, j + 1]])
                next_col_vec = th.tensor(
                    [X[i + 1, j], Y[i + 1, j], Z[i + 1, j]])
            except Exception:
                origin = th.tensor([X[i, j], Y[i, j], Z[i, j]])
                next_row_vec = th.tensor(
                    [X[i, j - 1], Y[i, j - 1], Z[i, j - 1]])
                next_col_vec = th.tensor(
                    [X[i - 1, j], Y[i - 1, j], Z[i - 1, j]])

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
    h = h.reshape(X.shape[0] * X.shape[1], -1).to(device)
    h_normal_vecs = normal_vecs.reshape(X.shape[0] * X.shape[1], -1).to(device)
    h_ideal_vecs = th.tile(
        th.tensor([0, 0, 1]),
        (h_normal_vecs.shape[0], 1),
    ).to(device)

    params = None
    return (
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
        dtype=th.get_default_dtype(),
        device=device,
    )
    h_normal_vectors = th.tile(normal_vector_direction, (len(h), 1))
    params = None
    return (
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

    # plotter.plot_heliostat(vertices, vertex_normals)
    rows = None
    cols = None
    params = {'name': name}
    return (
        vertices,
        vertices,
        vertex_normals,
        # TODO Implement Ideal Vecs
        vertex_normals,
        float(height),
        float(width),
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    x = th.tensor(
        [-z[1], z[0], 0],
        dtype=Position.dtype,
        device=Position.device,
    )
    x = x / th.linalg.norm(x)
    y = th.cross(z, x)
    return x, y, z


class AbstractHeliostat:
    device: th.device
    position_on_field: torch.Tensor
    cfg: CfgNode

    _discrete_points: torch.Tensor
    _ideal_discrete_points: torch.Tensor
    _normals: torch.Tensor

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError('do not construct an abstract class')

    def __len__(self) -> int:
        return len(self._discrete_points)

    def __call__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.discrete_points, self.get_ray_directions())

    @property
    def discrete_points(self) -> torch.Tensor:
        return self._discrete_points

    @property
    def ideal_discrete_points(self) -> torch.Tensor:
        return self._ideal_discrete_points

    @property
    def normals(self) -> torch.Tensor:
        return self._normals

    def get_ray_directions(self) -> torch.Tensor:
        raise NotImplementedError('please override `get_ray_directions`')

    def get_params(self) -> List[torch.Tensor]:
        raise TypeError(
            self.__class__.__name__ + ' has no trainable parameters')

    def to_dict(self) -> Dict[str, Any]:
        raise TypeError(
            'cannot convert ' + self.__class__.__name__ + ' to dictionary')

    @classmethod
    def from_dict(cls: Type[A], data: Dict[str, Any], *args, **kwargs) -> A:
        raise TypeError(
            'cannot construct ' + cls.__name__ + ' from dictionary')

    def align(
            self,
            sun_direction: torch.Tensor,
            receiver_center: torch.Tensor,
    ) -> 'AbstractHeliostat':
        raise NotImplementedError('please override `align`')


class Heliostat(AbstractHeliostat):
    def __init__(
            self,
            heliostat_config: CfgNode,
            device: th.device,
            setup_params: bool = True,
    ) -> None:
        self.cfg = heliostat_config
        if not self.cfg.is_frozen():
            self.cfg = self.cfg.clone()
            self.cfg.freeze()
        self.device = device
        self.position_on_field = th.tensor(
            self.cfg.POSITION_ON_FIELD,
            dtype=th.get_default_dtype(),
            device=self.device,
        )

        self._checked_dict = False
        self.params: Union[Dict[str, Any], CfgNode, None] = None

        self.load()
        if setup_params:
            self.setup_params()

    def load(self) -> None:

        cfg = self.cfg
        shape = cfg.SHAPE.lower()
        if shape == "ideal" or shape == "nurbs":
            heliostat_properties = ideal_heliostat(cfg.IDEAL, self.device)
        elif shape == "real":
            heliostat_properties = real_heliostat(
                cfg.DEFLECT_DATA, self.device)
        elif shape == "function":
            heliostat_properties = heliostat_by_function(
                cfg.FUNCTION, self.device)
        elif shape == "other":
            heliostat_properties = other_objects(cfg.OTHER, self.device)
        else:
            raise ValueError('unknown heliostat shape')

        (
            heliostat,
            heliostat_ideal,
            heliostat_normals,
            heliostat_ideal_vecs,
            height,
            width,
            rows,
            cols,
            params,
        ) = heliostat_properties
        self._discrete_points = heliostat
        self._ideal_discrete_points = heliostat_ideal
        self._normals = heliostat_normals
        self._normals_ideal = heliostat_ideal_vecs
        self.params = params
        self.height = height
        self.width = width
        self.rows = rows
        self.cols = cols

    @property
    def shape(self) -> Tuple[Optional[int], Optional[int]]:
        return (self.rows, self.cols)

    def align(
            self,
            sun_direction: torch.Tensor,
            receiver_center: torch.Tensor,
    ) -> 'AlignedHeliostat':
        return AlignedHeliostat(self, sun_direction, receiver_center)

    def setup_params(self) -> None:
        self._normals.requires_grad_(True)

    def get_params(self) -> List[torch.Tensor]:
        opt_params = [self._normals]
        return opt_params

    def get_ray_directions(self) -> torch.Tensor:
        raise NotImplementedError('Heliostat has to be aligned first')

    def step(self, *args, **kwargs) -> None:
        pass

    @property  # type: ignore[misc]
    @functools.lru_cache()
    def dict_keys(self) -> Set[str]:
        """All keys we assume in the dictionary returned by `_to_dict`."""
        return {
            'heliostat_points',
            'heliostat_normals',

            'config',
            'params',
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
        })
        return data

    @classmethod
    def from_dict(  # type: ignore[override]
            cls: Type[C],
            data: Dict[str, Any],
            device: th.device,
            config: Optional[CfgNode] = None,
            restore_strictly: bool = True,
            setup_params: bool = True,
    ) -> C:
        if config is None:
            config = data['config']
        self = cls(config, device, setup_params=False)
        self._from_dict(data, restore_strictly)
        if setup_params:
            self.setup_params()
        return self

    def _from_dict(self, data: Dict[str, Any], restore_strictly: bool) -> None:
        self._normals = data['heliostat_normals']

        if restore_strictly:
            self._discrete_points = data['heliostat_points']
            self.params = data['params']


class AlignedHeliostat(AbstractHeliostat):
    def __init__(
            self,
            heliostat: Heliostat,
            sun_direction: torch.Tensor,
            receiver_center: torch.Tensor,
            align_points: bool = True,
    ) -> None:
        if not hasattr(heliostat, '_discrete_points'):
            raise ValueError('Heliostat has to be loaded first')
        if isinstance(heliostat, AlignedHeliostat):
            raise ValueError('Heliostat is already aligned')

        self._heliostat = heliostat

        from_sun = -sun_direction
        self.from_sun = from_sun.unsqueeze(0)

        self.alignment = th.stack(heliostat_coord_system(
            self._heliostat.position_on_field,
            sun_direction,
            receiver_center,
        ))
        self.align_origin = throt.Rotate(self.alignment)

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
