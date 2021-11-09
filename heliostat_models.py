import copy
from enum import Enum
import functools
import struct

import torch as th

from rotation import rot_apply, rot_as_euler, rot_from_matrix, rot_from_rotvec
import utils


class AlignmentState(Enum):
    UNINITIALIZED = None
    ON_GROUND = 'OnGround'
    ALIGNED = 'Aligned'


def reflect_rays_(rays, normals):
    return rays - 2 * utils.batch_dot(rays, normals) * normals


def reflect_rays(rays, normals):
    normals = normals / th.linalg.norm(normals, dim=-1).unsqueeze(-1)
    return reflect_rays_(rays, normals)


# Heliostat Models
# ================

def real_heliostat(real_configs, device):
    """Return a heliostat loaded from deflectometric data."""
    cfg = real_configs
    dtype = th.get_default_dtype()
    concentratorHeader_struct_len = struct.calcsize(
        cfg.CONCENTRATORHEADER_STRUCT_FMT)
    facetHeader_struct_len = struct.calcsize(cfg.FACETHEADER_STRUCT_FMT)
    ray_struct_len = struct.calcsize(cfg.RAY_STRUCT_FMT)

    positions = []
    directions = []
    # powers = []
    with open(cfg.FILENAME, "rb") as file:
        byte_data = file.read(concentratorHeader_struct_len)
        concentratorHeader_data = struct.Struct(
            cfg.CONCENTRATORHEADER_STRUCT_FMT,
        ).unpack_from(byte_data)
        print("READING bpro filename: " + cfg.FILENAME)

        # hel_pos = concentratorHeader_data[0:3]
        width_height = concentratorHeader_data[3:5]
        # offsets = concentratorHeader_data[7:9]
        n_xy = concentratorHeader_data[5:7]

        nFacets = n_xy[0] * n_xy[1]
        # nFacets =1
        for f in range(nFacets):
            byte_data = file.read(facetHeader_struct_len)
            facetHeader_data = struct.Struct(
                cfg.FACETHEADER_STRUCT_FMT,
            ).unpack_from(byte_data)

            # 0 for square, 1 for round 2 triangle, ...
            # facetshape = facetHeader_data[0]
            # facet_pos = facetHeader_data[1:4]
            # facet_vec_x = facetHeader_data[4:7]
            # facet_vec_y = facetHeader_data[7:10]
            n_rays = facetHeader_data[10]

            for r in range(n_rays):
                byte_data = file.read(ray_struct_len)
                ray_data = struct.Struct(
                    cfg.RAY_STRUCT_FMT,
                ).unpack_from(byte_data)

                positions.append([ray_data[0], ray_data[1], ray_data[2]])
                directions.append([ray_data[3], ray_data[4], ray_data[5]])
                # powers.append(ray_data[6])

        h_normal_vecs = th.tensor(
            directions[0::int(len(directions)/cfg.TAKE_N_VECTORS)],
            dtype=dtype,
            device=device,
        )
        h = th.tensor(
            positions[0::int(len(positions)/cfg.TAKE_N_VECTORS)],
            dtype=dtype,
            device=device,
        )
        params = None
        return (
            h,
            h_normal_vecs,
            width_height[1],
            width_height[0],
            params,
            # powers,
        )


def heliostat_by_function(heliostat_function_cfg, device):
    cfg = heliostat_function_cfg

    # width = cfg.WIDTH / 2
    # height = cfg.HEIGHT / 2

    # X = th.linspace(-width, width, cfg.ROWS)
    # Y = th.linspace(-height, height, cfg.COLS)
    # X, Y = th.meshgrid(X, Y)

    columns = cfg.COLS
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

    reduction = cfg.REDUCTION_FACTOR
    fr = cfg.FREQUENCY
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

    stacked = th.stack((X, Y, Z)).T

    normal_vecs = th.zeros_like(stacked)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                origin = th.tensor([X[i, j], Y[i, j], Z[i, j]])  # .squeeze(0)
                next_row_vec = th.tensor(
                    [X[i, j + 1], Y[i, j + 1], Z[i, j + 1]])  # .squeeze(0)
                next_col_vec = th.tensor(
                    [X[i + 1, j], Y[i + 1, j], Z[i + 1, j]])  # .squeeze(0)
            except Exception:
                origin = th.tensor([X[i, j], Y[i, j], Z[i, j]])  # .squeeze(0)
                next_row_vec = th.tensor(
                    [X[i, j - 1], Y[i, j - 1], Z[i, j - 1]])  # .squeeze(0)
                next_col_vec = th.tensor(
                    [X[i - 1, j], Y[i - 1, j], Z[i - 1, j]])  # .squeeze(0)
            vec_1 = next_row_vec - origin

            vec_2 = next_col_vec - origin

            vec_1 = vec_1 / th.linalg.norm(vec_1)
            vec_2 = vec_2 / th.linalg.norm(vec_2)

            n = th.cross(vec_1, vec_2)
            n = n / th.linalg.norm(n)
            if n[2] < 0:
                n = -n
            normal_vecs[i, j] = n
    h = stacked.reshape(X.shape[0] * X.shape[1], -1).to(device)
    h_normal_vecs = normal_vecs.reshape(X.shape[0] * X.shape[1], -1).to(device)
    params = None

    return h, h_normal_vecs, cfg.HEIGHT, cfg.WIDTH, params


def ideal_heliostat(ideal_configs, device):
    """Return an ideally shaped heliostat lying flat on the ground."""
    cfg = ideal_configs

    columns = cfg.COLS
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
        [0, 0, 1],
        dtype=th.get_default_dtype(),
        device=device,
    )
    h_normal_vectors = th.tile(normal_vector_direction, (len(h), 1))
    params = None
    return h, h_normal_vectors, cfg.HEIGHT, cfg.WIDTH, params


def other_objects(config, device):  # Read Wavefront OBJ files.
    dtype = th.get_default_dtype()
    name = None
    vertices = []
    weights = []
    face_indices = []
    with open(config.FILENAME, 'r') as obj_file:
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
                        f'found multiple objects in {config.FILENAME}; '
                        f'this is not supported'
                    )
                name = contents[1]

    use_weighted_avg = config.USE_WEIGHTED_AVG

    vertices = th.stack(vertices)
    if use_weighted_avg:
        adjacent_surface_normals = [[] for i in range(len(vertices))]
        face_areas = [[] for i in range(len(vertices))]
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
        for (i, (normals, areas)) in enumerate(zip(
                adjacent_surface_normals,
                face_areas,
        )):
            normals = th.stack(normals)
            areas = th.stack(areas).unsqueeze(-1)
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
    params = {'name': name}
    return vertices, vertex_normals, height, width, params


# Heliostat-specific functions
# ============================

def rotate(h, hel_coordsystem, clockwise: bool):
    r = rot_from_matrix(hel_coordsystem)
    euler = rot_as_euler(r, 'xyx', degrees=True)
    ele_degrees = 270-euler[2]

    ele_radians = th.deg2rad(ele_degrees)
    ele_axis = th.tensor([0, 1, 0], dtype=h.dtype, device=h.device)
    ele_vector = ele_radians * ele_axis
    if not clockwise:
        ele_vector = -ele_vector
    ele = rot_from_rotvec(ele_vector)

    # TODO Max: re-add ax-offsets
    azi_degrees = euler[1]-90
    azi_radians = th.deg2rad(azi_degrees)
    azi_axis = th.tensor([0, 0, 1], dtype=h.dtype, device=h.device)
    azi_vector = azi_radians * azi_axis
    if not clockwise:
        azi_vector = -azi_vector
    azi = rot_from_rotvec(azi_vector)

    # darray with all heliostats (#heliostats, 3 coords)
    h_rotated = rot_apply(azi, rot_apply(ele, h.unsqueeze(-1)))
    return h_rotated.squeeze(-1)


def heliostat_coord_system(Position, Sun, Aimpoint):
    pSun = Sun
    pPosition = Position
    pAimpoint = Aimpoint

    # Berechnung Idealer Heliostat
    # 0. Iteration
    z = pAimpoint - pPosition
    z = z / th.linalg.norm(z)
    z = pSun + z
    z = z / th.linalg.norm(z)

    x = th.tensor(
        [z[1], -z[0], 0],
        dtype=Position.dtype,
        device=Position.device,
    )
    x = x / th.linalg.norm(x)
    y = th.cross(z, x)

    return x, y, z


class Heliostat(object):
    def __init__(self, heliostat_config, device):
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
        self.state = AlignmentState.UNINITIALIZED
        self.from_sun = None
        self.alignment = None
        self._discrete_points_orig = None
        self._normals_orig = None
        self._discrete_points_aligned = None
        self._normals_aligned = None
        self.params = None

        self.load()

    def load(self):

        cfg = self.cfg
        shape = cfg.SHAPE.lower()
        if shape == "ideal":
            heliostat, heliostat_normals, height, width, params = \
                ideal_heliostat(cfg.IDEAL, self.device)
        elif shape == "real":
            heliostat, heliostat_normals, height, width, params = \
                real_heliostat(cfg.DEFLECT_DATA, self.device)
        elif shape == "function":
            heliostat, heliostat_normals, height, width, params = \
                heliostat_by_function(cfg.FUNCTION, self.device)
        elif shape == "other":
            heliostat, heliostat_normals, height, width, params = \
                other_objects(cfg.OTHER, self.device)

        self._discrete_points_orig = heliostat
        self._normals_orig = heliostat_normals
        self.params = params
        self.state = AlignmentState.ON_GROUND
        self.height = height
        self.width = width

    def __call__(self):
        return (self.discrete_points, self.get_ray_directions())

    @property
    def shape(self):
        return (self.rows, self.cols)

    def align(self, sun_origin, receiver_center, verbose=True):
        if self.discrete_points is None:
            raise ValueError('Heliostat has to be loaded first')
        if self.state is AlignmentState.ALIGNED:
            raise ValueError('Heliostat is already aligned')

        # TODO Max: fix for other aimpoints
        # TODO Evtl auf H.Discrete Points umstellen
        from_sun = self.position_on_field - sun_origin
        from_sun /= th.linalg.norm(from_sun)
        self.from_sun = from_sun.unsqueeze(0)

        self.alignment = th.stack(heliostat_coord_system(
            self.position_on_field,
            sun_origin,
            receiver_center,
        ))

        self._align()
        self.state = AlignmentState.ALIGNED

    def _align(self):
        hel_rotated = rotate(
            self.discrete_points, self.alignment, clockwise=True)
        hel_rotated_in_field = hel_rotated + self.position_on_field

        normal_vectors_rotated = rotate(
            self.normals, self.alignment, clockwise=True)
        normal_vectors_rotated = (
            normal_vectors_rotated
            / th.linalg.norm(normal_vectors_rotated, dim=-1).unsqueeze(-1)
        )

        self._discrete_points_aligned = hel_rotated_in_field
        self._normals_aligned = normal_vectors_rotated

    def align_reverse(self):
        self._align_reverse()
        self.state = AlignmentState.ON_GROUND

    def _align_reverse(self):
        pass

    def reset_cache(self):
        pass

    @property
    def discrete_points(self):
        if self.state is AlignmentState.ON_GROUND:
            return self._discrete_points_orig
        elif self.state is AlignmentState.ALIGNED:
            return self._discrete_points_aligned
        else:
            raise ValueError(f'unknown state {self.state}')

    @property
    def normals(self):
        if self.state is AlignmentState.ON_GROUND:
            return self._normals_orig
        elif self.state is AlignmentState.ALIGNED:
            return self._normals_aligned
        else:
            raise ValueError(f'unknown state {self.state}')

    def setup_params(self):
        self._normals_orig.requires_grad_(True)

        opt_params = [self._normals_orig]
        return opt_params

    def get_ray_directions(self):
        if self.alignment is None:
            raise ValueError('Heliostat has to be aligned first')

        return reflect_rays_(self.from_sun, self.normals)

    def step(self, *args, **kwargs):
        pass

    @property
    @functools.lru_cache()
    def dict_keys(self):
        """All keys we assume in the dictionary returned by `_to_dict`."""
        return {
            'heliostat_points',
            'heliostat_normals',

            'config',
            'params',
        }

    def _check_dict(self, data):
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
    def _fixed_dict(self):
        """The part of the heliostat's configuration that does not change."""
        data = {
            'heliostat_points': self._discrete_points_orig,

            'config': self.cfg,
            'params': copy.deepcopy(self.params),
        }
        return data

    def to_dict(self):
        if self.state is not AlignmentState.ON_GROUND:
            print(
                'Warning; saving aligned heliostat! It is recommended to '
                '`align_reverse` the heliostat beforehand!'
            )
        data = self._to_dict()
        self._check_dict(data)
        return data

    def _to_dict(self):
        data = self._fixed_dict()
        data.update({
            'heliostat_normals': self._normals_orig.clone(),
        })
        return data

    @classmethod
    def from_dict(cls, data, device, config=None, restore_strictly=True):
        if config is None:
            config = data['config']
        self = cls(config, device)
        self._from_dict(data, restore_strictly)
        return self

    def _from_dict(self, data, restore_strictly):
        self._normals_orig = data['heliostat_normals']

        if restore_strictly:
            self._discrete_points_orig = data['heliostat_points']
            self.params = data['params']
