import torch as th
import struct
import sys
from rotation import rot_apply, rot_as_euler, rot_from_matrix, rot_from_rotvec

##### Heliostat Models #####
def real_heliostat(real_configs, device): # For heliostat with deflectometric data
    cfg = real_configs
    concentratorHeader_struct_len = struct.calcsize(cfg.CONCENTRATORHEADER_STRUCT_FMT)
    facetHeader_struct_len = struct.calcsize(cfg.FACETHEADER_STRUCT_FMT)
    ray_struct_len = struct.calcsize(cfg.RAY_STRUCT_FMT)

    positions= []
    directions = []
    # powers = []
    with open(cfg.FILENAME, "rb") as file:
        byte_data = file.read(concentratorHeader_struct_len)
        concentratorHeader_data = struct.Struct(cfg.CONCENTRATORHEADER_STRUCT_FMT).unpack_from(byte_data)
        print("READING bpro filename: " + cfg.FILENAME)

        # hel_pos = concentratorHeader_data[0:3]
        width_height = concentratorHeader_data[3:5]
        #offsets = concentratorHeader_data[7:9]
        n_xy = concentratorHeader_data[5:7]


        nFacets = n_xy[0] * n_xy[1]
        for f in range(nFacets):
        # for f in range(1):
            byte_data = file.read(facetHeader_struct_len)
            facetHeader_data = struct.Struct(cfg.FACETHEADER_STRUCT_FMT).unpack_from(byte_data)

            #facetshape = facetHeader_data[0] # 0 for square, 1 for round 2 triangle ....
            #facet_pos = facetHeader_data[1:4]
            #facet_vec_x = facetHeader_data[4:7]
            #facet_vec_y = facetHeader_data[7:10]
            n_rays = facetHeader_data[10]

            for r in range(n_rays):
                byte_data = file.read(ray_struct_len)
                ray_data = struct.Struct(cfg.RAY_STRUCT_FMT).unpack_from(byte_data)

                positions.append([ray_data[0],ray_data[1],ray_data[2]])
                directions.append([ray_data[3],ray_data[4],ray_data[5]])
                # powers.append(ray_data[6])

        h_normal_vecs = th.tensor(directions[0::int(len(directions)/cfg.TAKE_N_VECTORS)], device=device)
        h = th.tensor(positions[0::int(len(positions)/cfg.TAKE_N_VECTORS)], device = device)
        params = {"width_height": width_height}
        return h, h_normal_vecs, params# width_height #,powers

def ideal_heliostat(ideal_configs, device): # For ideal shaped heliostat
    cfg = ideal_configs
    # points_on_hel   = rows*cols # reflection points on hel
    points_on_hel   = th.tensor(cfg.ROWS * cfg.COLS, dtype=th.float32, device=device)
    # target_hel_origin      = define_heliostat(cfg.HEIGHT, cfg.WIDTH, rows, points_on_hel, device)

    columns = int(points_on_hel)//cfg.ROWS
    column = th.arange(columns, device=device)
    row = th.arange(cfg.ROWS, device=device)

    h_x = (row/(cfg.ROWS-1)*cfg.HEIGHT)-(cfg.HEIGHT/2)
    h_x = th.tile(h_x, (columns,))
    h_y = (column/(columns-1)*cfg.WIDTH)-(cfg.WIDTH/2) #heliostat y position
    h_y = th.tile(h_y.unsqueeze(-1), (1, columns)).ravel()
    h_z = th.zeros_like(h_x)

    h = th.hstack(list(map(lambda t: t.unsqueeze(-1), [h_x, h_y, h_z]))).reshape(len(h_x), -1)



    normal_vector_direction   = th.tensor([0,0,1], dtype=th.float32, device=device)
    h_normal_vectors = th.tile(normal_vector_direction, (len(h), 1))
    params = None
    return h, h_normal_vectors, params

def other_objects(): # For later
    return None
##### Heliostat specific functions #####
def rotate(h,hel_coordsystem, clockwise):
    r = rot_from_matrix(hel_coordsystem)
    euler = rot_as_euler(r, 'xyx', degrees = True)
    ele_degrees = 270-euler[2]

    ele_radians = th.deg2rad(ele_degrees)
    ele_axis = th.tensor([0, 1, 0], dtype=th.float32, device=h.device)
    ele_vector = ele_radians * ele_axis
    if not clockwise:
        ele_vector = -ele_vector
    ele = rot_from_rotvec(ele_vector)

    # TODO Max: re-add ax-offsets
    azi_degrees = euler[1]-90
    azi_radians = th.deg2rad(azi_degrees)
    azi_axis = th.tensor([0, 0, 1], dtype=th.float32, device=h.device)
    azi_vector = azi_radians * azi_axis
    if not clockwise:
        azi_vector = -azi_vector
    azi = rot_from_rotvec(azi_vector)

    h_rotated = rot_apply(azi, rot_apply(ele, h.unsqueeze(-1))) # darray with all heliostats (#heliostats, 3 coords)
    return h_rotated.squeeze(-1)

def heliostat_coord_system(Position, Sun, Aimpoint, verbose=True):

    pSun = Sun
    pPosition = Position
    pAimpoint = Aimpoint
    if verbose:
        print("Sun",pSun)
        print("Position", pPosition)
        print("Aimpoint", pAimpoint)


#Berechnung Idealer Heliostat
#0. Iteration
    z = pAimpoint - pPosition
    z = z/th.linalg.norm(z)
    z = pSun + z
    z = z/th.linalg.norm(z)

    x = th.tensor([z[1],-z[0], 0], dtype=th.float32, device=Position.device)
    x = x/th.linalg.norm(x)
    y = th.cross(z,x)


    return x,y,z

#### Heliostat Class #####
class Heliostat(object):
    def __init__(self, heliostat_config, device):
        self.cfg = heliostat_config
        self.device = device

        self.position_on_field   = th.tensor(self.cfg.POSITION_ON_FIELD, device = self.device)

        self.state = None
        self.alignment = None
        self._discrete_points_orig = None
        self._normals_orig = None
        self._discrete_points_aligned = None
        self._normals_aligned = None
        self.params = None

        self.load()

    def load(self):

        cfg = self.cfg
        if cfg.SHAPE == "Ideal":
            heliostat, heliostat_normals, params = ideal_heliostat(cfg.IDEAL, self.device)
        elif cfg.SHAPE == "Real":
            heliostat, heliostat_normals, params = real_heliostat(cfg.REAL, self.device)
        elif cfg.SHAPE == "Other":
            heliostat, heliostat_normals, params = other_objects(cfg.OTHER, self.device)

        self._discrete_points_orig = heliostat
        self._normals_orig = heliostat_normals
        self.params = params
        self.state = "OnGround"

    def align(self, sun_origin, receiver_center, verbose=True):
        if self.discrete_points is None:
            raise ValueError('Heliostat has to be loaded first')

        #TODO Max: fix for other aimpoints; need this to work inversely as well

        self.alignment = th.stack(heliostat_coord_system(
            self.position_on_field,
            sun_origin,
            receiver_center,
            verbose=verbose,
        ))

        hel_rotated     = rotate(self.discrete_points ,self.alignment, clockwise = True)
        hel_rotated_in_field    = hel_rotated+ self.position_on_field

        normal_vectors_rotated = rotate(self.normals, self.alignment, clockwise = True)
        normal_vectors_rotated /= normal_vectors_rotated.norm(dim=-1).unsqueeze(-1)

        self._discrete_points_aligned = hel_rotated_in_field
        self._normals_aligned = normal_vectors_rotated
        self.state = "Aligned"

    def align_reverse(self):
        self.state = "OnGround"

    @property
    def discrete_points(self):
        if self.state == 'OnGround':
            return self._discrete_points_orig
        elif self.state == 'Aligned':
            return self._discrete_points_aligned
        else:
            raise ValueError(f'unknown state {self.state}')

    @property
    def normals(self):
        if self.state == 'OnGround':
            return self._normals_orig
        elif self.state == 'Aligned':
            return self._normals_aligned
        else:
            raise ValueError(f'unknown state {self.state}')
