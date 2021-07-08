# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:38:11 2021

@author: parg_ma
"""
import math
import re

import matplotlib.pyplot as plt
import torch as th


def define_heliostat(h_height, h_width, rows, points_on_hel):
    h = th.empty((points_on_hel,3)) # darray with all heliostats (#heliostats, 3 coords)
    columns = points_on_hel//rows
    i= 0
    for column in range(columns):
        for row in range(rows):
            h[i,0] = (row/(rows-1)*h_height)-(h_height/2)
            h[i,1] = (column/(columns-1)*h_width)-(h_width/2) #heliostat y position
            h[i,2] = 0 # helioistat z position

            # h[i] = h[i]+ position_on_field
            i+=1
    return h

# Rotation functions ported from here:
# https://github.com/scipy/scipy/blob/50dbaf94a570fa817d0d72e0b94d9f87c4909a4e/scipy/spatial/transform/rotation.pyx

def rot_from_matrix(mat):
    is_single = False
    if (mat.ndim not in [2, 3] or mat.shape[-2:] != (3, 3)):
        raise ValueError(
            f'expected `mat` to have shape (3, 3) or (N, 3, 3), '
            f'got {mat.shape}'
        )

    if mat.shape == (3, 3):
        cmat = mat[None, :, :]
        is_single = True
    else:
        cmat = mat

    num_rots = cmat.shape[0]
    decision = th.empty(4)

    quat = th.empty(num_rots, 4)

    for ind in range(num_rots):
        decision[0] = cmat[ind, 0, 0]
        decision[1] = cmat[ind, 1, 1]
        decision[2] = cmat[ind, 2, 2]
        decision[3] = (
            cmat[ind, 0, 0]
            + cmat[ind, 1, 1]
            + cmat[ind, 2, 2]
        )
        choice = th.argmax(decision)

        if choice != 3:
            i = choice
            j = (i + 1) % 3
            k = (j + 1) % 3

            quat[ind, i] = 1 - decision[3] + 2 * cmat[ind, i, i]
            quat[ind, j] = cmat[ind, j, i] + cmat[ind, i, j]
            quat[ind, k] = cmat[ind, k, i] + cmat[ind, i, k]
            quat[ind, 3] = cmat[ind, k, j] - cmat[ind, j, k]
        else:
            quat[ind, 0] = cmat[ind, 2, 1] - cmat[ind, 1, 2]
            quat[ind, 1] = cmat[ind, 0, 2] - cmat[ind, 2, 0]
            quat[ind, 2] = cmat[ind, 1, 0] - cmat[ind, 0, 1]
            quat[ind, 3] = 1 + decision[3]

        quat[ind] /= th.linalg.norm(quat[ind])

    if is_single:
        return quat[0]
    else:
        return quat

def rot_from_rotvec(vec, degrees=False):
    is_single = False
    if degrees:
        vec = th.deg2rad(vec)

    if vec.ndim not in [1, 2] or vec.shape[-1] != 3:
        raise ValueError(
            f'expected `vec` to have shape (3,) or (N, 3), got {vec.shape}')

    if vec.shape == (3,):
        cvec = vec[None, :]
        is_single = True
    else:
        cvec = vec

    num_rots = cvec.shape[0]
    quat = th.empty(num_rots, 4)

    for ind in range(num_rots):
        angle = th.linalg.norm(cvec[ind, :])

        if angle <= 1e-3:
            sqangle = angle * angle
            scale = 0.5 - sqangle / 48 + sqangle * sqangle / 3840
        else:
            scale = th.sin(angle / 2) / angle

        quat[ind, 0] = scale * cvec[ind, 0]
        quat[ind, 1] = scale * cvec[ind, 1]
        quat[ind, 2] = scale * cvec[ind, 2]
        quat[ind, 3] = th.cos(angle / 2)

    if is_single:
        return quat[0]
    else:
        return quat

def rot_is_single(rot):
    return rot.shape == (4,)

def rot_as_matrix(rot):
    is_single = rot_is_single(rot)
    if is_single:
        rot = rot[None, :]

    num_rots = rot.shape[0]
    mat = th.empty(num_rots, 3, 3)

    for ind in range(num_rots):
        x = rot[ind, 0]
        y = rot[ind, 1]
        z = rot[ind, 2]
        w = rot[ind, 3]

        sqx = x * x
        sqy = y * y
        sqz = z * z
        sqw = w * w

        xy = x * y
        zw = z * w
        xz = x * z
        yw = y * w
        yz = y * z
        xw = x * w

        mat[ind, 0, 0] = sqx - sqy - sqz + sqw
        mat[ind, 1, 0] = 2 * (xy + zw)
        mat[ind, 2, 0] = 2 * (xz - yw)

        mat[ind, 0, 1] = 2 * (xy - zw)
        mat[ind, 1, 1] = -sqx + sqy - sqz + sqw
        mat[ind, 2, 1] = 2 * (yz + xw)

        mat[ind, 0, 2] = 2 * (xz + yw)
        mat[ind, 1, 2] = 2 * (yz - xw)
        mat[ind, 2, 2] = -sqx - sqy + sqz + sqw

    if is_single:
        return mat[0]
    else:
        return mat

def _elem_basis_vec(axis):
    vec = th.zeros(3)
    if axis == b'x':
        vec[0] = 1
    elif axis == b'y':
        vec[1] = 1
    elif axis == b'z':
        vec[2] = 1
    else:
        raise ValueError(f'unknown axis {axis}')
    return vec

def _compute_euler_from_matrix(mat, seq, extrinsic=False):
    if extrinsic:
        seq = seq[::-1]
    num_rots = mat.shape[0]

    n1 = _elem_basis_vec(seq[0:1])
    n2 = _elem_basis_vec(seq[1:2])
    n3 = _elem_basis_vec(seq[2:3])

    sl = th.dot(th.cross(n1, n2), n3)
    cl = th.dot(n1, n3)

    offset = th.atan2(sl, cl)
    c = th.empty(3, 3)
    c[0, :] = n2
    c[1, :] = th.cross(n1, n2)
    c[2, :] = n1

    rot = th.tensor([
        [1, 0, 0],
        [0, cl, sl],
        [0, -sl, cl],
    ], dtype=th.float32)

    angles = th.empty(num_rots, 3)
    eps = 1e-7

    for ind in range(num_rots):
        _angles = angles[ind, :]

        res = th.mm(c, mat[ind, :, :])
        matrix_trans = th.mm(res, c.T.mm(rot))

        matrix_trans[2, 2] = min(matrix_trans[2, 2], 1)
        matrix_trans[2, 2] = max(matrix_trans[2, 2], -1)
        _angles[1] = th.acos(matrix_trans[2, 2])

        safe1 = th.abs(_angles[1]) >= eps
        safe2 = th.abs(_angles[1] - math.pi) >= eps
        safe = safe1 and safe2

        _angles[1] += offset

        if safe:
            _angles[0] = th.atan2(matrix_trans[0, 2], -matrix_trans[1, 2])
            _angles[2] = th.atan2(matrix_trans[2, 0], matrix_trans[2, 1])

        if extrinsic:
            if not safe:
                _angles[0] = 0

            if not safe1:
                _angles[2] = th.atan2(matrix_trans[1, 0] - matrix_trans[0, 1],
                                      matrix_trans[0, 0] + matrix_trans[1, 1])

            if not safe2:
                _angles[2] = -th.atan2(matrix_trans[1, 0] + matrix_trans[0, 1],
                                       matrix_trans[0, 0] - matrix_trans[1, 1])

        else:
            if not safe:
                _angles[2] = 0

            if not safe1:
                _angles[0] = th.atan2(matrix_trans[1, 0] - matrix_trans[0, 1],
                                      matrix_trans[0, 0] + matrix_trans[1, 1])

            if not safe2:
                _angles[0] = th.atan2(matrix_trans[1, 0] + matrix_trans[0, 1],
                                      matrix_trans[0, 0] - matrix_trans[1, 1])

        if seq[0] == seq[2]:
            adjust = _angles[1] < 0 or _angles[1] > math.pi
        else:
            adjust = _angles[1] < -math.pi / 2 or _angles[1] > math.pi / 2

        if adjust and safe:
            _angles[0] += math.pi
            _angles[1] = 2 * offset - _angles[1]
            _angles[2] -= math.pi

        for i in range(3):
            if _angles[i] < -math.pi:
                _angles[i] += 2 * math.pi
            elif _angles[i] > math.pi:
                _angles[i] -= 2 * math.pi

        if extrinsic:
            tmp = _angles[0].clone()
            _angles[0] = _angles[2]
            _angles[2] = tmp

        if not safe:
            print('warning: gimbal lock detected; setting third angle to zero')

    return angles

def rot_as_euler(rot, seq, degrees=False):
    if len(seq) != 3:
        raise ValueError(f'expected 3 axes, got {seq}')

    intrinsic = re.match(r'^[XYZ]{1,3}$', seq) is not None
    extrinsic = re.match(r'^[xyz]{1,3}$', seq) is not None

    if not (intrinsic or extrinsic):
        raise ValueError(
            f'expected axes from `seq` to be from ["x", "y", "z"] or '
            f'["X", "Y", "Z"], got {seq}'
        )
    if any(seq[i] == seq[i + 1] for i in range(2)):
        raise ValueError(
            f'expected consecutive axes to be different, got {seq}')

    seq = seq.lower()
    mat = rot_as_matrix(rot)
    if mat.ndim == 2:
        mat = mat[None, :, :]
    angles = _compute_euler_from_matrix(mat, seq.encode(), extrinsic)
    if degrees:
        angles = th.rad2deg(angles)

    return angles[0] if rot_is_single(rot) else angles

def rot_apply(rot, vecs):
    mat = rot_as_matrix(rot)
    # mat: (3, 3)
    # vec: (1, 3)
    return th.squeeze(th.matmul(mat, vecs), 0)

def rotate_heliostat(h,hel_coordsystem, points_on_hel):
    h_rotated = th.empty((points_on_hel,3)) # darray with all heliostats (#heliostats, 3 coords)
    r = rot_from_matrix(hel_coordsystem)
    euler = rot_as_euler(r, 'xyx', degrees = True)
    for i in range(len(h[:])):
        ele_degrees = 90-euler[2]

        ele_radians = th.deg2rad(ele_degrees)
        ele_axis = th.tensor([0, 1, 0], dtype=th.float32)
        ele_vector = ele_radians * ele_axis
        ele = rot_from_rotvec(ele_vector)

        azi_degrees = euler[1]-90
        azi_radians = th.deg2rad(azi_degrees)
        azi_axis = th.tensor([0, 0, 1], dtype=th.float32)
        azi_vector = azi_radians * azi_axis
        azi = rot_from_rotvec(azi_vector)

        h_rotated[i] = rot_apply(azi, rot_apply(ele, h[i]))
    return h_rotated


def calc_aimpoints(h_rotated, position_on_field, aimpoint, rows):


    aimpoints = []
    # row = 0
    # column = 0
    for i in range(len(h_rotated[:])):
        # print("Aim",aimpoint)
        planeNormal = th.tensor([1, 0, 0], dtype=th.float32) # Muss noch dynamisch gestaltet werden
        planePoint = aimpoint #Any point on the plane

    	#Define ray

        rayDirection = aimpoint - position_on_field
        # print("Ray directioN", rayDirection)
        rayPoint = h_rotated[i] #Any point along the ray
        # print("ray_point", rayPoint)

        intersection = LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint)
        # print("intersection",intersection)
        # exit()
        # print("cr",column,row)
        # if i % (rows) == 0 and not i == 0:
        #     print("Hello")
        #     row +=1
        #     column=0

        # aimpoints[0,column,row] = intersection[0]
        # aimpoints[1,column,row] = intersection[1]
        # aimpoints[2,column,row] = intersection[2]
        aimpoints.append(intersection)
        # exit()

        # column +=1
    aimpoints = th.stack(aimpoints)

    return aimpoints


def flatten_aimpoints(aimpoints):
    X = th.flatten(aimpoints[0])
    Y = th.flatten(aimpoints[1])
    Z = th.flatten(aimpoints[2])
    aimpoints = th.stack((X,Y,Z), dim=1)
    return aimpoints


def draw_raytracer(h_rotated, h_matrix, position_on_field, aimpoint,aimpoints, sun):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    h_rotated = h_rotated
    aimpoints = aimpoints - position_on_field
    aimpoint = aimpoint - position_on_field

    # aimpoints = aimpoints-position_on_field
    # aimpoint = aimpoint-position_on_field

    ax.scatter(h_rotated[:,0],h_rotated[:,1],h_rotated[:,2]) #Heliostat
    ax.scatter(aimpoint[0],aimpoint[1],aimpoint[2]) #Aimpoint
    ax.scatter(aimpoints[:,0],aimpoints[:,1],aimpoints[:,2])
    ax.scatter(sun[0]*50,sun[1]*50,sun[2]*50) #Sun

    ax.set_xlim3d(-100, 100)
    ax.set_ylim3d(-100, 100)
    ax.set_zlim3d(0, 100)

    #Heliostat Coordsystem
    # ax.quiver(position_on_field[0], position_on_field[1], position_on_field[2], h_matrix[0][0], h_matrix[0][1], h_matrix[0][2], length=10, normalize=True, color="b")
    # ax.quiver(position_on_field[0], position_on_field[1], position_on_field[2], h_matrix[1][0], h_matrix[1][1], h_matrix[1][2], length=10, normalize=True, color="g")
    # ax.quiver(position_on_field[0], position_on_field[1], position_on_field[2], h_matrix[2][0], h_matrix[2][1], h_matrix[2][2], length=10, normalize=True, color="r")
    ax.quiver(0, 0, 0, h_matrix[0][0], h_matrix[0][1], h_matrix[0][2], length=10, normalize=True, color="b")
    ax.quiver(0, 0, 0, h_matrix[1][0], h_matrix[1][1], h_matrix[1][2], length=10, normalize=True, color="g")
    ax.quiver(0, 0, 0, h_matrix[2][0], h_matrix[2][1], h_matrix[2][2], length=10, normalize=True, color="r")
    plt.show()

def heliostat_coord_system (Position, Sun, Aimpoint):

    pSun = Sun
    print("Sun",pSun)
    pPosition = Position
    print("Position", pPosition)
    pAimpoint = Aimpoint
    print("Aimpoint", pAimpoint)


#Berechnung Idealer Heliostat
#0. Iteration
    z = pAimpoint - pPosition
    z = z/th.linalg.norm(z)
    z = pSun + z
    z = z/th.linalg.norm(z)

    x = th.tensor([z[1],-z[0], 0], dtype=th.float32)
    x = x/th.linalg.norm(x)
    y = th.cross(z,x)


    return x,y,z


def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):

	ndotu = planeNormal.dot(rayDirection)
	if th.abs(ndotu) < epsilon:
		raise RuntimeError("no intersection or line is within plane")

	w = rayPoint - planePoint
	si = -planeNormal.dot(w) / ndotu
	Psi = w + si * rayDirection + planePoint
	return Psi


	#Define plane
#Rotation Matricies
def Rx(alpha, vec):
    if not isinstance(alpha, th.Tensor):
        alpha = th.tensor(alpha)
    return th.matmul(th.tensor([[1, 0, 0],[0, th.cos(alpha), -th.sin(alpha)],[0, th.sin(alpha), th.cos(alpha)]], dtype=th.float32),vec)

def Ry(alpha, vec):
    if not isinstance(alpha, th.Tensor):
        alpha = th.tensor(alpha)
    return th.matmul(th.tensor([[th.cos(alpha), 0, th.sin(alpha)],[0, 1, 0],[-th.sin(alpha), 0, th.cos(alpha)]], dtype=th.float32),vec)

def Rz(alpha, vec):
    if not isinstance(alpha, th.Tensor):
        alpha = th.tensor(alpha)
    return th.matmul(th.tensor([[th.cos(alpha), -th.sin(alpha), 0],[th.sin(alpha), th.cos(alpha), 0],[0, 0, 1]], dtype=th.float32),vec)
