# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:38:11 2021

@author: parg_ma
"""
import matplotlib.pyplot as plt
import torch as th

from rotation import rot_apply, rot_as_euler, rot_from_matrix, rot_from_rotvec


def define_heliostat(h_height, h_width, rows, points_on_hel, device):
    columns = points_on_hel//rows
    column = th.arange(columns, device=device)
    row = th.arange(rows, device=device)

    h_x = (row/(rows-1)*h_height)-(h_height/2)
    h_x = th.tile(h_x, (columns,))
    h_y = (column/(columns-1)*h_width)-(h_width/2) #heliostat y position
    h_y = th.tile(h_y.unsqueeze(-1), (1, columns)).ravel()
    h_z = th.zeros_like(h_x)

    h = th.hstack(list(map(lambda t: t.unsqueeze(-1), [h_x, h_y, h_z]))).reshape(len(h_x), -1)
    return h

def rotate_heliostat(h,hel_coordsystem, points_on_hel):
    r = rot_from_matrix(hel_coordsystem)
    euler = rot_as_euler(r, 'xyx', degrees = True)
    ele_degrees = 90-euler[2]

    ele_radians = th.deg2rad(ele_degrees)
    ele_axis = th.tensor([0, 1, 0], dtype=th.float32, device=h.device)
    ele_vector = ele_radians * ele_axis
    ele = rot_from_rotvec(ele_vector)

    azi_degrees = euler[1]-90
    azi_radians = th.deg2rad(azi_degrees)
    azi_axis = th.tensor([0, 0, 1], dtype=th.float32, device=h.device)
    azi_vector = azi_radians * azi_axis
    azi = rot_from_rotvec(azi_vector)

    h_rotated = rot_apply(azi, rot_apply(ele, h.unsqueeze(-1))) # darray with all heliostats (#heliostats, 3 coords)
    return h_rotated.squeeze(-1)



def flatten_aimpoints(aimpoints):
    X = th.flatten(aimpoints[0])
    Y = th.flatten(aimpoints[1])
    Z = th.flatten(aimpoints[2])
    aimpoints = th.stack((X,Y,Z), dim=1)
    return aimpoints


def draw_raytracer(h_rotated, h_matrix, position_on_field, aimpoint,aimpoints, sun):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # aimpoints = aimpoints-position_on_field
    # aimpoint = aimpoint-position_on_field
    print(aimpoints.shape)
    ax.scatter(h_rotated[:,0],h_rotated[:,1],h_rotated[:,2]) #Heliostat
    ax.scatter(aimpoint[0],aimpoint[1],aimpoint[2]) #Aimpoint
    ax.scatter(aimpoints[0,:,0],aimpoints[0,:,1],aimpoints[0,:,2])
    ax.scatter(sun[0]*50,sun[1]*50,sun[2]*50) #Sun

    ax.set_xlim3d(-50, 50)
    ax.set_ylim3d(-50, 50)
    ax.set_zlim3d(0, 50)

    #Heliostat Coordsystem
    # ax.quiver(position_on_field[0], position_on_field[1], position_on_field[2], h_matrix[0][0], h_matrix[0][1], h_matrix[0][2], length=10, normalize=True, color="b")
    # ax.quiver(position_on_field[0], position_on_field[1], position_on_field[2], h_matrix[1][0], h_matrix[1][1], h_matrix[1][2], length=10, normalize=True, color="g")
    # ax.quiver(position_on_field[0], position_on_field[1], position_on_field[2], h_matrix[2][0], h_matrix[2][1], h_matrix[2][2], length=10, normalize=True, color="r")
    ax.quiver(0, 0, 0, h_matrix[0][0], h_matrix[0][1], h_matrix[0][2], length=10, normalize=True, color="b")
    ax.quiver(0, 0, 0, h_matrix[1][0], h_matrix[1][1], h_matrix[1][2], length=10, normalize=True, color="g")
    ax.quiver(0, 0, 0, h_matrix[2][0], h_matrix[2][1], h_matrix[2][2], length=10, normalize=True, color="r")
    plt.show()

def draw_heliostat(h_rotated, ray_directions):
    fig = plt.figure()
    ax = plt.axes(projection='3d')


    # aimpoints = aimpoints-position_on_field
    # aimpoint = aimpoint-position_on_field

    ax.scatter(h_rotated[:,0],h_rotated[:,1],h_rotated[:,2]) #Heliostat

    ax.set_xlim3d(-50, 0)
    ax.set_ylim3d(-10, 10)
    ax.set_zlim3d(0, 5) 
    ax.quiver(h_rotated[:,0],h_rotated[:,1],h_rotated[:,2], ray_directions[:,0], ray_directions[:,1], ray_directions[:,2], length=50, normalize=True, color="b")
    # ax.quiver(h_rotated[:,0],h_rotated[:,1],h_rotated[:,2], ray_directions[1][0], ray_directions[1][1], ray_directions[1][2], length=1, normalize=True, color="g")
    # ax.quiver(h_rotated[:,0],h_rotated[:,1],h_rotated[:,2], ray_directions[2][0], ray_directions[2][1], ray_directions[2][2], length=1, normalize=True, color="r")
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

    x = th.tensor([z[1],-z[0], 0], dtype=th.float32, device=Position.device)
    x = x/th.linalg.norm(x)
    y = th.cross(z,x)


    return x,y,z


def LinePlaneCollision(planeNormal, planePoint, rayDirections, rayPoints, epsilon=1e-6):

    ndotu = rayDirections.matmul(planeNormal)
    if (th.abs(ndotu) < epsilon).any():
        raise RuntimeError("no intersection or line is within plane")

    ws = rayPoints - planePoint
    sis = -ws.matmul(planeNormal) / ndotu
    Psis = ws + sis.unsqueeze(-1) * rayDirections + planePoint
    return Psis

	#Define plane
#Rotation Matricies
def Rx(alpha, mat):
    zeros = th.zeros_like(alpha)
    coss = th.cos(alpha)
    sins = th.sin(alpha)
    rots_x = th.hstack(list(map(lambda t: t.unsqueeze(-1), [th.ones_like(alpha), zeros, zeros])))
    rots_y = th.hstack(list(map(lambda t: t.unsqueeze(-1), [zeros, coss, -sins])))
    rots_z = th.hstack(list(map(lambda t: t.unsqueeze(-1), [zeros, sins, coss])))
    rots = th.hstack([rots_x, rots_y, rots_z]).reshape(rots_x.shape[0], rots_x.shape[1], -1)
    return th.matmul(rots,mat)

def Ry(alpha, mat):
    zeros = th.zeros_like(alpha)
    coss = th.cos(alpha)
    sins = th.sin(alpha)
    rots_x = th.hstack(list(map(lambda t: t.unsqueeze(-1), [coss, zeros, sins])))
    rots_y = th.hstack(list(map(lambda t: t.unsqueeze(-1), [zeros, th.ones_like(alpha), zeros])))
    rots_z = th.hstack(list(map(lambda t: t.unsqueeze(-1), [-sins, zeros, coss])))
    rots = th.hstack([rots_x, rots_y, rots_z]).reshape(rots_x.shape[0], rots_x.shape[1], -1)
    return th.matmul(rots,mat.transpose(0, -1))

def Rz(alpha, mat):
    zeros = th.zeros_like(alpha)
    coss = th.cos(alpha)
    sins = th.sin(alpha)
    rots_x = th.hstack(list(map(lambda t: t.unsqueeze(-1), [coss, -sins, zeros])))
    rots_y = th.hstack(list(map(lambda t: t.unsqueeze(-1), [sins, coss, zeros])))
    rots_z = th.hstack(list(map(lambda t: t.unsqueeze(-1), [zeros, zeros, th.ones_like(alpha)])))
    rots = th.hstack([rots_x, rots_y, rots_z]).reshape(rots_x.shape[0], rots_x.shape[1], -1)
    return th.matmul(rots,mat)

def compute_receiver_intersections(
        planeNormal,
        planePoint,
        ray_directions,
        rayPoints,
        hel_in_field,
        xi,
        yi,
):
    intersections = LinePlaneCollision(planeNormal, planePoint, ray_directions, rayPoints)
    as_ = intersections
    has = as_-hel_in_field

    # rotate: Calculate 3D rotationmatrix in heliostat system. 1 axis is pointin towards the receiver, the other are orthogonal
    rotates_x = th.hstack(list(map(lambda t: t.unsqueeze(-1), [has[:, 0],has[:, 1],has[:, 2]])))
    rotates_x = rotates_x / th.linalg.norm(rotates_x, dim=-1).unsqueeze(-1)
    rotates_y = th.hstack(list(map(lambda t: t.unsqueeze(-1), [has[:, 1],-has[:, 0],th.zeros(has.shape[:1], device=as_.device)])))
    rotates_y = rotates_y / th.linalg.norm(rotates_y, dim=-1).unsqueeze(-1)
    rotates_z = th.hstack(list(map(lambda t: t.unsqueeze(-1), [has[:, 2]*has[:, 0],has[:, 2]*has[:, 1],-has[:, 0]**2-has[:, 1]**2])))
    rotates_z = rotates_z / th.linalg.norm(rotates_z, dim=-1).unsqueeze(-1)
    rotates = th.hstack([rotates_x, rotates_y, rotates_z]).reshape(rotates_x.shape[0], rotates_x.shape[1], -1)
    inv_rot = th.linalg.inv(rotates) #inverse matrix
    # rays_tmp = th.tensor(ha, device=device)
    # print(rays_tmp.shape)

    # rays_tmp: first rotate aimpoint in right coord system, aplay xi,yi distortion, rotate back
    rotated_has = th.matmul(rotates, has.unsqueeze(-1))

    rays = th.matmul(inv_rot,
                               Rz(yi,
                                  Ry(xi,
                                     rotated_has
                                     )
                                  ).transpose(0, -1)
                               ).transpose(0, -1).transpose(1, -1)


    # rays = rays.to(th.float32)

    # Execute the kernel
    intersections = LinePlaneCollision(planeNormal, planePoint, rays, hel_in_field, epsilon=1e-6)
    # print(intersections)
    return intersections

def sample_bitmap(intersections, planex, planey, bitmap_height, bitmap_width):
    dx_ints = intersections[:, :, 1] +planex/2
    dy_ints = intersections[:, :, 2] +planey/2
    # checks the points of intersection  and chooses bins in bitmap
    indices = ( (-1 <= dx_ints) & (dx_ints < planex + 1) & (-1 <= dy_ints) & (dy_ints < planey + 1))
    return sample_bitmap_(dx_ints, dy_ints, indices, planex, planey, bitmap_height, bitmap_width)

def sample_bitmap_(dx_ints, dy_ints, indices, planex, planey, bitmap_height, bitmap_width):

    x_ints = dx_ints[indices]/planex*bitmap_height
    y_ints = dy_ints[indices]/planey*bitmap_width

    x_inds_low = x_ints.long()
    y_inds_low = y_ints.long()
    x_inds_high = x_inds_low + 1
    y_inds_high = y_inds_low + 1

    x_ints_low = x_inds_high - x_ints
    y_ints_low = y_inds_high - y_ints
    x_ints_high = x_ints - x_inds_low
    y_ints_high = y_ints - y_inds_low
    del x_ints
    del y_ints

    x_inds_1 = x_inds_low
    y_inds_1 = y_inds_low
    ints_1 = x_ints_low * y_ints_low

    x_inds_2 = x_inds_high
    y_inds_2 = y_inds_low
    ints_2 = x_ints_high * y_ints_low

    x_inds_3 = x_inds_low
    y_inds_3 = y_inds_high
    ints_3 = x_ints_low * y_ints_high
    del x_inds_low
    del y_inds_low
    del x_ints_low
    del y_ints_low

    x_inds_4 = x_inds_high
    y_inds_4 = y_inds_high
    ints_4 = x_ints_high * y_ints_high
    del x_inds_high
    del y_inds_high
    del x_ints_high
    del y_ints_high

    x_inds = th.hstack([x_inds_1, x_inds_2, x_inds_3, x_inds_4]).long().ravel()
    del x_inds_1
    del x_inds_2
    del x_inds_3
    del x_inds_4

    y_inds = th.hstack([y_inds_1, y_inds_2, y_inds_3, y_inds_4]).long().ravel()
    del y_inds_1
    del y_inds_2
    del y_inds_3
    del y_inds_4

    ints = th.hstack([ints_1, ints_2, ints_3, ints_4]).ravel()
    del ints_1
    del ints_2
    del ints_3
    del ints_4
    # Normalize
    ints = ints / th.max(ints)

    indices = (0 <= x_inds) & (x_inds < bitmap_width) & (0 <= y_inds) & (y_inds < bitmap_height)

    total_bitmap = th.zeros([bitmap_height, bitmap_width], dtype=th.float32, device=dx_ints.device) # Flux density map for heliostat field
    total_bitmap.index_put_(
        (x_inds[indices], y_inds[indices]),
        ints[indices],
        accumulate=True,
    )
    return total_bitmap

def curl(f, arg):
    jac = th.autograd.functional.jacobian(f, arg, create_graph=True)

    rot_x = jac[2][1] - jac[1][2]
    rot_y = jac[0][2] - jac[2][0]
    rot_z = jac[1][0] - jac[0][1]

    return th.tensor([rot_x, rot_y, rot_z])
