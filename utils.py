# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:38:11 2021

@author: parg_ma
"""
import matplotlib.pyplot as plt
import torch as th

from rotation import rot_apply, rot_as_euler, rot_from_matrix, rot_from_rotvec


def define_heliostat(h_height, h_width, rows, points_on_hel, device):
    h = th.empty((points_on_hel,3), device=device) # darray with all heliostats (#heliostats, 3 coords)
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
