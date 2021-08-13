# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:38:11 2021

@author: parg_ma
"""

import torch as th
import struct
import numpy as np
from rotation import rot_apply, rot_as_euler, rot_from_matrix, rot_from_rotvec

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

def load_deflec(filename, take_n_vectors,device="cpu", concentratorHeader_struct_fmt ='=5f2I2f',facetHeader_struct_fmt = '=i9fI', ray_struct_fmt = '=7f'): 
    """
    binp_filename : string including file ending binp or bpro
    Concentrator Header is expected to be float[5], unsigned int[2], float[2]
    Facet Header is expected to be  int, float[9], unsigned int
    Ray is Decribed by 3 positions floats, 3 directions floats, 1 power float. Therefor expected to be float[7]
    """
    concentratorHeader_struct_len = struct.calcsize(concentratorHeader_struct_fmt)
    facetHeader_struct_len = struct.calcsize(facetHeader_struct_fmt)
    ray_struct_len = struct.calcsize(ray_struct_fmt)
    
    positions= []
    directions = []
    # powers = []
    with open(filename, "rb") as file:
        byte_data = file.read(concentratorHeader_struct_len)
        concentratorHeader_data = struct.Struct(concentratorHeader_struct_fmt).unpack_from(byte_data)
        print("READING bpro filename: " + filename)
    
        hel_pos = concentratorHeader_data[0:3]
        print("Hel Position", hel_pos)
        width_height = concentratorHeader_data[3:5]
        print("Hel Width-Height", width_height)
        #offsets = concentratorHeader_data[7:9]
        n_xy = concentratorHeader_data[5:7]
        
        
        nFacets = n_xy[0] * n_xy[1]
        for f in range(nFacets):
        # for f in range(1):
            byte_data = file.read(facetHeader_struct_len)
            facetHeader_data = struct.Struct(facetHeader_struct_fmt).unpack_from(byte_data)
            
            #facetshape = facetHeader_data[0] # 0 for square, 1 for round 2 triangle ....
            #facet_pos = facetHeader_data[1:4]
            #facet_vec_x = facetHeader_data[4:7]
            #facet_vec_y = facetHeader_data[7:10]
            n_rays = facetHeader_data[10]

            for r in range(n_rays):
                byte_data = file.read(ray_struct_len)	
                ray_data = struct.Struct(ray_struct_fmt).unpack_from(byte_data)
                
                positions.append([ray_data[0],ray_data[1],ray_data[2]])
                directions.append([ray_data[3],ray_data[4],ray_data[5]])
                # powers.append(ray_data[6])

        directions = th.tensor(directions[0::int(len(directions)/take_n_vectors)], device=device)
        positions = th.tensor(positions[0::int(len(positions)/take_n_vectors)], device = device)
        return directions, positions #,powers

def define_heliostat(h_height, h_width, rows, points_on_hel, device):
    columns = int(points_on_hel)//rows
    column = th.arange(columns, device=device)
    row = th.arange(rows, device=device)

    h_x = (row/(rows-1)*h_height)-(h_height/2)
    h_x = th.tile(h_x, (columns,))
    h_y = (column/(columns-1)*h_width)-(h_width/2) #heliostat y position
    h_y = th.tile(h_y.unsqueeze(-1), (1, columns)).ravel()
    h_z = th.zeros_like(h_x)

    h = th.hstack(list(map(lambda t: t.unsqueeze(-1), [h_x, h_y, h_z]))).reshape(len(h_x), -1)
    return h

def rotate_heliostat(h,hel_coordsystem):
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

def add_distortion(vector_field, distortion_center, points_on_hel, threshold = 1.5, positive = True):
    device
    dc_x, dc_y = distortion_center
    
    x,y = th.meshgrid(th.linspace(-5+dc_x,5+dc_x,int(th.sqrt(points_on_hel))),
                      th.linspace(-5+dc_y,5+dc_y,int(th.sqrt(points_on_hel)))
                      )
    distortion_x = (x/2).reshape(int(points_on_hel))
    distortion_y = (y/2).reshape(int(points_on_hel))
    distortion = th.zeros((int(points_on_hel),3), device=device)
    distortion[:,0] = distortion_x
    distortion[:,1] = distortion_y
    
    for i in range(len(distortion)):
        if th.norm(distortion[i]) > threshold:
          distortion[i] = th.tensor([0,0,0], device=device)

    if positive:
        new_vec_field = (vector_field - distortion)/th.norm(vector_field - distortion, dim=1)[:,None]
    else:
        new_vec_field = (vector_field + distortion)/th.norm(vector_field + distortion, dim=1)[:,None]
        
    # print(new_vec_field)
    # exit()
    return new_vec_field

def flatten_aimpoints(aimpoints):
    X = th.flatten(aimpoints[0])
    Y = th.flatten(aimpoints[1])
    Z = th.flatten(aimpoints[2])
    aimpoints = th.stack((X,Y,Z), dim=1)
    return aimpoints





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
        hel_in_field,
        xi,
        yi,
):
    intersections = LinePlaneCollision(planeNormal, planePoint, ray_directions, hel_in_field)
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
    # rays = rotated_has.transpose(0, -1).transpose(1, -1)
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
