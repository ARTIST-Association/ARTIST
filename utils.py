# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:38:11 2021

@author: parg_ma
"""
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
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

def rotate_heliostat(h,hel_coordsystem, points_on_hel):
    h_rotated = th.empty((points_on_hel,3)) # darray with all heliostats (#heliostats, 3 coords)
    r = R.from_matrix(hel_coordsystem)
    euler = th.tensor(r.as_euler('xyx', degrees = True))
    for i in range(len(h[:])):
        ele_degrees = 90-euler[2]

        ele_radians = th.deg2rad(ele_degrees)
        ele_axis = th.tensor([0, 1, 0]).float()
        ele_vector = ele_radians * ele_axis
        ele = R.from_rotvec(ele_vector.numpy())

        azi_degrees = euler[1]-90
        azi_radians = th.deg2rad(azi_degrees)
        azi_axis = th.tensor([0, 0, 1]).float()
        azi_vector = azi_radians * azi_axis
        azi = R.from_rotvec(azi_vector.numpy())

        h_rotated[i] = th.tensor(azi.apply(ele.apply(h[i])))
    return h_rotated


def calc_aimpoints(h_rotated, position_on_field, aimpoint, rows):


    aimpoints = []
    # row = 0
    # column = 0
    for i in range(len(h_rotated[:])):
        # print("Aim",aimpoint)
        planeNormal = th.tensor([1, 0, 0]).float() # Muss noch dynamisch gestaltet werden
        planePoint = th.tensor(aimpoint).float() #Any point on the plane

    	#Define ray

        rayDirection = th.tensor(aimpoint).float() - th.tensor(position_on_field).float()
        # print("Ray directioN", rayDirection)
        rayPoint = th.tensor(h_rotated[i]) #Any point along the ray
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
    pPosition = th.tensor(Position)
    print("Position", pPosition)
    pAimpoint = th.tensor(Aimpoint)
    print("Aimpoint", pAimpoint)


#Berechnung Idealer Heliostat
#0. Iteration
    z = pAimpoint - pPosition
    z = z/th.linalg.norm(z)
    z = pSun + z
    z = z/th.linalg.norm(z)

    x = th.tensor([z[1],-z[0], 0]).float()
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
    alpha = th.tensor(alpha)
    return th.matmul(th.tensor([[1, 0, 0],[0, th.cos(alpha), -th.sin(alpha)],[0, th.sin(alpha), th.cos(alpha)]]).float(),vec)

def Ry(alpha, vec):
    alpha = th.tensor(alpha)
    return th.matmul(th.tensor([[th.cos(alpha), 0, th.sin(alpha)],[0, 1, 0],[-th.sin(alpha), 0, th.cos(alpha)]]).float(),vec)

def Rz(alpha, vec):
    alpha = th.tensor(alpha)
    return th.matmul(th.tensor([[th.cos(alpha), -th.sin(alpha), 0],[th.sin(alpha), th.cos(alpha), 0],[0, 0, 1]]).float(),vec)
