# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:38:11 2021

@author: parg_ma
"""
import matplotlib.pyplot as plt
import numpy as np 

def draw_raytracer(h_rotated, h_matrix, position_on_field, aimpoint,aimpoints, sun):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # h_rotated = h_rotated
    # aimpoints = aimpoints-position_on_field
    # aimpoint = aimpoint-position_on_field
    
    ax.scatter(h_rotated[:,0],h_rotated[:,1],h_rotated[:,2]) #Heliostat
    # ax.scatter(aimpoint[0],aimpoint[1],aimpoint[2]) #Aimpoint
    # ax.scatter(aimpoints[:,0],aimpoints[:,1],aimpoints[:,2])
    ax.scatter(sun[0]*50,sun[1]*50,sun[2]*50) #Sun
    
    # ax.set_xlim3d(-50, 50)
    # ax.set_ylim3d(-50, 50)
    # ax.set_zlim3d(0, 50)
    
    #Heliostat Coordsystem
    ax.quiver(position_on_field[0], position_on_field[1], position_on_field[2], h_matrix[0][0], h_matrix[0][1], h_matrix[0][2], length=10, normalize=True, color="b")
    ax.quiver(position_on_field[0], position_on_field[1], position_on_field[2], h_matrix[1][0], h_matrix[1][1], h_matrix[1][2], length=10, normalize=True, color="g")
    ax.quiver(position_on_field[0], position_on_field[1], position_on_field[2], h_matrix[2][0], h_matrix[2][1], h_matrix[2][2], length=10, normalize=True, color="r")
    
def heliostat_coord_system (Position, Sun, Aimpoint):

    pSun = Sun
    print("Sun",pSun)
    pPosition = np.array(Position) 
    print("Position", pPosition)
    pAimpoint = np.array(Aimpoint)
    print("Aimpoint", pAimpoint)
    

#Berechnung Idealer Heliostat
#0. Iteration
    z = pAimpoint - pPosition
    z = z/np.linalg.norm(z)
    z = pSun + z
    z = z/np.linalg.norm(z)
    
    x = [z[1],-z[0], 0]
    x = x/np.linalg.norm(x)
    y = np.cross(z,x)
    
    
    return x,y,z


def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
 
	ndotu = planeNormal.dot(rayDirection)
	if abs(ndotu) < epsilon:
		raise RuntimeError("no intersection or line is within plane")
 
	w = rayPoint - planePoint
	si = -planeNormal.dot(w) / ndotu
	Psi = w + si * rayDirection + planePoint
	return Psi
 
 
	#Define plane
#Rotation Matricies
def Rx(alpha, vec):
    return np.dot(np.array([[1, 0, 0],[0, np.cos(alpha), -np.sin(alpha)],[0, np.sin(alpha), np.cos(alpha)]]),vec)

def Ry(alpha, vec):
    return np.dot(np.array([[np.cos(alpha), 0, np.sin(alpha)],[0, 1, 0],[-np.sin(alpha), 0, np.cos(alpha)]]),vec)

def Rz(alpha, vec):
    return np.dot(np.array([[np.cos(alpha), -np.sin(alpha), 0],[np.sin(alpha), np.cos(alpha), 0],[0, 0, 1]]),vec)