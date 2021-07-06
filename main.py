# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 09:31:21 2019

@author: ober_lu
"""


from numba import cuda, int16, float32


import math

from timeit import default_timer as timer
from scipy.spatial.transform import Rotation as R
import torch as th
from utils import draw_raytracer, Rx, Ry, Rz, heliostat_coord_system,LinePlaneCollision, calc_aimpoints, define_heliostat, rotate_heliostat
import matplotlib.pyplot as plt
import os
libdir = os.environ.get('NUMBAPRO_CUDALIB')
# os.environ['NUMBA_ENABLE_CUDASIM'] = 1
os.environ['CUDA_VISIBLE_DEVICES'] = "0"



#####Parameters#####
# grids = (int(DIM**2)//256//fac, fac//1) #cuda grid from threads , optimale anordnung
# threads = (256, 1)

##Aimpoints
aimpoint = [-50,0,0]
aimpoint_mesh_dim = 2**5 #Number of Aimpoints on Receiver

##Receiver specific parameters
planex = 10 # Receiver width
planey = 10 #Receiver height
receiver_pos = 100

####Heliostat specific Parameters
h_width = 4 # in m
h_height = 4 # in m
rows = 64 #rows of reflection points. total number is rows**2
position_on_field = th.tensor([0,0,0]).float()

#sunposition
sun = th.tensor([0,0,1]).float()
mean = [0, 0]
cov = [[0.000001, 0], [0, 0.000001]]  # diagonal covariance, used for ray scattering
num_rays = 1000

######CUDA Kernel#######

@cuda.jit((float32[:,:], float32[:,:], float32[:,:,:], float32[:,:]))
def kernel(a_int, h_int, ray_int, bitmap): #Schnittpunkt Receiver
    # a_int: Aimpoints
    # h_int: Heliostat positions
    # ray: Ray direction vector
    # dx, dy: = Distance of Ray Receiver Intersection to the lower left corner

    z, x = cuda.grid(2) # alias for threadIdx.x + ( blockIdx.x * blockDim.x ),
                        #           threadIdx.y + ( blockIdx.y * blockDim.y )

    r_fac = -h_int[x,0]/ray_int[x,z,0] # z richtung Entfernung Receiver - Ray Origin
    dx_int = h_int[x,1]+r_fac*ray_int[x,z,1]+planex/2+a_int[x,1] # x direction
    dy_int = h_int[x,2]+r_fac*ray_int[x,z,2]+planey/2+a_int[x,2]-receiver_pos # y direction. Receiver is on heigt of 100m

    if ( 0 <= dx_int < planex): # checks the point of intersection  and chooses bin in bitmap
        if (0 <= dy_int < planey):
            x_int = int(dx_int/planex*50)
            y_int = int(dy_int/planey*50)
            bitmap[x_int,y_int] += 1




###Define derived variables#####

total_bitmap = th.zeros([50, 50], dtype=th.float32) # Flux density map for heliostat field



sun = th.tensor(sun/th.linalg.norm(sun))

points_on_hel = rows**2 # reflection points on hel
hel_origin = define_heliostat(h_height, h_width, rows, points_on_hel)
hel_coordsystem = th.stack(heliostat_coord_system(position_on_field, sun, aimpoint))
hel_rotated = rotate_heliostat(hel_origin,hel_coordsystem, points_on_hel)
hel_in_field = hel_rotated+ position_on_field
aimpoints =  calc_aimpoints(hel_in_field, position_on_field, aimpoint, rows)




# draw_raytracer(hel_rotated, hel_coordsystem, position_on_field, th.tensor(aimpoint).float(),aimpoints, sun)

xi, yi = th.distributions.MultivariateNormal(th.tensor(mean).float(), th.tensor(cov).float()).sample((num_rays,)).T # scatter rays a bit
aimpoint_mesh_dim = 2**5 #Number of Aimpoints on Receiver
a= aimpoints
# print(a)
# a = th.tensor([aimpoints[:,th.randint(0,aimpoint_mesh_dim),th.randint(0,aimpoint_mesh_dim)] for i in range(fac)]).to(th.float32) # draw a random aimpoint


# rays = th.empty((fac, DIM**2//fac, 3))

ha_list = a-hel_in_field # calculate distance heliostat to aimpoint

rays = th.zeros((points_on_hel, num_rays, 3))
for i, ha in enumerate(ha_list):

    # print(ha)

    # rotate: Calculate 3D rotationmatrix in heliostat system. 1 axis is pointin towards the receiver, the other are orthogonal
    rotate = th.stack([th.tensor([ha[0],ha[1],ha[2]])/th.linalg.norm(th.tensor([ha[0],ha[1],ha[2]])),
                       th.tensor([ha[1],-ha[0],0])/th.linalg.norm(th.tensor([ha[1],-ha[0],0])),
                       th.tensor([ha[2]*ha[0],ha[2]*ha[1],-ha[0]**2-ha[1]**2])/th.linalg.norm(th.tensor([ha[2]*ha[0],ha[2]*ha[1],-ha[0]**2-ha[1]**2]))])

    inv_rot = th.linalg.inv(rotate) #inverse matrix
    # rays_tmp = th.tensor(ha)
    # print(rays_tmp.shape)

    # rays_tmp: first rotate aimpoint in right coord system, aplay xi,yi distortion, rotate back
    rays_tmp = th.stack([th.matmul(inv_rot,
                                   Rz(yi[i],
                                      Ry(xi[i],
                                         th.matmul(rotate,
                                                   ha
                                                   )
                                         )
                                      )
                                   ) for i in range(num_rays)
                         ]).to(th.float32)
    rays[i] = rays_tmp


rays = th.tensor(rays.to(th.float32))
kernel_dt = 0
planeNormal = th.tensor([1, 0, 0]).float()
planePoint = th.tensor(aimpoint).float()
for j, point in enumerate(hel_in_field):
    bitmap = th.zeros([50, 50], dtype=th.float32) #Flux density map for single heliostat
    start = timer()
    # Execute the kernel
    for k, ray in enumerate(rays[j]):
        intersection = LinePlaneCollision(planeNormal, planePoint, ray, point, epsilon=1e-6)
        # print(intersection)
        dx_int = intersection[1] +planex/2
        dy_int = intersection[2] +planey/2
        if ( 0 <= dx_int < planex): # checks the point of intersection  and chooses bin in bitmap
            if (0 <= dy_int < planey):

                x_int = int(dx_int/planex*50)
                y_int = int(dy_int/planey*50)
                bitmap[x_int,y_int] += 1


    total_bitmap += bitmap#th.sum(d_bitmap, axis = 2)
    kernel_dt += timer() - start

plt.imshow(total_bitmap, cmap='gray')
