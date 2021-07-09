# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 09:31:21 2019

@author: ober_lu
"""

import os
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import torch as th

from utils import draw_raytracer, Rx, Ry, Rz, heliostat_coord_system,LinePlaneCollision, define_heliostat, rotate_heliostat

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


#####Parameters#####
# grids = (int(DIM**2)//256//fac, fac//1) #cuda grid from threads , optimale anordnung
# threads = (256, 1)
seed = 0
use_gpu = True
bitmap_width = 50
bitmap_height = 50

th.manual_seed(0)
device = th.device('cuda' if use_gpu and th.cuda.is_available() else 'cpu')

##Aimpoints
aimpoint = th.tensor([-50,0,0], dtype=th.float32, device=device)
aimpoint_mesh_dim = 2**5 #Number of Aimpoints on Receiver

##Receiver specific parameters
planex = 10 # Receiver width
planey = 10 #Receiver height
receiver_pos = 100

####Heliostat specific Parameters
h_width = 4 # in m
h_height = 4 # in m
rows = 64 #rows of reflection points. total number is rows**2
position_on_field = th.tensor([0,0,0], dtype=th.float32, device=device)

#sunposition
sun = th.tensor([0,0,1], dtype=th.float32, device=device)
mean = th.tensor([0, 0], dtype=th.float32, device=device)
cov = th.tensor([[0.000001, 0], [0, 0.000001]], dtype=th.float32, device=device)  # diagonal covariance, used for ray scattering
num_rays = 100

######CUDA Kernel#######

# @cuda.jit((float32[:,:], float32[:,:], float32[:,:,:], float32[:,:]))
# def kernel(a_int, h_int, ray_int, bitmap): #Schnittpunkt Receiver
#     # a_int: Aimpoints
#     # h_int: Heliostat positions
#     # ray: Ray direction vector
#     # dx, dy: = Distance of Ray Receiver Intersection to the lower left corner

#     z, x = cuda.grid(2) # alias for threadIdx.x + ( blockIdx.x * blockDim.x ),
#                         #           threadIdx.y + ( blockIdx.y * blockDim.y )

#     r_fac = -h_int[x,0]/ray_int[x,z,0] # z richtung Entfernung Receiver - Ray Origin
#     dx_int = h_int[x,1]+r_fac*ray_int[x,z,1]+planex/2+a_int[x,1] # x direction
#     dy_int = h_int[x,2]+r_fac*ray_int[x,z,2]+planey/2+a_int[x,2]-receiver_pos # y direction. Receiver is on heigt of 100m

#     if ( 0 <= dx_int < planex): # checks the point of intersection  and chooses bin in bitmap
#         if (0 <= dy_int < planey):
#             x_int = int(dx_int/planex*bitmap_height)
#             y_int = int(dy_int/planey*bitmap_width)
#             bitmap[x_int,y_int] += 1




###Define derived variables#####

total_bitmap = th.zeros([bitmap_height, bitmap_width], dtype=th.float32, device=device) # Flux density map for heliostat field



sun = sun/th.linalg.norm(sun)

points_on_hel = rows**2 # reflection points on hel
hel_origin = define_heliostat(h_height, h_width, rows, points_on_hel, device)
hel_coordsystem = th.stack(heliostat_coord_system(position_on_field, sun, aimpoint))
hel_rotated = rotate_heliostat(hel_origin,hel_coordsystem, points_on_hel)
hel_in_field = hel_rotated+ position_on_field

ray_directions =  th.tile(aimpoint- position_on_field, (len(hel_in_field), 1))




xi, yi = th.distributions.MultivariateNormal(mean, cov).sample((num_rays,)).T.to(device) # scatter rays a bit
aimpoint_mesh_dim = 2**5 #Number of Aimpoints on Receiver

# print(a)
# a = th.tensor([aimpoints[:,th.randint(0,aimpoint_mesh_dim),th.randint(0,aimpoint_mesh_dim)] for i in range(fac)], device=device).to(th.float32) # draw a random aimpoint


# rays = th.empty((fac, DIM**2//fac, 3), device=device)
# draw_raytracer(hel_rotated, hel_coordsystem, position_on_field, aimpoint,aimpoints, sun)
        # print("Ray directioN", rayDirection)


rays = th.zeros((points_on_hel, num_rays, 3), device=device)
planeNormal = th.tensor([1, 0, 0], dtype=th.float32, device=device) # Muss noch dynamisch gestaltet werden
planePoint = aimpoint #Any point on the plane

for i, heliostat_point in enumerate(hel_in_field):
    print(i/len(hel_in_field))
    rayPoint = heliostat_point #Any point along the ray
    ray_direction = ray_directions[i]

    intersection = LinePlaneCollision(planeNormal, planePoint, ray_direction, rayPoint)
    a = intersection
    ha = a-heliostat_point
    # rotate: Calculate 3D rotationmatrix in heliostat system. 1 axis is pointin towards the receiver, the other are orthogonal
    rotate = th.stack([th.tensor([ha[0],ha[1],ha[2]], device=device)/th.linalg.norm(th.tensor([ha[0],ha[1],ha[2]], device=device)),
               th.tensor([ha[1],-ha[0],0], device=device)/th.linalg.norm(th.tensor([ha[1],-ha[0],0], device=device)),
               th.tensor([ha[2]*ha[0],ha[2]*ha[1],-ha[0]**2-ha[1]**2], device=device)/th.linalg.norm(th.tensor([ha[2]*ha[0],ha[2]*ha[1],-ha[0]**2-ha[1]**2], device=device))])
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


rays = rays.to(th.float32)
kernel_dt = 0
bitmap = th.empty([bitmap_height, bitmap_width], dtype=th.float32, device=device) #Flux density map for single heliostat
for j, point in enumerate(hel_in_field):
    print(j/hel_in_field)
    bitmap[:] = 0
    start = timer()
    # Execute the kernel
    for k, ray in enumerate(rays[j]):
        intersection = LinePlaneCollision(planeNormal, planePoint, ray, point, epsilon=1e-6)
        # print(intersection)
        dx_int = intersection[1] +planex/2
        dy_int = intersection[2] +planey/2
        if ( 0 <= dx_int < planex): # checks the point of intersection  and chooses bin in bitmap
            if (0 <= dy_int < planey):

                x_int = int(dx_int/planex*bitmap_height)
                y_int = int(dy_int/planey*bitmap_width)
                bitmap[x_int,y_int] += 1


    total_bitmap += bitmap#th.sum(d_bitmap, axis = 2)
    kernel_dt += timer() - start

plt.imshow(total_bitmap.detach().cpu().numpy(), cmap='jet')
plt.show()
