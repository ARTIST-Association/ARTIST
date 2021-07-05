import numpy as np

from numba import cuda, int16, float32
from scipy.spatial.transform import Rotation as R

import math

from timeit import default_timer as timer

from utils import draw_raytracer, Rx, Ry, Rz, heliostat_coord_system,LinePlaneCollision
import os
libdir = os.environ.get('NUMBAPRO_CUDALIB')
# os.environ['NUMBA_ENABLE_CUDASIM'] = 1
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import sys
print(sys.prefix)

DIM = 2**8 #Threads auf Graka
fac = 2**8 
grids = (int(DIM**2)//256//fac, fac//1) #cuda grid from threads , optimale anordnung
threads = (256, 1)


##Aimpoints
aimpoint = [-50,0,100]
aimpoint_mesh_dim = 2**5 #Number of Aimpoints on Receiver

##Receiver specific parameters
planex = 10 # Receiver width
planey = 10 #Receiver height
receiver_pos = 100

####Heliostat specific Parameters
h_width = 4 # in m
h_height = 4 # in m
rows = 4 #rows of reflection points. total number is rows**2
position_on_field = np.array([0,0,0])

#sunposition
sun = np.array([0,0,1])
mean = [0, 0]
cov = [[0.000001, 0], [0, 0.000001]]  # diagonal covariance, used for ray scattering


# plt.close("all")




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

            
def aimpoints_new(euler, aimpoint, rows):
    print(rows)
    aimpoints = np.ones((3,int(rows),int(rows)))
    print(aimpoints)
    row = 0
    column = 0
    for i in range(len(h[:])):
        # print(len(h))
        # print(i)
        
        ele_degrees = 90-euler[2]
    
        ele_radians = np.radians(ele_degrees)
        ele_axis = np.array([0, 1, 0])
        ele_vector = ele_radians * ele_axis
        ele = R.from_rotvec(ele_vector)
    
        azi_degrees = euler[1]-90
        azi_radians = np.radians(azi_degrees)
        azi_axis = np.array([0, 0, 1])
        azi_vector = azi_radians * azi_axis
        azi = R.from_rotvec(azi_vector)
    
        h_rotated[i] = azi.apply(ele.apply(h[i]))
        
        
        planeNormal = np.array([1, 0, 0]) # Muss noch dynamisch gestaltet werden
        planePoint = np.array(aimpoint) #Any point on the plane
     
    	#Define ray
        
        rayDirection = np.array(aimpoint) - np.array(position_on_field)
        rayPoint = np.array(h_rotated[i]) #Any point along the ray
        
        intersection = LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint)
        print("cr",column,row)
        if i % (rows) == 0 and not i == 0:
            print("Hello")
            row +=1
            column=0
        
        aimpoints[0,column,row] = intersection[0]
        aimpoints[1,column,row] = intersection[1]
        aimpoints[2,column,row] = intersection[2]
        
        column +=1

        
        
            
    
    
    aimpoints = np.array(aimpoints)
    
    return aimpoints

def aimpoints_old(aimpoint_mesh_dim):
    grid = np.meshgrid(np.linspace(-4, 4, aimpoint_mesh_dim),np.linspace(-4, 4, aimpoint_mesh_dim))
    aimpoints = np.ones((3,int(aimpoint_mesh_dim),int(aimpoint_mesh_dim)))
    aimpoints[0,:,:] = 0
    aimpoints[1,:,:] = grid[0][:,:]
    aimpoints[2,:,:] = 100+grid[1][:,:]
    aimpoints = aimpoints.astype(np.float32)
    return aimpoints
    

bitmap = np.zeros([50, 50], dtype=np.float32) #Flux density map for single heliostat
total_bitmap = np.zeros([50, 50], dtype=np.float32) # Flux density map for heliostat field
d_bitmap = cuda.to_device(bitmap)







points_on_hel = rows**2 # reflection points on hel 
h = np.empty((points_on_hel,3)) # darray with all heliostats (#heliostats, 3 coords)
columns = points_on_hel//rows
i= 0
for column in range(columns):
    for row in range(rows):
        h[i,0] = (row/(rows-1)*h_height)-(h_height/2)
        h[i,1] = (column/(columns-1)*h_width)-(h_width/2) #heliostat y position
        h[i,2] = 0 # helioistat z position
        
        # h[i] = h[i]+ position_on_field
        i+=1

h_rotated = np.empty((points_on_hel,3)) # darray with all heliostats (#heliostats, 3 coords)
sun = np.array(sun/np.linalg.norm(sun))
h_matrix = np.array(heliostat_coord_system(position_on_field, sun, aimpoint))
# product = np.dot(np.array(h_matrix),np.array(h_matrix).T) # Check if Orthogonal
# np.fill_diagonal(product,0)
# if (product.any() == 0):
#     print("XXX")
r = R.from_matrix(h_matrix)
euler = r.as_euler('xyx', degrees = True)
aimpoint_mesh_dim = 4
aimpoints = aimpoints_old(aimpoint_mesh_dim)
# aimpoints = aimpoints_new(euler, aimpoint, rows)
print(aimpoints)

h = np.array(h)
h = h.astype(np.float32) 

draw_raytracer(h_rotated, h_matrix, position_on_field, aimpoint,aimpoints, sun)
exit()

xi, yi = np.random.multivariate_normal(mean, cov, DIM**2//fac).T # scatter rays a bit
aimpoint_mesh_dim = 2**5 #Number of Aimpoints on Receiver
# a= aimpoints
a = np.array([aimpoints[:,np.random.randint(0,aimpoint_mesh_dim),np.random.randint(0,aimpoint_mesh_dim)] for i in range(fac)]).astype(np.float32) # draw a random aimpoint
ha_tmp = a-h[00*fac:(0+1)*fac,:] # calculate distance heliostat to aimpoint
# print(ha_tmp)

rays = np.empty((fac, DIM**2//fac, 3))



# for j in range(len(ha_tmp)):
#     ha = ha_tmp[j,:]
#     # rotate: Calculate 3D rotationmatrix in heliostat system. 1 axis is pointin towards the receiver, the other are orthogonal
#     rotate = np.array([[ha[0],ha[1],ha[2]]/np.linalg.norm([ha[0],ha[1],ha[2]]),
#                        [ha[1],-ha[0],0]/np.linalg.norm([ha[1],-ha[0],0]),
#                        [ha[2]*ha[0],ha[2]*ha[1],-ha[0]**2-ha[1]**2]/np.linalg.norm([ha[2]*ha[0],ha[2]*ha[1],-ha[0]**2-ha[1]**2])]) 
    
#     inv_rot = np.linalg.inv(rotate) #inverse matrix
#     #rays_tmp: first rotate aimpoint in right coord system, aplay xi,yi distortion, rotate back
#     rays_tmp = np.array([np.dot(inv_rot,Rz(yi[i],Ry(xi[i], np.dot(rotate,ha)))) for i in range(DIM**2//fac)]).astype(np.float32) 
#     rays[j] = rays_tmp
# rays = np.array(rays.astype(np.float32))

# kernel_dt = 0

# for point in range(points_on_hel):
#     start = timer()
#     # Execute the kernel 
#     kernel[grids, threads](a, ha_tmp, rays, d_bitmap)
#     # Copy the result from the kernel ordering the ray tracing back to host
#     bitmap = d_bitmap.copy_to_host()
    
#     total_bitmap += bitmap#np.sum(d_bitmap, axis = 2)
#     kernel_dt += timer() - start

# plt.imshow(total_bitmap, cmap='gray')


####Codeläuft durch aber Bild bleibt noch schwarz. Fac ist noch ein kleines Rätsel was das sein soll. 
#Ray generation anschauen
