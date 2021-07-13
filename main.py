# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 09:31:21 2019

@author: ober_lu
"""

import os
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import torch as th

from utils import draw_raytracer, Rx, Ry, Rz, heliostat_coord_system,LinePlaneCollision, define_heliostat, invert_bitmap, rotate_heliostat, sort_indices, to_prediction

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


#####Parameters#####
# grids = (int(DIM**2)//256//fac, fac//1) #cuda grid from threads , optimale anordnung
# threads = (256, 1)
seed = 0
use_gpu = True
epochs = 200
bitmap_width = 50
bitmap_height = 50

th.manual_seed(0)
device = th.device('cuda' if use_gpu and th.cuda.is_available() else 'cpu')

##Aimpoints
aimpoint = th.tensor([-50,0,0], dtype=th.float32, device=device)

##Receiver specific parameters
planex = 10 # Receiver width
planey = 10 #Receiver height

####Heliostat specific Parameters
h_width = 4 # in m
h_height = 4 # in m
rows = 64 #rows of reflection points. total number is rows**2
position_on_field = th.tensor([0,0,0], dtype=th.float32, device=device)

#sunposition
sun = th.tensor([0,1,1], dtype=th.float32, device=device)
mean = th.tensor([0, 0], dtype=th.float32, device=device)
cov = th.tensor([[0.000001, 0], [0, 0.000001]], dtype=th.float32, device=device)  # diagonal covariance, used for ray scattering
num_rays = 1000


###Define derived variables#####

total_bitmap = th.zeros([bitmap_height, bitmap_width], dtype=th.float32, device=device) # Flux density map for heliostat field



sun = sun/th.linalg.norm(sun)

start = timer()
points_on_hel = rows**2 # reflection points on hel
hel_origin = define_heliostat(h_height, h_width, rows, points_on_hel, device)
hel_coordsystem = th.stack(heliostat_coord_system(position_on_field, sun, aimpoint))
hel_rotated = rotate_heliostat(hel_origin,hel_coordsystem, points_on_hel)
hel_in_field = hel_rotated+ position_on_field

ray_directions =  th.tile(aimpoint- position_on_field, (len(hel_in_field), 1))
init_dt = timer() - start
print(f'initialization took {init_dt} secs')




xi, yi = th.distributions.MultivariateNormal(mean, cov).sample((num_rays,)).T.to(device) # scatter rays a bit

# print(as_)
# draw_raytracer(hel_rotated, hel_coordsystem, position_on_field, aimpoint,aimpoints, sun)
        # print("Ray directioN", rayDirection)

start = timer()
planeNormal = th.tensor([1, 0, 0], dtype=th.float32, device=device) # Muss noch dynamisch gestaltet werden
planePoint = aimpoint #Any point on the plane


rayPoints = hel_in_field #Any point along the ray

intersections = LinePlaneCollision(planeNormal, planePoint, ray_directions, rayPoints)
as_ = intersections
has = as_-hel_in_field

# rotate: Calculate 3D rotationmatrix in heliostat system. 1 axis is pointin towards the receiver, the other are orthogonal
rotates_x = th.hstack(list(map(lambda t: t.unsqueeze(-1), [has[:, 0],has[:, 1],has[:, 2]])))
rotates_x /= th.linalg.norm(rotates_x, dim=-1).unsqueeze(-1)
rotates_y = th.hstack(list(map(lambda t: t.unsqueeze(-1), [has[:, 1],-has[:, 0],th.zeros(has.shape[:1], device=device)])))
rotates_y /= th.linalg.norm(rotates_y, dim=-1).unsqueeze(-1)
rotates_z = th.hstack(list(map(lambda t: t.unsqueeze(-1), [has[:, 2]*has[:, 0],has[:, 2]*has[:, 1],-has[:, 0]**2-has[:, 1]**2])))
rotates_z /= th.linalg.norm(rotates_z, dim=-1).unsqueeze(-1)
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
rays_dt = timer() - start
print(f'ray calculations took {rays_dt} secs')


# rays = rays.to(th.float32)
kernel_dt = 0
start = timer()

# Execute the kernel
intersections = LinePlaneCollision(planeNormal, planePoint, rays, hel_in_field, epsilon=1e-6)
# print(intersection)
dx_ints = intersections[:, :, 1] +planex/2
dy_ints = intersections[:, :, 2] +planey/2
# checks the points of intersection  and chooses bins in bitmap
indices = ( (0 <= dx_ints) & (dx_ints < planex) & (0 <= dy_ints) & (dy_ints < planey))

x_int = (dx_ints[indices]/planex*bitmap_height).long()
y_int = (dy_ints[indices]/planey*bitmap_width).long()
total_bitmap.index_put_(
    (x_int, y_int),
    th.ones(len(x_int), dtype=th.float32, device=device),
    accumulate=True,
)


kernel_dt += timer() - start

print(f'kernel calculations took {kernel_dt} secs')

# Dataset
target_imgs = total_bitmap.detach().clone().unsqueeze(0)

targets = th.stack(list(map(
    lambda img: sort_indices(
        invert_bitmap(img, planex, planey, bitmap_height, bitmap_width),
        bitmap_height,
    ),
    target_imgs,
)))
pred = to_prediction(intersections, bitmap_height)
print('inversion error', th.nn.functional.mse_loss(pred, targets[0]))

# Optimization setup
ray_directions += th.randn_like(ray_directions) * 0.1
ray_directions.requires_grad_(True)
opt = th.optim.Adam([ray_directions], lr=3e-2)
sched = th.optim.lr_scheduler.ReduceLROnPlateau(
    opt,
    factor=0.5,
    min_lr=1e-7,
    verbose=True,
)

# Just for printing purposes
epoch_shift_width = len(str(epochs))
for epoch in range(epochs):
    opt.zero_grad()
    loss = 0

    for target in targets:
        intersections = LinePlaneCollision(planeNormal, planePoint, ray_directions, rayPoints)
        as_ = intersections
        has = as_-hel_in_field

        # rotate: Calculate 3D rotationmatrix in heliostat system. 1 axis is pointin towards the receiver, the other are orthogonal
        rotates_x = th.hstack(list(map(lambda t: t.unsqueeze(-1), [has[:, 0],has[:, 1],has[:, 2]])))
        rotates_x = rotates_x / th.linalg.norm(rotates_x, dim=-1).unsqueeze(-1)
        rotates_y = th.hstack(list(map(lambda t: t.unsqueeze(-1), [has[:, 1],-has[:, 0],th.zeros(has.shape[:1], device=device)])))
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
        # print(intersection)

        pred = to_prediction(intersections, bitmap_height)
        loss += th.nn.functional.mse_loss(pred, target)


    loss /= len(targets)
    loss.backward()
    if ray_directions.grad is None or (ray_directions.grad == 0).all():
        print('no more optimization possible; ending...')
        break

    opt.step()
    sched.step(loss)
    if epoch % 10 == 0:
        indices = (0 <= dx_ints) & (dx_ints < planex) & (0 <= dy_ints) & (dy_ints < planey)
        num_missed = indices.numel() - indices.count_nonzero()
        print(
            f'[{epoch:>{epoch_shift_width}}/{epochs}] '
            f'loss: {loss.detach().cpu().numpy()}, '
            f'missed: {num_missed.detach().cpu().item()}'
        )


total_bitmap.fill_(0)

x_int = x_int.long()
y_int = y_int.long()
total_bitmap.index_put_(
    (x_int, y_int),
    th.ones(len(x_int), dtype=th.float32, device=device),
    accumulate=True,
)

plt.imshow(total_bitmap.detach().cpu().numpy(), cmap='jet')
plt.show()
