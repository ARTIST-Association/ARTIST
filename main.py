# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 09:31:21 2019

@author: ober_lu
"""

import os

import matplotlib.pyplot as plt
from matplotlib import animation
import torch as th
import time

from utils import compute_receiver_intersections, curl, draw_raytracer, draw_heliostat, heliostat_coord_system, define_heliostat, rotate_heliostat, sample_bitmap, sample_bitmap_

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


#####Parameters#####
# grids = (int(DIM**2)//256//fac, fac//1) #cuda grid from threads , optimale anordnung
# threads = (256, 1)
seed = 0
use_gpu = True
# Using curl takes a _lot_ of memory
use_curl = False
epochs = 500
bitmap_width = 50
bitmap_height = 50


th.manual_seed(0)
device = th.device('cuda' if use_gpu and th.cuda.is_available() else 'cpu')

##Aimpoints
aimpoint = th.tensor([-50,0,0], dtype=th.float32, device=device)
planeNormal = th.tensor([1, 0, 0], dtype=th.float32, device=device) # Muss noch dynamisch gestaltet werden
##Receiver specific parameters
planex = 10 # Receiver width
planey = 10 #Receiver height

####Heliostat specific Parameters
h_width = 4 # in m
h_height = 4 # in m
rows = 64 #rows of reflection points. total number is rows**2
position_on_field = th.tensor([0,0,0], dtype=th.float32, device=device)

#sunposition
sun = th.tensor([0,1,0], dtype=th.float32, device=device)
mean = th.tensor([0, 0], dtype=th.float32, device=device)
cov = th.tensor([[0.000001, 0], [0, 0.000001]], dtype=th.float32, device=device)  # diagonal covariance, used for ray scattering
num_rays = 1000


###Define derived variables#####

sun = sun/th.linalg.norm(sun)

points_on_hel = rows**2 # reflection points on hel
hel_origin = define_heliostat(h_height, h_width, rows, points_on_hel, device)
hel_coordsystem = th.stack(heliostat_coord_system(position_on_field, sun, aimpoint))
hel_rotated = rotate_heliostat(hel_origin,hel_coordsystem, points_on_hel)
hel_in_field = hel_rotated+ position_on_field
del hel_origin
del hel_coordsystem
del hel_rotated

ray_directions =  th.tile(aimpoint- position_on_field, (len(hel_in_field), 1))


xi, yi = th.distributions.MultivariateNormal(mean, cov).sample((num_rays,)).T.to(device) # scatter rays a bit

# print(as_)





rayPoints = hel_in_field #Any point along the ray

intersections = compute_receiver_intersections(
    planeNormal,
    aimpoint, # any point on plane
    ray_directions,
    rayPoints,
    hel_in_field,
    xi,
    yi,
)

# draw_raytracer(hel_rotated.detach().cpu().numpy(),
#                 hel_coordsystem.detach().cpu().numpy(),
#                 position_on_field.detach().cpu().numpy(),
#                 aimpoint.detach().cpu().numpy(),
#                 intersections.detach().cpu().numpy(),
#                 sun.detach().cpu().numpy())

total_bitmap = sample_bitmap(intersections, planex, planey, bitmap_height, bitmap_width)




# Dataset
# When we load images, we should also normalize them here towards [0, 1]
targets = total_bitmap.detach().clone().unsqueeze(0)
print(targets[0])

# Optimization setup
# ray_directions += th.randn_like(ray_directions) * 0.1 # Durch den Invertierung gibt es schon einen minimalen Fehler, zeile kann übergangen werden bis der Algorithmus konvergiert
ray_directions.requires_grad_(True)
opt = th.optim.Adam([ray_directions], lr=3e-2)
sched = th.optim.lr_scheduler.ReduceLROnPlateau(
    opt,
    factor=0.5,
    min_lr=1e-12,
    patience=1000,
    verbose=True,
)


def loss_func(pred, target, compute_intersections, rayPoints):
    loss = th.nn.functional.l1_loss(pred, target, 0.1)
    if use_curl:
        loss -= th.sum(th.abs(curl(compute_intersections, rayPoints)))
    return loss


# Just for printing purposes
epoch_shift_width = len(str(epochs))
im = plt.imshow(total_bitmap.detach().cpu().numpy(), cmap='jet')
del total_bitmap

xi, yi = th.distributions.MultivariateNormal(mean, cov).sample((num_rays,)).T.to(device) #has to be another xi and yi as in the preprocessing
for epoch in range(epochs):
    opt.zero_grad()
    loss = 0
    # print(ray_directions)
    for target in targets:
        intersections = compute_receiver_intersections(
            planeNormal,
            aimpoint,
            ray_directions,
            rayPoints, #Raypoints und hel_in_field scheinen aktuell das gleiche zu sein
            hel_in_field, #Sobald wir mehrere Bilder verwenden stimmt das hier nicht mehr, dann sollte hier etwas stehen wie hel_in_field[target], da mehrere Bilder unterschiedlichen Sonnenstanden entsprechen und die Punkte sich dann auch woanders befinden.
            xi,
            yi,
        )
        dx_ints = intersections[:, :, 1] +planex/2
        dy_ints = intersections[:, :, 2] +planey/2
        indices = (-1 <= dx_ints) & (dx_ints < planex + 1) & (-1 <= dy_ints) & (dy_ints < planey + 1)
        pred = sample_bitmap_(dx_ints, dy_ints, indices, planex, planey, bitmap_height, bitmap_width)
        loss += loss_func(
            pred,
            target,
            lambda rayPoints: compute_receiver_intersections(
                planeNormal,
                aimpoint,
                ray_directions,
                rayPoints,
                hel_in_field,
                xi,
                yi,
            ),
            rayPoints,
        )

    if epoch %  5== 0 and not epoch == 0:#
        im.set_data(pred.detach().cpu().numpy())
        im.autoscale()
        plt.pause(0.001)  # In interactive mode, need a small delay to get the plot to appear
        plt.draw()
        # print(pred)
        # exit()


    loss /= len(targets)
    loss.backward()
    if ray_directions.grad is None or (ray_directions.grad == 0).all():
        print('no more optimization possible; ending...')
        break

    opt.step()
    sched.step(loss)
    if epoch % 10 == 0:
        num_missed = indices.numel() - indices.count_nonzero()
        print(
            f'[{epoch:>{epoch_shift_width}}/{epochs}] '
            f'loss: {loss.detach().cpu().numpy()}, '
            f'missed: {num_missed.detach().cpu().item()}'
        )

# print(ray_directions)
# draw_heliostat(hel_rotated.detach().cpu().numpy(), ray_directions.detach().cpu().numpy())

total_bitmap = sample_bitmap(intersections, planex, planey, bitmap_height, bitmap_width)
plt.imshow(total_bitmap.detach().cpu().numpy(), cmap='jet')
plt.show()
