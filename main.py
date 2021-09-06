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
import glob
from PIL import Image
import math

import nurbs
from utils import (
    add_distortion,
    calc_ray_diffs,
    calc_reflection_normals,
    calc_reflection_normals_,
    compute_receiver_intersections,
    curl,
    define_heliostat,
    find_larger_divisor,
    find_perpendicular_pair,
    heliostat_coord_system,
    initialize_spline_ctrl_points,
    initialize_spline_ctrl_points_perfectly,
    initialize_spline_eval_points,
    initialize_spline_knots,
    load_deflec,
    reflect_rays,
    reflect_rays_,
    rotate_heliostat,
    sample_bitmap,
    sample_bitmap_,
)
from plotter import plot_surface_diff, plot_normal_vectors, plot_raytracer, plot_heliostat, plot_bitmap

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


#####Parameters#####
seed = 0
use_gpu = True
# Using curl takes a _lot_ of memory
use_curl = False
#load defec settings 
load_deflec_data = True
filename = "Helio_AA33_Rim0_STRAL-Input.binp"
take_n_vectors = 2000
# NURBS settings
use_splines = True
# If setting this to `False`, be aware that the NURBS surface will
# always be evaluated at each surface position independently of the ray
# origins.
set_up_with_knowledge = True
fix_spline_ctrl_weights = True
spline_degree = 2
epochs = 200
bitmap_width = 256
bitmap_height = 256


th.manual_seed(seed)
device = th.device('cuda' if use_gpu and th.cuda.is_available() else 'cpu')

##Aimpoints
# Receiver
aimpoint = th.tensor([-25,0,0], dtype=th.float32, device=device)
planeNormal = th.tensor([1, 0, 0], dtype=th.float32, device=device) # Muss noch dynamisch gestaltet werden
# TODO add plane up vec ("rotation")
##Receiver specific parameters
planex = 10 # Receiver width
planey = 10 #Receiver height

####Heliostat specific Parameters
h_width = 4 # in m
h_height = 4 # in m 
rows = 32 #rows of reflection points. total number is rows*cols (see below)
cols = rows
# Heliostat origin center
position_on_field = th.tensor([0,0,0], dtype=th.float32, device=device)
# TODO add heliostat up vec ("rotation")

#sunposition
sun_orig = th.tensor([-1,0,0], dtype=th.float32, device=device)
mean = th.tensor([0, 0], dtype=th.float32, device=device)
cov = th.tensor([[0.000005, 0], [0, 0.000005]], dtype=th.float32, device=device)  # diagonal covariance, used for ray scattering

num_rays = 1000
ideal_normal_vec = th.tensor([0,0,1], dtype=th.float32, device=device) #valid only for planar heliostat

if not os.path.exists("images"):
    os.makedirs("images")


##Define Target Heliostat##
if load_deflec_data:
    (
        target_normal_vectors_orig,
        target_hel_origin,
        (target_h_width, target_h_height),
    ) = load_deflec(filename, take_n_vectors, device)
    ideal_normal_vecs =  th.tile(ideal_normal_vec, (len(target_hel_origin), 1)) #valid only for planar heliostat
    
    ###Plotting Stuff
    # plot_surface_diff(target_hel_origin, ideal_normal_vecs, target_normal_vectors_orig)
    # plot_normal_vectors(target_hel_origin, target_normal_vectors_orig)
    
    # TODO implement target ratio for trying to find divisor so it
    #      matches ratio between target_h_width and target_h_height
    edge_ratio = target_h_width / target_h_height
    rows = find_larger_divisor(len(target_hel_origin))
    cols = len(target_hel_origin) // rows
    if edge_ratio < 1:
        rows, cols = cols, rows
else:
    points_on_hel   = rows*cols # reflection points on hel
    points_on_hel   = th.tensor(points_on_hel, dtype=th.float32, device=device)
    target_hel_origin      = define_heliostat(h_height, h_width, rows, points_on_hel, device)
    target_normal_vector   = th.tensor([0,0,1], dtype=th.float32, device=device)
    target_normal_vectors_orig = th.tile(target_normal_vector, (len(target_hel_origin), 1))

sun = sun_orig/th.linalg.norm(sun_orig)
# TODO Max: fix for other aimpoints; need this to work inversely as well
target_hel_coords = th.stack(heliostat_coord_system(position_on_field, sun, aimpoint))
target_hel_rotated = rotate_heliostat(target_hel_origin,target_hel_coords)
target_hel_in_field = target_hel_rotated+ position_on_field

target_normal_vectors = rotate_heliostat(target_normal_vectors_orig,target_hel_coords)
target_normal_vectors /= target_normal_vectors.norm(dim=-1).unsqueeze(-1)



# del target_hel_origin
del target_hel_rotated

###Plotting Stuff
# plot_normal_vectors(target_hel_in_field, target_normal_vectors)

from_sun = position_on_field - sun
from_sun /= from_sun.norm()
from_sun = from_sun.unsqueeze(0)
target_ray_directions = reflect_rays_(from_sun, target_normal_vectors)
target_rayPoints = target_hel_in_field #Any point along the ray
# TODO Max: use for reflection instead
xi, yi = th.distributions.MultivariateNormal(mean, cov).sample((num_rays,)).T.to(device) # scatter rays a bit

intersections = compute_receiver_intersections(
    planeNormal,
    aimpoint, # any point on plane
    target_ray_directions,
    target_rayPoints,
    xi,
    yi,
)

# draw_raytracer(target_hel_in_field.detach().cpu().numpy(),
#                 target_hel_coords.detach().cpu().numpy(),
#                 position_on_field.detach().cpu().numpy(),
#                 aimpoint.detach().cpu().numpy(),
#                 intersections.detach().cpu().numpy(),
#                 sun.detach().cpu().numpy())
del sun_orig
del target_normal_vectors_orig

dx_ints, dy_ints, indices = get_intensities_and_sampling_indices(
    intersections, aimpoint, planex, planey)
target_total_bitmap = sample_bitmap_(dx_ints, dy_ints, indices, planex, planey, bitmap_height, bitmap_width)
target_num_missed = indices.numel() - indices.count_nonzero()
print('Missed for target:', target_num_missed.detach().cpu().item())
im = plt.imshow(target_total_bitmap.detach().cpu().numpy(), cmap='jet')
im.set_data(target_total_bitmap.detach().cpu().numpy())
im.autoscale()
plt.savefig(os.path.join("images", "original.jpeg"))
targets = target_total_bitmap.detach().clone().unsqueeze(0)

###Plotting Stuff
# plot_bitmap(target_total_bitmap)
del target_total_bitmap


##Define ideal heliostat.

ray_direction = (aimpoint- position_on_field)
ray_direction /= ray_direction.norm()
ray_directions =  th.tile(ray_direction, (len(target_hel_in_field), 1)) #works only for planar heliostat
# ray_directions += th.randn_like(ray_directions) * 0.3 # Da wir jetzt nicht mehr mit Idealen Heliostaten Rechnen kann das eigentlich weg, es schadet aber glaube ich auch nicht
rayPoints = target_hel_in_field #maybe define the ideal heliostat on its own




# Dataset
# When we load images, we should also normalize them here towards [0, 1]


# Optimization setup

# TODO große winkel zwischen vektoren bestrafen
if use_splines:
    (ctrl_points, ctrl_weights, knots_x, knots_y) = nurbs.setup_nurbs_surface(
        spline_degree, spline_degree, rows, cols, device)
    eval_points = initialize_spline_eval_points(rows, cols, device)

    if set_up_with_knowledge:
        initialize_spline_ctrl_points_perfectly(
            ctrl_points,
            target_hel_origin,
        )
    else:
        # Use perfect, unrotated heliostat at `position_on_field` as
        # starting point with width and height as initially guessed.
        initialize_spline_ctrl_points(
            ctrl_points,
            position_on_field,
            rows,
            cols,
            h_width,
            h_height,
        )

    ctrl_points_xy = ctrl_points[:, :-1]
    ctrl_points_z = ctrl_points[:, -1:]
    ctrl_points_z.requires_grad_(True)
    opt_params = [ctrl_points_z]
    if fix_spline_ctrl_weights:
        ctrl_weights[:] = 1
    else:
        ctrl_weights.requires_grad_(True)
        opt_params.append(ctrl_weights)

    initialize_spline_knots(knots_x, knots_y, spline_degree, spline_degree)

    # knots_x.requires_grad_(True)
    # knots_y.requires_grad_(True)
    # opt_params.extend([knots_x, knots_y])
    opt = th.optim.Adam(opt_params, lr=3e-6, weight_decay=0.01)
else:
    ray_directions.requires_grad_(True)
    opt = th.optim.Adam([ray_directions], lr=3e-2, weight_decay=0.1)
sched = th.optim.lr_scheduler.ReduceLROnPlateau(
    opt,
    factor=0.5,
    min_lr=1e-12,
    patience=10,
    verbose=True,
)


def loss_func(pred, target, compute_intersections, rayPoints):
    loss = th.nn.functional.mse_loss(pred, target, 0.1)
    if use_curl:
        curls = th.stack([
            curl(compute_intersections, rayPoints_)
            for rayPoints_ in rayPoints
        ])
        loss += th.sum(th.abs(curls))
    return loss


# Just for printing purposes


epoch_shift_width = len(str(epochs))

xi, yi = th.distributions.MultivariateNormal(mean, cov).sample((num_rays,)).T.to(device) #has to be another xi and yi as in the preprocessing
for epoch in range(epochs):
    opt.zero_grad()
    loss = 0
    # print(ray_directions)
    for target in targets:
        if use_splines:
            ctrl_points = th.hstack((ctrl_points_xy, ctrl_points_z))
            hel_origin, surface_normals = (
                nurbs.calc_normals_and_surface_slow(
                    eval_points[:, 0],
                    eval_points[:, 1],
                    spline_degree,
                    spline_degree,
                    ctrl_points,
                    ctrl_weights,
                    knots_x,
                    knots_y,
                )
            )

            hel_rotated = rotate_heliostat(hel_origin, target_hel_coords)
            rayPoints = hel_rotated + position_on_field

            surface_normals = rotate_heliostat(surface_normals, target_hel_coords)
            surface_normals = surface_normals / surface_normals.norm(dim=-1).unsqueeze(-1)
            ray_directions = reflect_rays_(from_sun, surface_normals)
        intersections = compute_receiver_intersections(
            planeNormal,
            aimpoint,
            ray_directions,
            rayPoints, 
            xi,
            yi,
        )
        dx_ints, dy_ints, indices = get_intensities_and_sampling_indices(
            intersections, aimpoint, planex, planey)
        pred = sample_bitmap_(dx_ints, dy_ints, indices, planex, planey, bitmap_height, bitmap_width)
        loss += loss_func(
            pred,
            target,
            lambda rayPoints: compute_receiver_intersections(
                planeNormal,
                aimpoint,
                ray_directions,
                rayPoints,
                xi,
                yi,
            ),
            rayPoints,
        )
    if epoch %  10== 0:#
        im.set_data(pred.detach().cpu().numpy())
        im.autoscale()
        plt.savefig(os.path.join("images", f"{epoch}.png"))


    loss /= len(targets)
    loss.backward()
    if not use_splines:
        if ray_directions.grad is None or (ray_directions.grad == 0).all():
            print('no more optimization possible; ending...')
            break

    opt.step()
    sched.step(loss)
    if epoch % 1 == 0:
        num_missed = indices.numel() - indices.count_nonzero()
        ray_diff = calc_ray_diffs(
            ray_directions.detach(),
            target_ray_directions,
        )
        print(
            f'[{epoch:>{epoch_shift_width}}/{epochs}] '
            f'loss: {loss.detach().cpu().numpy()}, '
            f'missed: {num_missed.detach().cpu().item()}, '
            f'ray differences: {ray_diff.detach().cpu().item()}'
        )

# Save trained model and optimizer state
if use_splines:
    model_name = 'nurbs'
    save_data = {
        'degree_x': spline_degree,
        'degree_y': spline_degree,
        'ctrl_points': ctrl_points,
        'ctrl_weights': ctrl_weights,
        'knots_x': knots_x,
        'knots_y': knots_y,
    }
else:
    model_name = 'ray_dirs'
    save_data = {'ray_directions': ray_directions}
th.save(save_data, f'{model_name}.pt')
th.save({'opt': opt.state_dict()}, f'{model_name}_opt.pt')

fp_in = "images/*.png"
fp_out = "images/results.gif"
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
print(img)
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=500, loop=1)

total_bitmap = sample_bitmap(intersections, aimpoint, planex, planey, bitmap_height, bitmap_width)


# plot_surface_diff(target_hel_origin, ideal_normal_vectors, predicted_normal_vectors) #predicted normal vectos has to be calculated from final raydirections
# plot_surface_diff(target_hel_origin, target_normal_vectors, predicted_normal_vectors) #
