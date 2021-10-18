# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:38:11 2021

@author: parg_ma
"""

import math
from matplotlib import cm
import torch as th
import numpy as np

import nurbs

def colorize(image_tensor , colormap='jet'):
    """

    Parameters
    ----------
    image_tensor : tensor
        expects tensor of shape [H,W]
    colormap : string, optional
        choose_colormap. The default is 'jet'.

    Returns
    -------
    colored image tensor of CHW

    """
    image_tensor= image_tensor.clone()/image_tensor.max()
    prediction_image = image_tensor.detach().cpu().numpy()
    
    color_map = cm.get_cmap('jet')
    mapped_image = th.tensor(color_map(prediction_image)).permute(2,1,0)
    # mapped_image8 = (255*mapped_image).astype('uint8')
    # print(colored_prediction_image.shape)
    
    
    
    
    return mapped_image

def flatten_aimpoints(aimpoints):
    X = th.flatten(aimpoints[0])
    Y = th.flatten(aimpoints[1])
    Z = th.flatten(aimpoints[2])
    aimpoints = th.stack((X, Y, Z), dim=1)
    return aimpoints




def curl(f, arg):
    jac = th.autograd.functional.jacobian(f, arg, create_graph=True)

    rot_x = jac[2][1] - jac[1][2]
    rot_y = jac[0][2] - jac[2][0]
    rot_z = jac[1][0] - jac[0][1]

    return th.tensor([rot_x, rot_y, rot_z])


def find_larger_divisor(num):
    divisor = int(th.sqrt(th.tensor(num)))
    while num % divisor != 0:
        divisor += 1
    return divisor


def find_perpendicular_pair(base_vec, vecs):
    half_pi = th.tensor(math.pi, device=vecs.device) / 2
    for vec_x in vecs[1:]:
        surface_direction_x = vec_x - base_vec
        surface_direction_x /= th.linalg.norm(surface_direction_x)
        for vec_y in vecs[2:]:
            surface_direction_y = vec_y - base_vec
            surface_direction_y /= th.linalg.norm(surface_direction_y)
            if th.isclose(
                    th.acos(th.dot(
                        surface_direction_x,
                        surface_direction_y,
                    )),
                    half_pi,
            ):
                return surface_direction_x, surface_direction_y
    raise ValueError('could not calculate surface normal')


def _cartesian_linspace_around(
        minval_x,
        maxval_x,
        num_x,
        minval_y,
        maxval_y,
        num_y,
        device,
        dtype=None,
):
    if dtype is None:
        dtype = th.get_default_dtype()
    if not isinstance(minval_x, th.Tensor):
        minval_x = th.tensor(minval_x, dtype=dtype, device=device)
    if not isinstance(maxval_x, th.Tensor):
        maxval_x = th.tensor(maxval_x, dtype=dtype, device=device)
    if not isinstance(minval_y, th.Tensor):
        minval_y = th.tensor(minval_y, dtype=dtype, device=device)
    if not isinstance(maxval_y, th.Tensor):
        maxval_y = th.tensor(maxval_y, dtype=dtype, device=device)
    spline_max = 1

    minval_x = minval_x.clamp(0, spline_max)
    maxval_x = maxval_x.clamp(0, spline_max)
    minval_y = minval_y.clamp(0, spline_max)
    maxval_y = maxval_y.clamp(0, spline_max)

    points_x = th.linspace(minval_x, maxval_x, num_x, device=device)
    points_y = th.linspace(minval_y, maxval_y, num_y, device=device)
    points = th.cartesian_prod(points_x, points_y)
    return points


def initialize_spline_eval_points(
        rows,
        cols,
        device,
):
    return _cartesian_linspace_around(0, 1, rows, 0, 1, cols, device)


def initialize_spline_eval_points_perfectly(
        points,
        degree_x,
        degree_y,
        ctrl_points,
        ctrl_weights,
        knots_x,
        knots_y,
):
    eval_points, distances = nurbs.invert_points(
            points,
            degree_x,
            degree_y,
            ctrl_points,
            ctrl_weights,
            knots_x,
            knots_y,
    )
    return eval_points


def initialize_spline_ctrl_points(
        control_points,
        origin,
        rows,
        cols,
        h_width,
        h_height,
):
    device = control_points.device
    origin_offsets_x = th.linspace(
        -h_width / 2, h_width / 2, rows, device=device)
    origin_offsets_y = th.linspace(
        -h_height / 2, h_height / 2, cols, device=device)
    origin_offsets = th.cartesian_prod(origin_offsets_x, origin_offsets_y)
    origin_offsets = th.hstack((
        origin_offsets,
        th.zeros((len(origin_offsets), 1), device=device),
    ))
    control_points[:] = (origin + origin_offsets).reshape(control_points.shape)


# def initialize_spline_ctrl_points_perfectly(control_points, points):
#     # FIXME need to sort points so normals always point in the
#     #       correct direction
#     control_points[:] = points.reshape(control_points.shape)


def initialize_spline_knots(
        knots_x,
        knots_y,
        spline_degree_x,
        spline_degree_y,
):
    num_knot_vals_x = len(knots_x[spline_degree_x:-spline_degree_x])
    knot_vals_x = th.linspace(0, 1, num_knot_vals_x)
    num_knot_vals_y = len(knots_y[spline_degree_y:-spline_degree_y])
    knot_vals_y = th.linspace(0, 1, num_knot_vals_y)
    knots_x[spline_degree_x:-spline_degree_x] = knot_vals_x
    knots_y[spline_degree_y:-spline_degree_y] = knot_vals_y


def calc_ray_diffs(pred, target):
    return th.nn.functional.l1_loss(pred, target)


def calc_reflection_normals_(in_reflections, out_reflections):
    normals = ((in_reflections + out_reflections) / 2 - in_reflections)
    # Handle pass-through "reflection"
    normals = th.where(
        th.isclose(normals, th.zeros_like(normals[0])),
        out_reflections,
        normals / normals.norm(dim=-1).unsqueeze(-1),
    )
    return normals


def calc_reflection_normals(in_reflections, out_reflections):
    in_reflections = in_reflections / in_reflections.norm(dim=-1).unsqueeze(-1)
    out_reflections = \
        out_reflections / out_reflections.norm(dim=-1).unsqueeze(-1)
    return calc_reflection_normals_(in_reflections, out_reflections)


def batch_dot(x, y):
    return (x * y).sum(-1).unsqueeze(-1)


# def reflect_rays_(rays, normals):
#     return rays - 2 * batch_dot(rays, normals) * normals


# def reflect_rays(rays, normals):
#     normals = normals / th.linalg.norm(normals, dim=-1).unsqueeze(-1)
#     return reflect_rays_(rays, normals)


def save_target(
        heliostat_origin_center,
        heliostat_face_normal,
        heliostat_points,
        heliostat_normals,
        heliostat_up_dir,

        receiver_origin_center,
        receiver_width,
        receiver_height,
        receiver_normal,
        receiver_up_dir,

        sun,
        num_rays,
        mean,
        cov,
        xi,
        yi,

        target_ray_directions,
        target_ray_points,
        path,
):
    th.save({
        'heliostat_origin_center': heliostat_origin_center,
        'heliostat_face_normal': heliostat_face_normal,
        'heliostat_points': heliostat_points,
        'heliostat_normals': heliostat_normals,
        'heliostat_up_dir': heliostat_up_dir,

        'receiver_origin_center': receiver_origin_center,
        'receiver_width': receiver_width,
        'receiver_height': receiver_height,
        'receiver_normal': receiver_normal,
        'receiver_up_dir': receiver_up_dir,

        'sun': sun,
        'num_rays': num_rays,
        'mean': mean,
        'cov': cov,
        'xi': xi,
        'yi': yi,
    }, path)
