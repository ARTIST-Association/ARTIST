from typing import List, Optional, Tuple, Type, TypeVar, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch as th

def setup_nurbs_surface(
        degree_x: int,
        degree_y: int,
        num_control_points_x: int,
        num_control_points_y: int,
        device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute uninitialized parameters for a NURBS surface with the desired properties on the given device.

    Parameters
    ----------
    degree_x : int
        Spline degree in x direction.
    degree_y : int
        Spline degree in y direction.
    num_control_points_x : int
        Number of control points in x direction, (rows).
    num_control_points_y : int
        Number of control points in y direction, (columns).
    device : torch.device
        Specifies the device type responsible to load tensors into memory.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        The control points, control weights, and knots in x and y direction.
    """
    next_degree_x = degree_x + 1
    next_degree_y = degree_y + 1
    assert num_control_points_x > degree_x, \
        f'need at least {next_degree_x} control points in x direction'
    assert num_control_points_y > degree_y, \
        f'need at least {next_degree_y} control points in y direction'

    control_points_shape = (num_control_points_x, num_control_points_y)
    control_points = th.empty(
        control_points_shape + (3,),
        device=device,
    )
    # to get b-splines, set these weights to all ones
    control_point_weights = th.empty(
        control_points_shape + (1,),
        device=device,
    )
    control_point_weights.clamp_(1e-8, th.finfo().max)

    knots_x = th.zeros(num_control_points_x + next_degree_x, device=device)
    # knots_x[:next_degree_x] = 0
    knots_x[next_degree_x:-next_degree_x] = 0.5
    knots_x[-next_degree_x:] = 1

    knots_y = th.zeros(num_control_points_y + next_degree_y, device=device)
    # knots_y[:next_degree_y] = 0
    knots_y[next_degree_y:-next_degree_y] = 0.5
    knots_y[-next_degree_y:] = 1
    return control_points, control_point_weights, knots_x, knots_y

