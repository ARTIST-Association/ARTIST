from typing import Optional, Tuple, Union
import torch
from ..physics_objects.heliostats.surface.nurbs import nurbs


def batch_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x * y).sum(-1).unsqueeze(-1)

def initialize_spline_knots(
        knots_x: torch.Tensor,
        knots_y: torch.Tensor,
        spline_degree_x: int,
        spline_degree_y: int,
) -> None:
    """
    Initialize the spline knots in x and y direction.

    Parameters
    ----------
    knots_x : torch.Tensor
        List of numbers representing the knots in x direction.
    knots_y : torch.Tensor
        List of numbers representing the knots in y direction.
    spline_degree_x : int
        Spline degree in x direction.
    spline_degree_y : int 
        Spline degree in y direction.
    """
    initialize_spline_knots_(knots_x, spline_degree_x)
    initialize_spline_knots_(knots_y, spline_degree_y)


def initialize_spline_knots_(knots: torch.Tensor, spline_degree: int) -> None:
    """
    Initialize spline knots.

    Knots is a list of numbers with (spline degree + number of control points - 1) entries.

    Parameters
    ----------
    knots : torch.Tensor
        List of numbers representing the knots.
    spline_degree_x : int
        Spline degree, positive integer.
    """
    num_knot_vals = len(knots[spline_degree:-spline_degree])
    knot_vals = torch.linspace(0, 1, num_knot_vals)
    knots[:spline_degree] = 0
    knots[spline_degree:-spline_degree] = knot_vals
    knots[-spline_degree:] = 1