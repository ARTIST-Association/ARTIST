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
        List of numbers representing the knots in x dimension.
    knots_y : torch.Tensor
        List of numbers representing the knots in y dimension.
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


def initialize_spline_ctrl_points(
        control_points: torch.Tensor,
        origin: torch.Tensor,
        rows: int,
        cols: int,
        h_width: float,
        h_height: float,
) -> None:
    """
    Initialize the spline control points.

    Parameters
    ----------
    control_points : torch.Tensor
        The control points.
    origin : torch.Tensor
        Initialize at the origin where the heliostat's discrete points are as well.
    rows : int
        Number of rows.
    cols : int
        Number of columns
    h_width : float
        Width of the heliostat.
    h_height : float
        Height of the heliostat.
    """
    device = control_points.device
    origin_offsets_x = torch.linspace(
        -h_width / 2, h_width / 2, rows, device=device)
    origin_offsets_y = torch.linspace(
        -h_height / 2, h_height / 2, cols, device=device)
    origin_offsets = torch.cartesian_prod(origin_offsets_x, origin_offsets_y)
    origin_offsets = torch.hstack((
        origin_offsets,
        torch.zeros((len(origin_offsets), 1), device=device),
    ))
    control_points[:] = (origin + origin_offsets).reshape(control_points.shape)


def initialize_spline_eval_points(
        rows: int,
        cols: int,
        device: torch.device,
) -> torch.Tensor:
    """
    Initialize the spline evaluation points.

    Parameters
    ----------
    rows : int
        Number of rows.
    cols : int
        Number of columns
    device : torch.device
        Specifies the device type responsible to load tensors into memory.

    Returns
    -------
    torch.Tensor
        The evaluation points of the spline.
    """
    return _cartesian_linspace_around(0, 1, rows, 0, 1, cols, device)

def _cartesian_linspace_around(
        minval_x: Union[float, torch.Tensor],
        maxval_x: Union[float, torch.Tensor],
        num_x: int,
        minval_y: Union[float, torch.Tensor],
        maxval_y: Union[float, torch.Tensor],
        num_y: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Comute the initialized evaluation points of the spline.

    Parameters
    minval_x : Union[float, torch.Tensor]
        Minimum value for x.
    maxval_x : Union[float, torch.Tensor]
        Maximum value for x
    num_x: int
        Number of rows.
    minval_y: Union[float, torch.Tensor]
        Minimum value for y.
    maxval_y: Union[float, torch.Tensor]
        Maximum value for y    
    num_y: int
        Number of columns
    device: torch.device
        Specifies the device type responsible to load tensors into memory.
    dtype: Optional[torch.dtype] = None
        The data type of the tensor.

    Returns
    -------
    torch.Tensor
        The initialized evaluation points of the spline.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if not isinstance(minval_x, torch.Tensor):
        minval_x = torch.tensor(minval_x, dtype=dtype, device=device)
    if not isinstance(maxval_x, torch.Tensor):
        maxval_x = torch.tensor(maxval_x, dtype=dtype, device=device)
    if not isinstance(minval_y, torch.Tensor):
        minval_y = torch.tensor(minval_y, dtype=dtype, device=device)
    if not isinstance(maxval_y, torch.Tensor):
        maxval_y = torch.tensor(maxval_y, dtype=dtype, device=device)
    spline_max = 1

    minval_x = minval_x.clamp(0, spline_max)
    maxval_x = maxval_x.clamp(0, spline_max)
    minval_y = minval_y.clamp(0, spline_max)
    maxval_y = maxval_y.clamp(0, spline_max)

    points_x = torch.linspace(
        minval_x, maxval_x, num_x, device=device)  # type: ignore[arg-type]
    points_y = torch.linspace(
        minval_y, maxval_y, num_y, device=device)  # type: ignore[arg-type]
    points = torch.cartesian_prod(points_x, points_y)
    return points
