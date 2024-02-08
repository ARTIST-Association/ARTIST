import math
from typing import List, Optional, Tuple, TypeVar, Union, cast

import torch

# We would like to say that T can be everything but a list.
T = TypeVar("T")


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
    if spline_degree_x <= 0:
        raise ValueError("Spline degree must be a positive integer.")
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
        Number of columns.
    h_width : float
        Width of the heliostat.
    h_height : float
        Height of the heliostat.
    """
    device = control_points.device
    origin_offsets_x = torch.linspace(-h_width / 2, h_width / 2, rows, device=device)
    origin_offsets_y = torch.linspace(-h_height / 2, h_height / 2, cols, device=device)
    origin_offsets = torch.cartesian_prod(origin_offsets_x, origin_offsets_y)
    origin_offsets = torch.hstack(
        (
            origin_offsets,
            torch.zeros((len(origin_offsets), 1), device=device),
        )
    )
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
        Number of columns.
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
    Compute the initialized evaluation points of the spline.

    Parameters
    ----------
    minval_x : Union[float, torch.Tensor]
        Minimum value for x.
    maxval_x : Union[float, torch.Tensor]
        Maximum value for x.
    num_x: int
        Number of rows.
    minval_y: Union[float, torch.Tensor]
        Minimum value for y.
    maxval_y: Union[float, torch.Tensor]
        Maximum value for y.
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
        minval_x, maxval_x, num_x, device=device
    )
    points_y = torch.linspace(
        minval_y, maxval_y, num_y, device=device
    )
    points = torch.cartesian_prod(points_x, points_y)
    return points


def with_outer_list(values: Union[List[T], List[List[T]]]) -> List[List[T]]:
    # Type errors come from T being able to be a list. So we ignore them
    # as "type negation" ("T can be everything except a list") is not
    # currently supported.

    if isinstance(values[0], list):
        return cast(List[List[T]], values)
    return cast(List[List[T]], [values])


def axis_angle_rotation(
    axis: torch.Tensor,
    angle_rad: torch.Tensor,
) -> torch.Tensor:
    """
    Rotate the axis by a specified angle.

    Parameters
    ----------
    axis : torch.Tensor
        The axis to be rotated.
    angle_rad : torch.Tensor
        The angle in radians by which the axis is rotated.

    Returns
    -------
    torch.Tensor
        The rotated axis.
    """
    cos = torch.cos(angle_rad)
    icos = 1 - cos
    sin = torch.sin(angle_rad)
    x = axis[..., 0]
    y = axis[..., 1]
    z = axis[..., 2]
    axis_sq = axis**2

    rows = [
        torch.stack(row, dim=-1)
        for row in [
            [
                cos + axis_sq[..., 0] * icos,
                x * y * icos - z * sin,
                x * z * icos + y * sin,
            ],
            [
                y * x * icos + z * sin,
                cos + axis_sq[..., 1] * icos,
                y * z * icos - x * sin,
            ],
            [
                z * x * icos - y * sin,
                z * y * icos + x * sin,
                cos + axis_sq[..., 2] * icos,
            ],
        ]
    ]
    return torch.stack(rows, dim=1)


def get_rot_matrix(
    start: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute a rotation matrix to rotate from start to target.

    Parameters
    ----------
    start : torch.Tensor
        Starting direction.
    target : torch.Tensor
        Target direction.

    Returns
    -------
    torch.Tensor
        The rotation matrix.
    """
    rot_angle = angle_between(start, target)
    # Handle parallel start/target normals.
    if rot_angle == 0:
        return torch.eye(3)
    elif rot_angle == math.pi:
        return -torch.eye(3)
    rot_axis = torch.cross(target, start)
    rot_axis /= torch.linalg.norm(rot_axis)
    full_rot = axis_angle_rotation(rot_axis, rot_angle)
    return full_rot


def angle_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute the angle between direction a and b.

    Parameters
    ----------
    a : torch.Tensor
        Starting direction.
    b : torch.Tensor
        Target direction.

    Returns
    -------
    torch.Tensor
        The angle.
    """
    angles = torch.acos(
        torch.clamp(
            (
                batch_dot(a, b).squeeze(-1)
                / (torch.linalg.norm(a, dim=-1) * torch.linalg.norm(b, dim=-1))
            ),
            -1.0,
            1.0,
        )
    ).squeeze(-1)
    return angles


def make_structured_points(
    points: torch.Tensor,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    tolerance: float = 0.0075,
) -> Tuple[torch.Tensor, int, int]:
    """
    Structure the given points.

    Parameters
    ----------
    points : torch.tensor
        The points to be structured.
    rows : Optional[int] = None
        The number of rows.
    cols : Optional[int] = None
        The number of columns.
    tolerance : float = 0.0075
        The tolerance.

    Returns
    -------
    Tuple[torch.Tensor, int, int]
        Return the structured points

    """
    if rows is None or cols is None:
        return _make_structured_points_from_unique(points, tolerance)
    else:
        return _make_structured_points_from_corners(points, rows, cols)


def _make_structured_points_from_unique(
    points: torch.Tensor,
    tolerance: float,
) -> Tuple[torch.Tensor, int, int]:
    x_vals = points[:, 0]
    x_vals = torch.unique(x_vals, sorted=True)

    prev_i = 0
    keep_indices = [prev_i]
    for i, x in enumerate(x_vals[1:]):
        if not torch.isclose(x, x_vals[prev_i], atol=tolerance):
            prev_i = i
            keep_indices.append(prev_i)
    x_vals = x_vals[keep_indices]

    y_vals = points[:, 0]
    y_vals = torch.unique(y_vals, sorted=True)

    prev_i = 0
    keep_indices = [prev_i]
    for i, y in enumerate(y_vals[1:]):
        if not torch.isclose(y, y_vals[prev_i], atol=tolerance):
            prev_i = i
            keep_indices.append(prev_i)
    y_vals = y_vals[keep_indices]

    structured_points = torch.cartesian_prod(x_vals, y_vals)

    distances = torch.linalg.norm(
        (structured_points.unsqueeze(1) - points[:, :-1].unsqueeze(0)),
        dim=-1,
    )
    argmin_distances = distances.argmin(1)
    z_vals = points[argmin_distances, -1]
    structured_points = torch.cat([structured_points, z_vals.unsqueeze(-1)], dim=-1)

    rows = len(x_vals)
    cols = len(y_vals)
    return structured_points, rows, cols


def _make_structured_points_from_corners(
    points: torch.Tensor,
    rows: int,
    cols: int,
) -> Tuple[torch.Tensor, int, int]:
    x_vals = points[:, 0]
    y_vals = points[:, 1]

    x_min = x_vals.min()
    x_max = x_vals.max()
    y_min = y_vals.min()
    y_max = y_vals.max()

    x_vals = torch.linspace(
        x_min, x_max, rows, device=x_vals.device
    )  # type: ignore[arg-type]
    y_vals = torch.linspace(
        y_min, y_max, cols, device=y_vals.device
    )  # type: ignore[arg-type]

    structured_points = torch.cartesian_prod(x_vals, y_vals)

    distances = torch.linalg.norm(
        (structured_points.unsqueeze(1) - points[:, :-1].unsqueeze(0)),
        dim=-1,
    )
    argmin_distances = distances.argmin(1)
    z_vals = points[argmin_distances, -1]
    structured_points = torch.cat([structured_points, z_vals.unsqueeze(-1)], dim=-1)

    return structured_points, rows, cols


def axis_angle_rotation(
    axis: torch.Tensor,
    angle_rad: torch.Tensor,
) -> torch.Tensor:
    cos = torch.cos(angle_rad)
    icos = 1 - cos
    sin = torch.sin(angle_rad)
    x = axis[..., 0]
    y = axis[..., 1]
    z = axis[..., 2]
    axis_sq = axis**2

    rows = [
        torch.stack(row, dim=-1)
        for row in [
            [
                cos + axis_sq[..., 0] * icos,
                x * y * icos - z * sin,
                x * z * icos + y * sin,
            ],
            [
                y * x * icos + z * sin,
                cos + axis_sq[..., 1] * icos,
                y * z * icos - x * sin,
            ],
            [
                z * x * icos - y * sin,
                z * y * icos + x * sin,
                cos + axis_sq[..., 2] * icos,
            ],
        ]
    ]
    return torch.stack(rows, dim=1)
