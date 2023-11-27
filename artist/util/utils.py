import math
from typing import List, Optional, Tuple, TypeVar, Union, cast
import torch
from ..physics_objects.heliostats.surface.nurbs import nurbs

# We would like to say that T can be everything but a list.
T = TypeVar('T')

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

def with_outer_list(values: Union[List[T], List[List[T]]]) -> List[List[T]]:
    # Type errors come from T being able to be a list. So we ignore them
    # as "type negation" ("T can be everything except a list") is not
    # currently supported.

    if isinstance(values[0], list):
        return cast(List[List[T]], values)
    return cast(List[List[T]], [values])

def deflec_facet_zs_many(
        points: torch.Tensor,
        normals: torch.Tensor,
        normals_ideal: torch.Tensor,
        num_samples: int = 4,
        use_weighted_average: bool = False,
        eps: float = 1e-6,
) -> torch.Tensor:
    """
    Calculate z values for a surface given by normals at x-y-planar
    positions.

    We are left with a different unknown offset for each z value; we
    assume this to be constant.
    
    Parameters
    ----------
    points : torch.Tensor
        Points on the given surface.
    normals : torch.Tensor 
        Normal vectors coressponding to the points.
    normals_ideal : torch.Tensor
        Ideal normal vectors.
    num_samples : int = 4
        Number of samples.
    use_weighted_average : bool = False
        Wether or not to use weighted averages.
    eps : float = 1e-6
        Cut-off value/ limit.
    
    Returns
    -------
    torch.Tensor
        The z values.
    """
    # TODO When `num_samples == 1`, we can just use the old method.
    device = points.device
    dtype = points.dtype

    distances = horizontal_distance(
        points.unsqueeze(0),
        points.unsqueeze(1),
    )
    distances, distance_sorted_indices = distances.sort(dim=-1)
    del distances
    # Take closest point that isn't the point itself.
    closest_indices = distance_sorted_indices[..., 1]

    # Take closest point in different directions from the given point.

    # For that, first calculate angles between direction to closest
    # point and all others, sorted by distance.
    angles = _all_angles(
        points,
        normals,
        closest_indices,
        distance_sorted_indices[..., 2:],
    ).unsqueeze(0)

    # Find positions of all angles in each slice except the zeroth one.
    angles_in_slice, angle_slices = _find_angles_in_other_slices(
        angles, num_samples)

    # And take the first one.angle we found in each slice. Remember
    # these are still sorted by distance, so we obtain the first
    # matching angle that is also closest to the desired point.
    #
    # We need to handle not having any slices except the zeroth one
    # extra.
    if len(angles_in_slice) > 1:
        angle_indices = torch.argmax(angles_in_slice.long(), dim=-1)
    else:
        angle_indices = torch.empty(
            (0, len(points)), dtype=torch.long, device=device)

    # Select the angles we found for each slice.
    angles = torch.gather(angles.squeeze(0), -1, angle_indices.T)

    # Handle _not_ having found an angle. We here create an array of
    # booleans, indicating whether we found an angle, for each slice.
    found_angles = torch.gather(
        angles_in_slice,
        -1,
        angle_indices.unsqueeze(-1),
    ).squeeze(-1)
    # We always found something in the zeroth slice, so add those here.
    found_angles = torch.cat([
        torch.ones((1,) + found_angles.shape[1:], dtype=torch.bool, device=device),
        found_angles,
    ], dim=0)
    del angles_in_slice

    # Set up some numbers for averaging.
    if use_weighted_average:
        angle_diffs = (
            torch.cat([
                torch.zeros((len(angles), 1), dtype=dtype, device=device),
                angles,
            ], dim=-1)
            - angle_slices.squeeze(-1).T
        )
        # Inverse difference in angle.
        weights = 1 / (angle_diffs + eps).T
        del angle_diffs
    else:
        # Number of samples we found angles for.
        num_available_samples = torch.count_nonzero(found_angles, dim=0)

    # Finally, combine the indices of the closest points (zeroth slice)
    # with the indices of all closest points in the other slices.
    closest_indices = torch.cat((
        closest_indices.unsqueeze(0),
        angle_indices,
    ), dim=0)
    del angle_indices, angle_slices

    midway_normal = normals + normals[closest_indices]
    midway_normal /= torch.linalg.norm(midway_normal, dim=-1, keepdims=True)

    rot_90deg = axis_angle_rotation(
        normals_ideal, torch.tensor(math.pi / 2, dtype=dtype, device=device))

    connector = points[closest_indices] - points
    connector_norm = torch.linalg.norm(connector, dim=-1)
    orthogonal = torch.matmul(
        rot_90deg.unsqueeze(0),
        connector.unsqueeze(-1),
    ).squeeze(-1)
    orthogonal /= torch.linalg.norm(orthogonal, dim=-1, keepdims=True)
    tilted_connector = torch.cross(orthogonal, midway_normal, dim=-1)
    tilted_connector /= torch.linalg.norm(tilted_connector, dim=-1, keepdims=True)
    tilted_connector *= torch.sign(connector[..., -1]).unsqueeze(-1)

    angle = torch.acos(torch.clamp(
        (
            batch_dot(tilted_connector, connector).squeeze(-1)
            / connector_norm
        ),
        -1,
        1,
    ))
    # Here, we handle values for which we did not find an angle. For
    # some reason, the NaNs those create propagate even to supposedly
    # unaffected values, so we handle them explicitly.
    angle = torch.where(
        found_angles & ~torch.isnan(angle),
        angle,
        torch.tensor(0.0, dtype=dtype, device=device),
    )

    # Average over each slice.
    if use_weighted_average:
        zs = (
            (weights * connector_norm * torch.tan(angle)).sum(dim=0)
            / (weights * found_angles.to(dtype)).sum(dim=0)
        )
    else:
        zs = (
            (connector_norm * torch.tan(angle)).sum(dim=0)
            / num_available_samples
        )

    return zs

def horizontal_distance(
        a: torch.Tensor,
        b: torch.Tensor,
        ord: Union[int, float, str] = 2,
) -> torch.Tensor:
    """Return the horizontal distance between a and b"""
    return torch.linalg.norm(b[..., :-1] - a[..., :-1], dim=-1, ord=ord)


def _all_angles(
        points: torch.Tensor,
        normals: torch.Tensor,
        closest_indices: torch.Tensor,
        remaining_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate angles between direction to closest point and all others

    Parameters
    ----------
    points : torch.Tensor
        The points to be considered.
    normals : torch.Tensor
        The normals corresponding to the points.
    closest_indices : torch.Tensor
        Indices of the closest points.
    remaining_indices : torch.Tensor
        Indices of the remaining points.
    
    Returns
    -------
    torch.Tensor
        The angles between the direction of the closest point and all others.
    """
    connector = (points[closest_indices] - points).unsqueeze(1)
    other_connectors = (
        points[remaining_indices]
        - points.unsqueeze(1)
    )
    angles = torch.acos(torch.clamp(
        (
            batch_dot(connector, other_connectors).squeeze(-1)
            / (
                torch.linalg.norm(connector, dim=-1)
                * torch.linalg.norm(other_connectors, dim=-1)
            )
        ),
        -1,
        1,
    )).squeeze(-1)

    # Give the angles a rotation direction.
    angles *= (
        1
        - 2 * (
            batch_dot(
                normals.unsqueeze(1),
                # Cross product does not support broadcasting, so do it
                # manually.
                torch.cross(
                    torch.tile(connector, (1, other_connectors.shape[1], 1)),
                    other_connectors,
                    dim=-1,
                ),
            ).squeeze(-1)
            < 0
        )
    )

    # And convert to 360° rotations.
    tau = 2 * torch.tensor(math.pi, dtype=angles.dtype, device=angles.device)
    angles = torch.where(angles < 0, tau + angles, angles)
    return angles


def _find_angles_in_other_slices(
        angles: torch.Tensor,
        num_slices: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find positions of all angles in each slice

    Parameters
    ----------
    angles : torch.Tensor
        The angles.
    num_slices : int 
        The number of slices.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The positions of all angles ine ach slice and all the slices.
    """
    dtype = angles.dtype
    device = angles.device
    # Set up uniformly sized cake/pizza slices for which to find angles.
    tau = 2 * torch.tensor(math.pi, dtype=dtype, device=device)
    angle_slice = tau / num_slices

    angle_slices = (
        torch.arange(
            num_slices,
            dtype=dtype,
            device=device,
        )
        * angle_slice
    ).unsqueeze(-1).unsqueeze(-1)
    # We didn't calculate angles in the "zeroth" slice so we disregard them.
    angle_start = angle_slices[1:] - angle_slice / 2
    angle_end = angle_slices[1:] + angle_slice / 2

    # Find all angles lying in each slice.
    angles_in_slice = ((angle_start <= angles) & (angles < angle_end))
    return angles_in_slice, angle_slices


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
    Compute a rotation Matrix to rotate from start to target.

    Parameters
    ----------
    start : torch.Tensor
        Starting direction.
    target : torch.Tensor
        Target direction.

    Returns
    -------
    torch.Tensor
        Return the rotation matrix.
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
        target direction.
    
    Returns
    -------
    torch.Tensor
    """
    angles = torch.acos(torch.clamp(
        (
            batch_dot(a, b).squeeze(-1)
            / (
                torch.linalg.norm(a, dim=-1)
                * torch.linalg.norm(b, dim=-1)
            )
        ),
        -1.0,
        1.0,
    )).squeeze(-1)
    return angles

def initialize_spline_ctrl_points_perfectly(
        control_points: torch.Tensor,
        world_points: torch.Tensor,
        num_points_x: int,
        num_points_y: int,
        degree_x: int,
        degree_y: int,
        knots_x: torch.Tensor,
        knots_y: torch.Tensor,
        change_z_only: bool,
        change_knots: bool,
) -> None:
    """
    Initialize the spline control points.

    Parameters
    ----------
    control_points : torch.Tensor
        The control points of the NURBS surface
    world_points : torch.Tensor
        The world points.
    num_points_x : int
        number of points in x direction.
    num_points_y : int
        number of points in y direction.
    degree_x : int
        The degree of the NURBS surface in x direction.
    degree_y : int 
        The degree of the NURBS surface in y direction.
    knots_x : torch.Tensor
        The knots of the NURBS surface in x direction.
    knots_y : torch.Tensor
        The knots of the NURBS surface in y direction.
    change_z_only : bool
        Which parameters of the control points  to change.
    change_knots : bool
        Change knots or leave them.
    """
    new_control_points, new_knots_x, new_knots_y = nurbs.approximate_surface(
        world_points,
        num_points_x,
        num_points_y,
        degree_x,
        degree_y,
        control_points.shape[0],
        control_points.shape[1],
        knots_x if change_knots else None,
        knots_y if change_knots else None,
    )

    if not change_z_only:
        control_points[:, :, :-1] = new_control_points[:, :, :-1]
    control_points[:, :, -1:] = new_control_points[:, :, -1:]
    if change_knots:
        knots_x[:] = new_knots_x
        knots_y[:] = new_knots_y


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

    Returns
    -------
    Tuple[torch.Tensor, int, int]
        Return the structured points as 

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
    for (i, x) in enumerate(x_vals[1:]):
        if not torch.isclose(x, x_vals[prev_i], atol=tolerance):
            prev_i = i
            keep_indices.append(prev_i)
    x_vals = x_vals[keep_indices]

    y_vals = points[:, 0]
    y_vals = torch.unique(y_vals, sorted=True)

    prev_i = 0
    keep_indices = [prev_i]
    for (i, y) in enumerate(y_vals[1:]):
        if not torch.isclose(y, y_vals[prev_i], atol=tolerance):
            prev_i = i
            keep_indices.append(prev_i)
    y_vals = y_vals[keep_indices]

    structured_points = torch.cartesian_prod(x_vals, y_vals)

    distances = torch.linalg.norm(
        (
            structured_points.unsqueeze(1)
            - points[:, :-1].unsqueeze(0)
        ),
        dim=-1,
    )
    argmin_distances = distances.argmin(1)
    z_vals = points[argmin_distances, -1]
    structured_points = torch.cat(
        [structured_points, z_vals.unsqueeze(-1)], dim=-1)

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
        x_min, x_max, rows, device=x_vals.device)  # type: ignore[arg-type]
    y_vals = torch.linspace(
        y_min, y_max, cols, device=y_vals.device)  # type: ignore[arg-type]

    structured_points = torch.cartesian_prod(x_vals, y_vals)

    distances = torch.linalg.norm(
        (
            structured_points.unsqueeze(1)
            - points[:, :-1].unsqueeze(0)
        ),
        dim=-1,
    )
    argmin_distances = distances.argmin(1)
    z_vals = points[argmin_distances, -1]
    structured_points = torch.cat(
        [structured_points, z_vals.unsqueeze(-1)], dim=-1)

    return structured_points, rows, cols

def initialize_spline_eval_points_perfectly(
        points: torch.Tensor,
        degree_x: int,
        degree_y: int,
        ctrl_points: torch.Tensor,
        ctrl_weights: torch.Tensor,
        knots_x: torch.Tensor,
        knots_y: torch.Tensor,
) -> torch.Tensor:
    eval_points, distances = nurbs.invert_points_slow(
            points,
            degree_x,
            degree_y,
            ctrl_points,
            ctrl_weights,
            knots_x,
            knots_y,
    )
    return eval_points
