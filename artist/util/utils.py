from typing import Union

import torch

from artist.util import config_dictionary


def batch_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Calculate the batch-wise dot product.

    Parameters
    ----------
    x : torch.Tensor
        Single tensor with dimension (1, 4).
    y : torch.Tensor
        Single tensor with dimension (N, 4).


    Returns
    -------
    torch.Tensor
        Dot product of x and y as a tensor with dimension (N, 1).
    """
    return (x * y).sum(-1).unsqueeze(-1)


def rotate_distortions(
    e: torch.Tensor, u: torch.Tensor, device: Union[torch.device, str] = "cuda"
) -> torch.Tensor:
    """
    Rotate the distortions for the sun.

    Rotate around the up and then the east axis in this very order in a right-handed east-north-up
    coordinate system. Positive angles result in a rotation in the mathematical direction of rotation, i.e.,
    counter-clockwise. Points need to be multiplied as column vectors from the right-hand side with the
    resulting rotation matrix. Note that the order is fixed due to the non-commutative property of matrix-matrix
    multiplication.

    Parameters
    ----------
    e : torch.Tensor
        East rotation angle in radians.
    u : torch.Tensor
        Up rotation angle in radians.
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Returns
    -------
    torch.Tensor
        Corresponding rotation matrix.
    """
    assert (
        e.shape == u.shape
    ), "The two tensors containing angles for the east and up rotation must have the same shape."
    device = torch.device(device)

    cos_e = torch.cos(e)
    sin_e = -torch.sin(e)  # Heliostat convention
    cos_u = torch.cos(u)
    sin_u = torch.sin(u)
    zeros = torch.zeros(e.shape, device=device)
    ones = torch.ones(e.shape, device=device)

    return torch.stack(
        [
            torch.stack([cos_u, -sin_u, zeros, zeros], dim=1),
            torch.stack([cos_e * sin_u, cos_e * cos_u, sin_e, zeros], dim=1),
            torch.stack([-sin_e * sin_u, -sin_e * cos_u, cos_e, zeros], dim=1),
            torch.stack([zeros, zeros, zeros, ones], dim=1),
        ],
        dim=1,
    ).permute(0, 3, 4, 1, 2)


def rotate_e(
    e: torch.Tensor, device: Union[torch.device, str] = "cuda"
) -> torch.Tensor:
    """
    Rotate around the east axis.

    Rotate around the east axis in a right-handed east-north-up coordinate system. Positive angles result in a rotation
    in the mathematical direction of rotation, i.e., counter-clockwise. Points need to be multiplied as column vectors
    from the right-hand side with the resulting rotation matrix.


    Parameters
    ----------
    e : torch.Tensor
        East rotation angle in radians.
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Returns
    -------
    torch.Tensor
        Corresponding rotation matrix.
    """
    device = torch.device(device)

    cos_e = torch.cos(e)
    sin_e = -torch.sin(e)  # Heliostat convention
    zeros = torch.zeros(e.shape, device=device)
    ones = torch.ones(e.shape, device=device)
    return torch.stack(
        [
            torch.stack([ones, zeros, zeros, zeros]),
            torch.stack([zeros, cos_e, sin_e, zeros]),
            torch.stack([zeros, -sin_e, cos_e, zeros]),
            torch.stack([zeros, zeros, zeros, ones]),
        ],
    ).squeeze(-1)


def rotate_n(
    n: torch.Tensor, device: Union[torch.device, str] = "cuda"
) -> torch.Tensor:
    """
    Rotate around the north axis.

    Rotate around the north axis in a right-handed east-north-up coordinate system. Positive angles result in a rotation
    in the mathematical direction of rotation, i.e., counter-clockwise. Points need to be multiplied as column vectors
    from the right-hand side with the resulting rotation matrix.

    Parameters
    ----------
    n : torch.Tensor
        North rotation angle in radians.
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Returns
    -------
    torch.Tensor
        Corresponding rotation matrix.
    """
    device = torch.device(device)

    cos_n = torch.cos(n)
    sin_n = torch.sin(n)
    zeros = torch.zeros(n.shape, device=device)
    ones = torch.ones(n.shape, device=device)

    return torch.stack(
        [
            torch.stack([cos_n, zeros, -sin_n, zeros]),
            torch.stack([zeros, ones, zeros, zeros]),
            torch.stack([sin_n, zeros, cos_n, zeros]),
            torch.stack([zeros, zeros, zeros, ones]),
        ],
    ).squeeze(-1)


def rotate_u(
    u: torch.Tensor, device: Union[torch.device, str] = "cuda"
) -> torch.Tensor:
    """
    Rotate around the up axis.

    Rotate around the up axis in a right-handed east-north-up coordinate system. Positive angles result in a rotation in
    the mathematical direction of rotation, i.e., counter-clockwise. Points need to be multiplied as column vectors from
    the right-hand side with the resulting rotation matrix.

    Parameters
    ----------
    u : torch.Tensor
        Up rotation angle in radians.
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Returns
    -------
    torch.Tensor
        Corresponding rotation matrix.
    """
    device = torch.device(device)

    cos_u = torch.cos(u)
    sin_u = torch.sin(u)
    zeros = torch.zeros(u.shape, device=device)
    ones = torch.ones(u.shape, device=device)

    return torch.stack(
        [
            torch.stack([cos_u, -sin_u, zeros, zeros]),
            torch.stack([sin_u, cos_u, zeros, zeros]),
            torch.stack([zeros, zeros, ones, zeros]),
            torch.stack([zeros, zeros, zeros, ones]),
        ],
    ).squeeze(-1)


def translate_enu(
    e: torch.Tensor,
    n: torch.Tensor,
    u: torch.Tensor,
    device: Union[torch.device, str] = "cuda",
) -> torch.Tensor:
    """
    Translate in all directions.

    Translate a given point in the east, north, and up direction. Note that the point must be multiplied as a column
    vector from the right-hand side of the resulting matrix.

    Parameters
    ----------
    e : torch.Tensor
        East translation.
    n : torch.Tensor
        North translation.
    u : torch.Tensor
        Up translation.
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Returns
    -------
    torch.Tensor
        Corresponding rotation matrix.
    """
    assert (
        e.shape == u.shape == n.shape
    ), "The three tensors containing the east, north, and up translations must have the same shape."
    device = torch.device(device)

    zeros = torch.zeros(e.shape, device=device)
    ones = torch.ones(e.shape, device=device)

    return torch.stack(
        [
            torch.stack([ones, zeros, zeros, e]),
            torch.stack([zeros, ones, zeros, n]),
            torch.stack([zeros, zeros, ones, u]),
            torch.stack([zeros, zeros, zeros, ones]),
        ],
    ).squeeze(-1)


def azimuth_elevation_to_enu(
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    srange: float = 1.0,
    degree: bool = True,
) -> torch.Tensor:
    """
    Coordinate transformation from azimuth and elevation to east, north, up.

    Parameters
    ----------
    azimuth : torch.Tensor
        Azimuth, clockwise from north (degrees).
    elevation : torch.Tensor
        Elevation angle above horizon, neglecting aberrations (degrees).
    srange : float
        Slant range (meters).
    degree : bool
        Specifies if input is given in degrees or radians.

    Returns
    -------
    torch.Tensor
        The east, north, up (enu) coordinates.
    """
    if degree:
        elevation = torch.deg2rad(elevation)
        azimuth = torch.deg2rad(azimuth)

    r = srange * torch.cos(elevation)

    enu = torch.stack(
        [
            r * torch.sin(azimuth),
            -r * torch.cos(azimuth),
            srange * torch.sin(elevation),
        ],
        dim=0,
    )
    return enu


def convert_3d_points_to_4d_format(
    point: torch.Tensor, device: Union[torch.device, str] = "cuda"
) -> torch.Tensor:
    """
    Append ones to the last dimension of a 3D point vector.

    Includes the convention that points have a 1 and directions have 0 as 4th dimension.

    Parameters
    ----------
    point : torch.Tensor
        Input point in a 3D format.
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Returns
    -------
    torch.Tensor
        Point vector with ones appended at the last dimension.
    """
    device = torch.device(device)
    assert (
        point.size(dim=-1) == 3
    ), f"Expected a 3D point but got a point of shape {point.shape}!"
    ones_tensor = torch.ones(point.shape[:-1] + (1,), dtype=point.dtype, device=device)
    return torch.cat((point, ones_tensor), dim=-1)


def convert_3d_direction_to_4d_format(
    direction: torch.Tensor, device: Union[torch.device, str] = "cuda"
) -> torch.Tensor:
    """
    Append zeros to the last dimension of a 3D direction vector.

    Includes the convention that points have a 1 and directions have 0 as 4th dimension.

    Parameters
    ----------
    direction : torch.Tensor
        Input direction in a 3D format.
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Returns
    -------
    torch.Tensor
        Direction vector with ones appended at the last dimension.
    """
    device = torch.device(device)
    assert (
        direction.size(dim=-1) == 3
    ), f"Expected a 3D direction vector but got a director vector of shape {direction.shape}!"
    zeros_tensor = torch.zeros(
        direction.shape[:-1] + (1,), dtype=direction.dtype, device=device
    )
    return torch.cat((direction, zeros_tensor), dim=-1)


def convert_WGS84_coordinates_to_local_enu(
    coordinates_to_transform: torch.Tensor,
    reference_point: torch.Tensor,
    device: Union[torch.device, str] = "cuda",
) -> torch.Tensor:
    """
    Transform coordinates from latitude, longitude and altitude (WGS84) to local east, north, up (ENU).

    This function calculates the north and east offsets in meters of a coordinate from the reference point.
    It converts the latitude and longitude to radians, calculates the radius of curvature values,
    and then computes the offsets based on the differences between the coordinate and the refernce point.
    Finally, it returns a tensor containing these offsets along with the altitude difference.

    Parameters
    ----------
    coordinates_to_transform : torch.Tensor
        The coordinates in lat, lon, alt that are to be transformed.
    reference_point : torch.Tensor
        The center of origin of the ENU coordinate system in WGS84 coordinates.
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Returns
    -------
    torch.Tensor
        The east offset in meters, north offset in meters, and the altitude difference from the reference point.
    """
    device = torch.device(device)
    # Convert latitude and longitude to radians
    lat_rad = torch.deg2rad(coordinates_to_transform[0])
    lon_rad = torch.deg2rad(coordinates_to_transform[1])
    alt = coordinates_to_transform[2] - reference_point[2]
    lat_tower_rad = torch.deg2rad(reference_point[0])
    lon_tower_rad = torch.deg2rad(reference_point[1])

    # Calculate meridional radius of curvature for the first latitude
    sin_lat1 = torch.sin(lat_rad)
    rn1 = config_dictionary.WGS84_A / torch.sqrt(
        1 - config_dictionary.WGS84_E2 * sin_lat1**2
    )

    # Calculate transverse radius of curvature for the first latitude
    rm1 = (config_dictionary.WGS84_A * (1 - config_dictionary.WGS84_E2)) / (
        (1 - config_dictionary.WGS84_E2 * sin_lat1**2) ** 1.5
    )

    # Calculate delta latitude and delta longitude in radians
    dlat_rad = lat_tower_rad - lat_rad
    dlon_rad = lon_tower_rad - lon_rad

    # Calculate north and east offsets in meters
    north_offset_m = dlat_rad * rm1
    east_offset_m = dlon_rad * rn1 * torch.cos(lat_rad)

    return torch.tensor([-east_offset_m, -north_offset_m, alt], device=device)


def get_center_of_mass(
    bitmap: torch.Tensor,
    target_center: torch.Tensor,
    plane_e: torch.Tensor,
    plane_u: torch.Tensor,
    device: Union[torch.device, str] = "cuda",
) -> torch.Tensor:
    """
    Calculate the coordinates of the flux density center of mass.

    First determine the indices of the bitmap center of mass.
    Next determine the position (coordinates) of the center of mass on the target.

    Parameters
    ----------
    bitmap : torch.Tensor
        The flux density in form of a bitmap.
    target_center : torch.Tensor
        The position of the center of the target.
    plane_e : torch.Tensor
        The width of the target surface.
    plane_u : torch.Tensor
        The height of the target surface.
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    torch.Tensor
        The coordinates of the flux density center of mass.
    """
    device = torch.device(device)
    # Calculate center of mass of the bitmap
    x_indices = torch.arange(bitmap.shape[0], device=device)
    y_indices = torch.arange(bitmap.shape[1], device=device)
    x_indices, y_indices = torch.meshgrid(x_indices, y_indices, indexing='ij')
    
    total_mass = bitmap.sum()
    normalized_bitmap = bitmap / total_mass

    center_of_mass_x = (normalized_bitmap * x_indices).sum()
    center_of_mass_y = (normalized_bitmap * y_indices).sum()
    center_of_mass_bitmap = torch.stack([center_of_mass_x, center_of_mass_y], dim=-1)

    # Construct the coordinates relative to target center
    de = torch.tensor([plane_e, 0.0, 0.0, 0.0], device=device)
    du = torch.tensor([0.0, 0.0, plane_u, 0.0], device=device)
    normalized_center = center_of_mass_bitmap / bitmap.size(-1)
    e = normalized_center[1]
    u = 1 - normalized_center[0]
    center_coordinates = target_center - 0.5 * (de + du) + e * de + u * du

    return center_coordinates