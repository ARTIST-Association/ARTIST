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


def calculate_position_in_m_from_lat_lon(
    coordinates_to_transform: torch.Tensor,
    power_plant_coordinates: torch.Tensor,
    device: Union[torch.device, str] = "cuda",
) -> torch.Tensor:
    """
    Transform coordinates from latitude, longitude and altitude to meters.

    This function calculates the north and east offsets in meters of a coordinate from the power plant location.
    It converts the latitude and longitude to radians, calculates the radius of curvature values,
    and then computes the offsets based on the differences between the coordinate and power plant center of origin.
    Finally, it returns a tensor containing these offsets along with the altitude difference.

    Parameters
    ----------
    coordinates_to_transform : torch.Tensor
        The coordinates in lat, lon, alt that are to be transformed.
    power_plant_coordinates : torch.Tensor
        The center of origin of the power plant as a reference point.
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Returns
    -------
    torch.Tensor
        The east offset in meters, north offset in meters, and the altitude difference from the power plant.
    """
    device = torch.device(device)
    # Convert latitude and longitude to radians
    lat_rad = torch.deg2rad(coordinates_to_transform[0])
    lon_rad = torch.deg2rad(coordinates_to_transform[1])
    alt = coordinates_to_transform[2] - power_plant_coordinates[2]
    lat_tower_rad = torch.deg2rad(power_plant_coordinates[0])
    lon_tower_rad = torch.deg2rad(power_plant_coordinates[1])

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


# Function to convert LLA to ECEF
def lla_to_ecef(lat, lon, alt):
    a = 6378137.0  # WGS-84 Earth semimajor axis (meters)
    f = 1 / 298.257223563  # WGS-84 flattening factor
    b = a * (1 - f)  # Semi-minor axis

    # Convert to radians
    lat = torch.deg2rad(lat)
    lon = torch.deg2rad(lon)

    # Precompute trigonometric values
    cos_lat = torch.cos(lat)
    sin_lat = torch.sin(lat)
    cos_lon = torch.cos(lon)
    sin_lon = torch.sin(lon)

    # Calculate N (radius of curvature in the prime vertical)
    N = a / torch.sqrt(1 - f * (2 - f) * sin_lat**2)

    # Calculate ECEF coordinates
    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1 - f) ** 2 + alt) * sin_lat
    return torch.stack((x, y, z), dim=-1)


# Function to calculate the ENU transformation matrix
def enu_matrix(lat_ref, lon_ref):
    lat_ref = torch.deg2rad(lat_ref)
    lon_ref = torch.deg2rad(lon_ref)

    # Precompute trigonometric values for rotation matrix
    cos_lat_ref = torch.cos(lat_ref)
    sin_lat_ref = torch.sin(lat_ref)
    cos_lon_ref = torch.cos(lon_ref)
    sin_lon_ref = torch.sin(lon_ref)

    # Rotation matrix from ECEF to ENU
    rot_matrix = torch.tensor(
        [
            [-sin_lon_ref, cos_lon_ref, 0],
            [-sin_lat_ref * cos_lon_ref, -sin_lat_ref * sin_lon_ref, cos_lat_ref],
            [cos_lat_ref * cos_lon_ref, cos_lat_ref * sin_lon_ref, sin_lat_ref],
        ]
    )
    return rot_matrix


# Function to convert LLA to ENU
def lla_to_enu(lat, lon, alt, lat_ref, lon_ref, alt_ref):
    # Convert the reference point and target point to ECEF
    ref_ecef = lla_to_ecef(lat_ref, lon_ref, alt_ref)
    target_ecef = lla_to_ecef(lat, lon, alt)

    # Calculate the ENU rotation matrix using the reference point
    rot_matrix = enu_matrix(lat_ref, lon_ref)

    # Calculate the difference vector in ECEF
    diff = target_ecef - ref_ecef

    # Reshape diff to (3, 1) for compatibility with torch.matmul, and apply the ENU rotation matrix
    enu = torch.matmul(rot_matrix, diff.unsqueeze(1)).squeeze(1)
    return enu
