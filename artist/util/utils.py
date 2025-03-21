from typing import Generator, Union

import torch


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
        East rotation angles in radians.
    u : torch.Tensor
        Up rotation angles in radians.
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Raises
    ------
    ValueError
        If the sizes of the input tensors do not match.

    Returns
    -------
    torch.Tensor
        Corresponding rotation matrix.
    """
    if e.shape != u.shape:
        raise ValueError(
            "The two tensors containing angles for the east and up rotation must have the same shape."
        )
    device = torch.device(device)

    cos_e = torch.cos(e)
    sin_e = -torch.sin(e)  # Heliostat convention.
    cos_u = torch.cos(u)
    sin_u = torch.sin(u)
    ones = torch.ones(e.shape, device=device)

    matrix = torch.zeros(e.shape[0], e.shape[1], e.shape[2], 4, 4, device=device)

    matrix[:, :, :, 0, 0] = cos_u
    matrix[:, :, :, 0, 1] = -sin_u
    matrix[:, :, :, 1, 0] = cos_e * sin_u
    matrix[:, :, :, 1, 1] = cos_e * cos_u
    matrix[:, :, :, 1, 2] = sin_e
    matrix[:, :, :, 2, 0] = -sin_e * sin_u
    matrix[:, :, :, 2, 1] = -sin_e * cos_u
    matrix[:, :, :, 2, 2] = cos_e
    matrix[:, :, :, 3, 3] = ones

    return matrix


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
        East rotation angles in radians.
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Returns
    -------
    torch.Tensor
        Corresponding rotation matrix.
    """
    device = torch.device(device)

    cos_e = torch.cos(e)
    sin_e = -torch.sin(e)  # Heliostat convention.
    ones = torch.ones(e.shape, device=device)

    matrix = torch.zeros(e.shape[0], 4, 4, device=device)

    matrix[:, 0, 0] = ones
    matrix[:, 1, 1] = cos_e
    matrix[:, 1, 2] = sin_e
    matrix[:, 2, 1] = -sin_e
    matrix[:, 2, 2] = cos_e
    matrix[:, 3, 3] = ones

    return matrix


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
        North rotation angles in radians.
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
    ones = torch.ones(n.shape, device=device)

    matrix = torch.zeros(n.shape[0], 4, 4, device=device)

    matrix[:, 0, 0] = cos_n
    matrix[:, 0, 2] = -sin_n
    matrix[:, 1, 1] = ones
    matrix[:, 2, 0] = sin_n
    matrix[:, 2, 2] = cos_n
    matrix[:, 3, 3] = ones

    return matrix


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
        Up rotation angles in radians.
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
    ones = torch.ones(u.shape, device=device)

    matrix = torch.zeros(u.shape[0], 4, 4, device=device)

    matrix[:, 0, 0] = cos_u
    matrix[:, 0, 1] = -sin_u
    matrix[:, 1, 0] = sin_u
    matrix[:, 1, 1] = cos_u
    matrix[:, 2, 2] = ones
    matrix[:, 3, 3] = ones

    return matrix


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
        East translations.
    n : torch.Tensor
        North translations.
    u : torch.Tensor
        Up translations.
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Raises
    ------
    ValueError
        If the sizes of the input tensors do not match.

    Returns
    -------
    torch.Tensor
        Corresponding rotation matrix.
    """
    if not (e.shape == u.shape == n.shape):
        raise ValueError(
            "The three tensors containing the east, north, and up translations must have the same shape."
        )

    device = torch.device(device)

    ones = torch.ones(e.shape, device=device)

    matrix = torch.zeros(e.shape[0], 4, 4, device=device)

    matrix[:, 0, 0] = ones
    matrix[:, 0, 3] = e
    matrix[:, 1, 1] = ones
    matrix[:, 1, 3] = n
    matrix[:, 2, 2] = ones
    matrix[:, 2, 3] = u
    matrix[:, 3, 3] = ones

    return matrix


def azimuth_elevation_to_enu(
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    slant_range: float = 1.0,
    degree: bool = True,
    device: Union[torch.device, str] = "cuda",
) -> torch.Tensor:
    """
    Transform coordinates from azimuth and elevation to east, north, up.

    This method assumes a south-oriented azimuth-elevation coordiante system, where 0° points toward the south.

    Parameters
    ----------
    azimuth : torch.Tensor
        Azimuth, 0° points toward the south (degrees).
    elevation : torch.Tensor
        Elevation angle above horizon, neglecting aberrations (degrees).
    slant_range : float
        Slant range in meters (default is 1.0).
    degree : bool
        Whether input is given in degrees (default is True).
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Returns
    -------
    torch.Tensor
        The east, north, up (ENU) coordinates.
    """
    if degree:
        elevation = torch.deg2rad(elevation)
        azimuth = torch.deg2rad(azimuth)

    if azimuth < 0.0:
        azimuth += torch.pi * 2

    r = slant_range * torch.cos(elevation)

    enu = torch.zeros(3, device=device)

    enu[0] = r * torch.sin(azimuth)
    enu[1] = -r * torch.cos(azimuth)
    enu[2] = slant_range * torch.sin(elevation)

    return enu


def convert_3d_point_to_4d_format(
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

    Raises
    ------
    ValueError
        If the input is not 3D.

    Returns
    -------
    torch.Tensor
        Point vector with ones appended at the last dimension.
    """
    device = torch.device(device)
    if point.size(dim=-1) != 3:
        raise ValueError(f"Expected a 3D point but got a point of shape {point.shape}!")

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

    Raises
    ------
    ValueError
        If the input is not 3D.

    Returns
    -------
    torch.Tensor
        Direction vector with ones appended at the last dimension.
    """
    device = torch.device(device)
    if direction.size(dim=-1) != 3:
        raise ValueError(
            f"Expected a 3D direction but got a direction of shape {direction.shape}!"
        )

    zeros_tensor = torch.zeros(
        direction.shape[:-1] + (1,), dtype=direction.dtype, device=device
    )
    return torch.cat((direction, zeros_tensor), dim=-1)


def convert_wgs84_coordinates_to_local_enu(
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
        The coordinates in latitude, longitude, altitude that are to be transformed.
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
    wgs84_a = 6378137.0  # Major axis in meters
    wgs84_b = 6356752.314245  # Minor axis in meters
    wgs84_e2 = (wgs84_a**2 - wgs84_b**2) / wgs84_a**2  # Eccentricity squared

    # Convert latitude and longitude to radians.
    lat_rad = torch.deg2rad(coordinates_to_transform[0])
    lon_rad = torch.deg2rad(coordinates_to_transform[1])
    alt = coordinates_to_transform[2] - reference_point[2]
    lat_tower_rad = torch.deg2rad(reference_point[0])
    lon_tower_rad = torch.deg2rad(reference_point[1])

    # Calculate meridional radius of curvature for the first latitude.
    sin_lat1 = torch.sin(lat_rad)
    rn1 = wgs84_a / torch.sqrt(1 - wgs84_e2 * sin_lat1**2)

    # Calculate transverse radius of curvature for the first latitude.
    rm1 = (wgs84_a * (1 - wgs84_e2)) / ((1 - wgs84_e2 * sin_lat1**2) ** 1.5)

    # Calculate delta latitude and delta longitude in radians.
    dlat_rad = lat_tower_rad - lat_rad
    dlon_rad = lon_tower_rad - lon_rad

    # Calculate north and east offsets in meters.
    north_offset_m = dlat_rad * rm1
    east_offset_m = dlon_rad * rn1 * torch.cos(lat_rad)

    return torch.tensor(
        [-east_offset_m, -north_offset_m, alt], dtype=torch.float32, device=device
    )


def normalize_points(points: torch.Tensor) -> torch.Tensor:
    """
    Normalize points in a tensor to the open interval of (0,1).

    Parameters
    ----------
    points : torch.Tensor
        A tensor containing points to be normalized.

    Returns
    -------
    torch.Tensor
        The normalized points.
    """
    # Since we want the open interval (0,1), a small offset is required to also exclude the boundaries.
    points_normalized = (points[:] - min(points[:]) + 1e-5) / max(
        (points[:] - min(points[:])) + 2e-5
    )
    return points_normalized


def corner_points_to_plane(
    upper_left: torch.Tensor,
    upper_right: torch.Tensor,
    lower_left: torch.Tensor,
    lower_right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Span a plane from corner points.

    Parameters
    ----------
    upper_left : torch.Tensor
        The upper left corner coordinate.
    upper_right : torch.Tensor
        The upper right corner coordinate.
    lower_left : torch.Tensor
        The lower left corner coordinate.
    lower_right : torch.Tensor
        The lower right corner coordinate.

    Returns
    -------
    torch.Tensor
        The plane measurement in east direction.
    torch.Tensor
        The plane measurement in up direction.
    """
    plane_e = (
        torch.abs(upper_right[0] - upper_left[0])
        + torch.abs(lower_right[0] - lower_left[0])
    ) / 2
    plane_u = (
        torch.abs(upper_left[2] - lower_left[2])
        + torch.abs(upper_right[2] - lower_right[2])
    ) / 2
    return plane_e, plane_u


def decompose_rotations(
    initial_vector: torch.Tensor,
    target_vector: torch.Tensor,
    device: Union[torch.device, str] = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get the individual angles along the east-, north- and up-axis, to rotate and initial vector into a target vector.

    Parameters
    ----------
    initial_vector : torch.Tensor
        The initial vector.
    rotated_vector : torch.Tensor
        The rotated vector.
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Returns
    -------
    torch.Tensor
        The angle for the east-axis rotation.
    torch.Tensor
        The angle for the north-axis rotation.
    torch.Tensor
        The angle for the up-axis rotation.

    """
    device = torch.device(device)

    # Normalize the input vectors.
    initial_vector = torch.nn.functional.normalize(initial_vector, p=2, dim=1)
    target_vector = torch.nn.functional.normalize(target_vector, p=2, dim=0).unsqueeze(
        0
    )

    # Compute the cross product (rotation axis).
    r = torch.linalg.cross(initial_vector, target_vector)

    # Normalize the rotation axis.
    r_normalized = torch.nn.functional.normalize(r, p=2, dim=1)

    # Compute the angle between the vectors.
    theta = torch.arccos(torch.clamp(initial_vector @ target_vector.T, -1.0, 1.0))

    # Decompose the angle along each axis.
    theta_components = theta * r_normalized

    return theta_components[:, 0], theta_components[:, 1], theta_components[:, 2]


def angle_between_vectors(
    vector_1: torch.Tensor, vector_2: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the angle between two vectors.

    Parameters
    ----------
    vector_1 : torch.Tensor
        The first vector.
    vector_2 : torch.Tensor
        The second vector.

    Return
    ------
    torch.Tensor
        The angle between the input vectors.
    """
    dot_product = torch.dot(vector_1, vector_2)

    norm_u = torch.norm(vector_1)
    norm_v = torch.norm(vector_2)

    angle = dot_product / (norm_u * norm_v)

    angle = torch.clamp(angle, -1.0, 1.0)

    angle = torch.acos(angle)

    return angle


def transform_initial_angle(
    initial_angle: torch.Tensor,
    initial_orientation: torch.Tensor,
    device: Union[torch.device, str] = "cuda",
) -> torch.Tensor:
    """
    Compute the transformed angle of an initial angle in a rotated coordinate system.

    This function accounts for a known offset, the initial angle, in the
    initial orientation vector. The offset represents a rotation around the
    east-axis. When the coordinate system is rotated to align
    the initial orientation with the ``ARTIST`` standard orientation, the axis for
    the offset rotation also changes. This function calculates the equivalent
    transformed angle for the offset in the rotated coordinate system.

    Parameters
    ----------
    initial_angle : torch.Tensor
        The initial angle, or offset along the east-axis.
    initial_orientation : torch.Tensor
        The initial orientation of the coordiante system.
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Returns
    -------
    torch.Tensor
        The transformed angle in the rotated coordinate system.
    """
    device = torch.device(device)
    # ``ARTIST`` is oriented towards the south ([0.0, -1.0, 0.0]) ENU.
    artist_standard_orientation = torch.tensor([0.0, -1.0, 0.0, 0.0], device=device)

    # Apply the rotation by the initial angle to the initial orientation.
    initial_orientation_with_offset = initial_orientation @ rotate_e(
        e=initial_angle,
        device=device,
    ).squeeze(0)

    # Compute the transformed angle relative to the reference orientation.
    transformed_initial_angle = angle_between_vectors(
        initial_orientation[:-1], initial_orientation_with_offset[:-1]
    ) - angle_between_vectors(
        initial_orientation[:-1], artist_standard_orientation[:-1]
    )

    return transformed_initial_angle


def get_center_of_mass(
    bitmap: torch.Tensor,
    target_center: torch.Tensor,
    plane_e: float,
    plane_u: float,
    threshold: float = 0.0,
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
    plane_e : float
        The width of the target surface.
    plane_u : float
        The height of the target surface.
    threshold : float
        Determines how intense a pixel in the bitmap needs to be to be registered (default is 0.0).
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Returns
    -------
    torch.Tensor
        The coordinates of the flux density center of mass.
    """
    device = torch.device(device)
    height, width = bitmap.shape

    # Threshold the bitmap values. Any values below the threshold are set to zero.
    flux_thresholded = torch.where(
        bitmap >= threshold, bitmap, torch.zeros_like(bitmap, device=device)
    )
    total_intensity = flux_thresholded.sum()

    # Generate normalized east and up coordinates adjusted for pixel centers.
    # The "+ 0.5" adjustment ensures coordinates are centered within each pixel.
    e_indices = (torch.arange(width, dtype=torch.float32, device=device) + 0.5) / width
    u_indices = (
        torch.arange(height, dtype=torch.float32, device=device) + 0.5
    ) / height

    # Compute the center of intensity using weighted sums of the coordinates.
    center_of_mass_e = (flux_thresholded.sum(dim=0) * e_indices).sum() / total_intensity
    center_of_mass_u = 1 - (
        (flux_thresholded.sum(dim=1) * u_indices).sum() / total_intensity
    )

    # Construct the coordinates relative to target center.
    de = torch.tensor([-plane_e, 0.0, 0.0, 0.0], device=device)
    du = torch.tensor([0.0, 0.0, plane_u, 0.0], device=device)

    center_coordinates = (
        target_center - 0.5 * (de + du) + center_of_mass_e * de + center_of_mass_u * du
    )

    return center_coordinates


def setup_distributed_environment(
    device: Union[torch.device, str] = "cuda",
) -> Generator[tuple[torch.device, bool, int, int], None, None]:
    """
    Set up the distributed environment and destroy it in the end.

    Based on the available devices, the process group is initialized with the
    appropriate backend. For computation on GPUs the nccl backend optimized for
    NVIDIA GPUs is chosen. For computation on CPUs gloo is used as backend. If
    the program is run without the intention of being distributed, the world_size
    will be set to 1, accordingly the only rank is 0.

    Parameters
    ----------
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Yields
    ------
    torch.device
        The device for each rank.
    bool
        Distributed mode enabled or disabled.
    int
        The rank of the current process.
    int
        The world size or total number of processes.
    """
    device = torch.device(device)
    backend = "nccl" if device.type == "cuda" else "gloo"

    is_distributed = False
    rank, world_size = 0, 1

    try:
        # Attempt to initialize the process group.
        torch.distributed.init_process_group(backend=backend, init_method="env://")
        is_distributed = True
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        if rank == 0:
            print(f"Using device type: {device.type} and backend: {backend}.")
            print(f"Distributed Mode: {'Enabled.' if is_distributed else 'Disabled.'}")
        print(
            f"Distributed process group initialized: Rank {rank}, World Size {world_size}"
        )

    except Exception:
        print(f"Using device type: {device.type} and backend: {backend}.")
        print("Running in single-device mode.")

    if device.type == "cuda" and is_distributed:
        gpu_count = torch.cuda.device_count()
        device_id = rank % gpu_count
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)

    try:
        yield device, is_distributed, rank, world_size
    finally:
        if is_distributed:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
