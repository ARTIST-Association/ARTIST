import json
import pathlib
from typing import TYPE_CHECKING, Union

import torch

if TYPE_CHECKING:
    from artist.field.kinematic_rigid_body import RigidBody


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
    if e.shape != u.shape:
        raise ValueError(
            "The two tensors containing angles for the east and up rotation must have the same shape."
        )
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
    if e.shape != u.shape != n.shape:
        raise ValueError(
            "The three tensors containing the east, north, and up translations must have the same shape."
        )

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
    slant_range: float = 1.0,
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
    slant_range : float
        Slant range (meters).
    degree : bool
        Whether input is given in degrees or radians.

    Returns
    -------
    torch.Tensor
        The east, north, up (enu) coordinates.
    """
    if degree:
        elevation = torch.deg2rad(elevation)
        azimuth = torch.deg2rad(azimuth)

    r = slant_range * torch.cos(elevation)

    enu = torch.stack(
        [
            r * torch.sin(azimuth),
            -r * torch.cos(azimuth),
            slant_range * torch.sin(elevation),
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

    Returns
    -------
    torch.Tensor
        Direction vector with ones appended at the last dimension.
    """
    device = torch.device(device)
    if direction.size(dim=-1) != 3:
        raise ValueError(
            f"Expected a 3D point but got a point of shape {direction.shape}!"
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


def get_center_of_mass(
    bitmap: torch.Tensor,
    target_center: torch.Tensor,
    plane_e: float,
    plane_u: float,
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
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Returns
    -------
    torch.Tensor
        The coordinates of the flux density center of mass.
    """
    device = torch.device(device)
    # Calculate center of mass of the bitmap.
    e_indices = torch.arange(bitmap.shape[0], device=device)
    u_indices = torch.arange(bitmap.shape[1], device=device)
    e_indices, u_indices = torch.meshgrid(e_indices, u_indices, indexing="ij")

    total_mass = bitmap.sum()
    normalized_bitmap = bitmap / total_mass

    center_of_mass_e = (normalized_bitmap * e_indices).sum()
    center_of_mass_u = (normalized_bitmap * u_indices).sum()
    center_of_mass_bitmap = torch.stack([center_of_mass_e, center_of_mass_u], dim=-1)

    # Construct the coordinates relative to target center.
    de = torch.tensor([plane_e, 0.0, 0.0, 0.0], device=device)
    du = torch.tensor([0.0, 0.0, plane_u, 0.0], device=device)
    normalized_center = center_of_mass_bitmap / bitmap.size(-1)
    e = normalized_center[1]
    u = 1 - normalized_center[0]
    center_coordinates = target_center - 0.5 * (de + du) + e * de + u * du

    return center_coordinates


def get_rigid_body_kinematic_parameters_from_scenario(
    kinematic: "RigidBody",
) -> list[torch.Tensor]:
    """
    Extract all deviation parameters and actuator parameters from a rigid body kinematic.

    Parameters
    ----------
    kinematic : RigidBody
        The kinematic from which to extract the parameters.

    Returns
    -------
    list[torch.Tensor]
        The parameters from the kinematic (requires_grad is True).
    """
    parameters_list = [
        kinematic.deviation_parameters.first_joint_translation_e,
        kinematic.deviation_parameters.first_joint_translation_n,
        kinematic.deviation_parameters.first_joint_translation_u,
        kinematic.deviation_parameters.first_joint_tilt_e,
        kinematic.deviation_parameters.first_joint_tilt_n,
        kinematic.deviation_parameters.first_joint_tilt_u,
        kinematic.deviation_parameters.second_joint_translation_e,
        kinematic.deviation_parameters.second_joint_translation_n,
        kinematic.deviation_parameters.second_joint_translation_u,
        kinematic.deviation_parameters.second_joint_tilt_e,
        kinematic.deviation_parameters.second_joint_tilt_n,
        kinematic.deviation_parameters.second_joint_tilt_u,
        kinematic.deviation_parameters.concentrator_translation_e,
        kinematic.deviation_parameters.concentrator_translation_n,
        kinematic.deviation_parameters.concentrator_translation_u,
        kinematic.deviation_parameters.concentrator_tilt_e,
        kinematic.deviation_parameters.concentrator_tilt_n,
        kinematic.deviation_parameters.concentrator_tilt_u,
        kinematic.actuators.actuator_list[0].increment,
        kinematic.actuators.actuator_list[0].initial_stroke_length,
        kinematic.actuators.actuator_list[0].offset,
        kinematic.actuators.actuator_list[0].pivot_radius,
        kinematic.actuators.actuator_list[0].initial_angle,
        kinematic.actuators.actuator_list[1].increment,
        kinematic.actuators.actuator_list[1].initial_stroke_length,
        kinematic.actuators.actuator_list[1].offset,
        kinematic.actuators.actuator_list[1].pivot_radius,
        kinematic.actuators.actuator_list[1].initial_angle,
    ]
    for parameter in parameters_list:
        if parameter is not None:
            parameter.requires_grad_()

    return parameters_list


def get_calibration_properties(
    calibration_properties_path: pathlib.Path,
    power_plant_position: torch.Tensor,
    device: Union[torch.device, str] = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get the calibration properties.

    Parameters
    ----------
    calibration_properties_path : pathlib.Path
        The path to the calibration properties file.
    power_plant_position : torch.Tensor
        The position of the power plant in latitude, longitude and elevation.
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Returns
    -------
    torch.Tensor
        The calibration flux density center.
    torch.Tensor
        The incident ray direction.
    torch.Tensor
        The motor positions.
    """
    device = torch.device(device)
    with open(calibration_properties_path, "r") as file:
        calibration_dict = json.load(file)
        center_calibration_image = convert_wgs84_coordinates_to_local_enu(
            torch.tensor(
                calibration_dict["focal_spot"]["UTIS"],
                dtype=torch.float64,
                device=device,
            ),
            power_plant_position,
            device=device,
        )
        center_calibration_image = convert_3d_points_to_4d_format(
            center_calibration_image, device=device
        )
        sun_azimuth = torch.tensor(calibration_dict["Sun_azimuth"], device=device)
        sun_elevation = torch.tensor(calibration_dict["Sun_elevation"], device=device)
        incident_ray_direction = convert_3d_direction_to_4d_format(
            azimuth_elevation_to_enu(sun_azimuth, sun_elevation, degree=True),
            device=device,
        )
        motor_positions = torch.tensor(
            [
                calibration_dict["motor_position"]["Axis1MotorPosition"],
                calibration_dict["motor_position"]["Axis2MotorPosition"],
            ],
            device=device,
        )

    return center_calibration_image, incident_ray_direction, motor_positions


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


def decompose_rotation(
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
    # Normalize the input vectors
    initial_vector = initial_vector / torch.linalg.norm(initial_vector)
    target_vector = target_vector / torch.linalg.norm(target_vector)

    # Compute the cross product (rotation axis)
    r = torch.linalg.cross(initial_vector, target_vector)
    r_norm = torch.linalg.norm(r)

    # If the cross product is zero, the vectors are aligned; no rotation needed
    if r_norm == 0:
        return torch.tensor([0.0, 0.0, 0.0], device=device)

    # Normalize the rotation axis
    r_normalized = r / r_norm

    # Compute the angle between the vectors
    cos_theta = torch.clip(torch.dot(initial_vector, target_vector), -1.0, 1.0)
    theta = torch.arccos(cos_theta)

    # Decompose the angle along each axis
    theta_components = theta * r_normalized

    return theta_components[0], theta_components[1], theta_components[2]


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
    )

    # Compute the transformed angle relative to the reference orientation
    transformed_initial_angle = angle_between_vectors(
        initial_orientation[:-1], initial_orientation_with_offset[:-1]
    ) - angle_between_vectors(
        initial_orientation[:-1], artist_standard_orientation[:-1]
    )

    return transformed_initial_angle
