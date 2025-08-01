import torch
import torch.nn.functional as F

from artist.util.environment_setup import get_device


def rotate_distortions(
    e: torch.Tensor,
    u: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Rotate the distortions for the light source.

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
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Raises
    ------
    ValueError
        If the sizes of the input tensors do not match.

    Returns
    -------
    torch.Tensor
        Corresponding rotation matrix.
    """
    device = get_device(device=device)

    if e.shape != u.shape:
        raise ValueError(
            "The two tensors containing angles for the east and up rotation must have the same shape."
        )

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
    e: torch.Tensor,
    device: torch.device | None = None,
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
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        Corresponding rotation matrix.
    """
    device = get_device(device=device)

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


def rotate_n(n: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    """
    Rotate around the north axis.

    Rotate around the north axis in a right-handed east-north-up coordinate system. Positive angles result in a rotation
    in the mathematical direction of rotation, i.e., counter-clockwise. Points need to be multiplied as column vectors
    from the right-hand side with the resulting rotation matrix.

    Parameters
    ----------
    n : torch.Tensor
        North rotation angles in radians.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        Corresponding rotation matrix.
    """
    device = get_device(device=device)

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


def rotate_u(u: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    """
    Rotate around the up axis.

    Rotate around the up axis in a right-handed east-north-up coordinate system. Positive angles result in a rotation in
    the mathematical direction of rotation, i.e., counter-clockwise. Points need to be multiplied as column vectors from
    the right-hand side with the resulting rotation matrix.

    Parameters
    ----------
    u : torch.Tensor
        Up rotation angles in radians.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        Corresponding rotation matrix.
    """
    device = get_device(device=device)

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
    device: torch.device | None = None,
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
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Raises
    ------
    ValueError
        If the sizes of the input tensors do not match.

    Returns
    -------
    torch.Tensor
        Corresponding rotation matrix.
    """
    device = get_device(device=device)

    if not (e.shape == u.shape == n.shape):
        raise ValueError(
            "The three tensors containing the east, north, and up translations must have the same shape."
        )

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


def convert_3d_points_to_4d_format(
    points: torch.Tensor, device: torch.device | None = None
) -> torch.Tensor:
    """
    Append ones to the last dimension of a 3D point vectors.

    Includes the convention that points have a 1 and directions have 0 as 4th dimension.
    This function can handle batched tensors.

    Parameters
    ----------
    points : torch.Tensor
        Input points in a 3D format.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Raises
    ------
    ValueError
        If the input is not 3D.

    Returns
    -------
    torch.Tensor
        Point vector with ones appended at the last dimension.
    """
    device = get_device(device=device)

    if points.size(dim=-1) != 3:
        raise ValueError(f"Expected 3D points but got points of shape {points.shape}!")

    ones_tensor = torch.ones(
        points.shape[:-1] + (1,), dtype=points.dtype, device=device
    )
    return torch.cat((points, ones_tensor), dim=-1)


def convert_3d_directions_to_4d_format(
    directions: torch.Tensor, device: torch.device | None = None
) -> torch.Tensor:
    """
    Append zeros to the last dimension of 3D direction vectors.

    Includes the convention that points have a 1 and directions have 0 as 4th dimension.
    This function can handle batched tensors.

    Parameters
    ----------
    directions : torch.Tensor
        Input direction in a 3D format.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Raises
    ------
    ValueError
        If the input is not 3D.

    Returns
    -------
    torch.Tensor
        Direction vectors with ones appended at the last dimension.
    """
    device = get_device(device=device)

    if directions.size(dim=-1) != 3:
        raise ValueError(
            f"Expected 3D directions but got directions of shape {directions.shape}!"
        )

    zeros_tensor = torch.zeros(
        directions.shape[:-1] + (1,), dtype=directions.dtype, device=device
    )
    return torch.cat((directions, zeros_tensor), dim=-1)


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
    range = points - min(points)
    points_normalized = (range + 1e-5) / max(range + 2e-5)
    return points_normalized


def decompose_rotations(
    initial_vector: torch.Tensor,
    target_vector: torch.Tensor,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get the individual angles along the east-, north- and up-axis, to rotate and initial vector into a target vector.

    Parameters
    ----------
    initial_vector : torch.Tensor
        The initial vector.
    rotated_vector : torch.Tensor
        The rotated vector.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The angle for the east-axis rotation.
    torch.Tensor
        The angle for the north-axis rotation.
    torch.Tensor
        The angle for the up-axis rotation.

    """
    device = get_device(device=device)

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
    device: torch.device | None = None,
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
        The initial orientation of the coordinate system.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The transformed angle in the rotated coordinate system.
    """
    device = get_device(device=device)

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
    bitmaps: torch.Tensor,
    target_centers: torch.Tensor,
    target_widths: float,
    target_heights: float,
    threshold: float = 0.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Calculate the coordinates of the flux density center of mass.

    First determine the indices of the bitmap center of mass.
    Next determine the position (coordinates) of the center of mass on the target.

    Parameters
    ----------
    bitmaps : torch.Tensor
        The flux densities in form of bitmaps.
    target_centers : torch.Tensor
        The positions of the centers of the targets.
    target_widths : float
        The widths of the target surfaces.
    target_heights : float
        The heights of the target surfaces.
    threshold : float
        Determines how intense a pixel in the bitmap needs to be to be registered (default is 0.0).
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The coordinates of the flux density centers of mass.
    """
    device = get_device(device=device)

    _, heights, widths = bitmaps.shape

    # Threshold the bitmap values. Any values below the threshold are set to zero.
    flux_thresholds = torch.where(
        bitmaps >= threshold, bitmaps, torch.zeros_like(bitmaps, device=device)
    )
    total_intensities = flux_thresholds.sum(dim=(1, 2))

    # Generate normalized east and up coordinates adjusted for pixel centers.
    # The "+ 0.5" adjustment ensures coordinates are centered within each pixel.
    e_indices = (
        torch.arange(widths, dtype=torch.float32, device=device) + 0.5
    ) / widths
    u_indices = (
        torch.arange(heights, dtype=torch.float32, device=device) + 0.5
    ) / heights

    # Compute the centers of intensity using weighted sums of the coordinates.
    center_of_masses_e = (
        torch.sum(
            (flux_thresholds.sum(dim=1).unsqueeze(1) * e_indices), dim=-1
        ).squeeze(-1)
        / total_intensities
    )
    center_of_masses_u = 1 - (
        torch.sum(
            (flux_thresholds.sum(dim=2).unsqueeze(1) * u_indices), dim=-1
        ).squeeze(-1)
        / total_intensities
    )

    # Construct the coordinates relative to target centers.
    de = torch.zeros((bitmaps.shape[0], 4), device=device)
    de[:, 0] = -target_widths
    du = torch.zeros((bitmaps.shape[0], 4), device=device)
    du[:, 2] = target_heights

    center_coordinates = (
        target_centers
        - 0.5 * (de + du)
        + center_of_masses_e.unsqueeze(-1) * de
        + center_of_masses_u.unsqueeze(-1) * du
    )

    return center_coordinates


def create_nurbs_evaluation_grid(
    number_of_evaluation_points: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Create a grid of evaluation points for a nurbs surface.

    Parameters
    ----------
    number_of_evaluation_points : torch.Tensor
        The number of nurbs evaluation points in east and north direction.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The evaluation points.
    """
    device = get_device(device=device)

    evaluation_points_e = torch.linspace(
        1e-5, 1 - 1e-5, number_of_evaluation_points[0], device=device
    )
    evaluation_points_n = torch.linspace(
        1e-5, 1 - 1e-5, number_of_evaluation_points[1], device=device
    )
    evaluation_points = torch.cartesian_prod(evaluation_points_e, evaluation_points_n)

    return evaluation_points


def normalize_bitmaps(
    flux_distributions: torch.Tensor,
    target_area_widths: torch.Tensor,
    target_area_heights: torch.Tensor,
    number_of_rays: torch.Tensor,
) -> torch.Tensor:
    """
    Normalize a bitmap.

    Parameters
    ----------
    flux_distributions : torch.Tensor
        The flux distributions to be normalized.
    target_area_widths : troch.Tensor
        The target area widths.
    target_area_heights : troch.Tensor
        The target area heights.
    number_of_rays : troch.Tensor
        The number of rays used to generate the flux.

    Returns
    -------
    torch.Tensor
        The normalized and scaled flux density distributions.
    """
    plane_areas = target_area_widths * target_area_heights
    num_pixels = flux_distributions.shape[1] * flux_distributions.shape[2]
    plane_area_per_pixel = plane_areas / num_pixels

    normalized_fluxes = flux_distributions / (
        number_of_rays * plane_area_per_pixel
    ).unsqueeze(-1).unsqueeze(-1)

    scaled_fluxes = (
        normalized_fluxes - torch.mean(normalized_fluxes, dim=(1, 2), keepdim=True)
    ) / torch.std(normalized_fluxes, dim=(1, 2), keepdim=True)

    return scaled_fluxes


def crop_image_region(
    images: torch.Tensor,
    crop_width_m: float,
    crop_height_m: float,
    target_plane_x_m: torch.Tensor,
    target_plane_y_m: torch.Tensor,
) -> torch.Tensor:
    """
    Crop a centered rectangular region from grayscale intensity images based on physical dimensions.

    This function identifies the center of mass in each image and then crops a region centered at this point
    with the specified physical width and height (in meters). The cropping is applied via affine transformation,
    which accounts for the desired crop size relative to the target's physical plane dimensions.

    Parameters
    ----------
    images : torch.Tensor
        Grayscale intensity images with shape (B, H, W), where B is the batch size.
    crop_width_m : float
        Desired width of the cropped region in meters.
    crop_height_m : float
        Desired height of the cropped region in meters.
    target_plane_x_m : torch.Tensor
        Physical width in meters of each image in the batch with shape (B,).
    target_plane_y_m : torch.Tensor
        Physical height in meters of each image in the batch with shape (B,).

    Returns
    -------
    torch.Tensor
        The cropped image regions of shape (B, H, W).
    """
    B, H, W = images.shape
    device = images.device

    # 1. Compute center of mass
    mass_map = images / (images.sum(dim=(1, 2), keepdim=True) + 1e-8)

    y_coords = torch.linspace(-1, 1, H, device=device)
    x_coords = torch.linspace(-1, 1, W, device=device)
    Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
    X = X.expand(B, -1, -1)
    Y = Y.expand(B, -1, -1)

    x_center = (X * mass_map).sum(dim=(1, 2))  # (B,)
    y_center = (Y * mass_map).sum(dim=(1, 2))  # (B,)

    # 2. Compute scale to match desired crop size in meters
    scale_x = crop_width_m / target_plane_x_m   # (B,)
    scale_y = crop_height_m / target_plane_y_m  # (B,)

    # 3. Build affine transform matrices (scale and center)
    theta = torch.zeros(B, 2, 3, device=device)
    theta[:, 0, 0] = scale_x
    theta[:, 1, 1] = scale_y
    theta[:, 0, 2] = x_center
    theta[:, 1, 2] = y_center

    # 4. Apply affine transform
    images_exp = images[:, None, :, :]  # (B, 1, H, W)
    grid = F.affine_grid(theta, size=images_exp.shape, align_corners=False)
    cropped = F.grid_sample(images_exp, grid, align_corners=False, padding_mode='zeros')

    return cropped[:, 0, :, :]  # (B, H, W)



