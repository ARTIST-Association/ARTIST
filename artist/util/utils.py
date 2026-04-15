import torch

from artist.field.solar_tower import SolarTower
from artist.field.tower_target_areas_cylindrical import TowerTargetAreasCylindrical
from artist.field.tower_target_areas_planar import TowerTargetAreasPlanar
from artist.util import config_dictionary, index_mapping
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
        Tensor of shape [number_of_heliostats, number_of_rays, number_of_surface_points].
    u : torch.Tensor
        Up rotation angles in radians.
        Tensor of shape [number_of_heliostats, number_of_rays, number_of_surface_points].
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Raises
    ------
    ValueError
        If the shapes of the input tensors do not match.

    Returns
    -------
    torch.Tensor
        Batched 4×4 rotation matrices, one per distortion sample.
        Tensor of shape [number_of_heliostats, number_of_rays, number_of_surface_points, 4, 4].
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

    matrix = torch.zeros(
        e.shape[index_mapping.heliostat_dimension],
        e.shape[index_mapping.facet_dimension],
        e.shape[index_mapping.points_dimension],
        4,
        4,
        device=device,
    )

    matrix[:, :, :, index_mapping.e, index_mapping.e] = cos_u
    matrix[:, :, :, index_mapping.e, index_mapping.n] = -sin_u
    matrix[:, :, :, index_mapping.n, index_mapping.e] = cos_e * sin_u
    matrix[:, :, :, index_mapping.n, index_mapping.n] = cos_e * cos_u
    matrix[:, :, :, index_mapping.n, index_mapping.u] = sin_e
    matrix[:, :, :, index_mapping.u, index_mapping.e] = -sin_e * sin_u
    matrix[:, :, :, index_mapping.u, index_mapping.n] = -sin_e * cos_u
    matrix[:, :, :, index_mapping.u, index_mapping.u] = cos_e
    matrix[
        :, :, :, index_mapping.transform_homogenous, index_mapping.transform_homogenous
    ] = ones

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
        Tensor of shape [number_of_heliostats].
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        Batched 4×4 east-axis rotation matrices, one per heliostat.
        Tensor of shape [number_of_heliostats, 4, 4].
    """
    device = get_device(device=device)

    cos_e = torch.cos(e)
    sin_e = torch.sin(e)
    ones = torch.ones(e.shape, device=device)

    matrix = torch.zeros(
        e.shape[index_mapping.heliostat_dimension], 4, 4, device=device
    )

    matrix[:, index_mapping.e, index_mapping.e] = ones
    matrix[:, index_mapping.n, index_mapping.n] = cos_e
    matrix[:, index_mapping.n, index_mapping.u] = -sin_e
    matrix[:, index_mapping.u, index_mapping.n] = sin_e
    matrix[:, index_mapping.u, index_mapping.u] = cos_e
    matrix[
        :, index_mapping.transform_homogenous, index_mapping.transform_homogenous
    ] = ones

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
        Tensor of shape [number_of_heliostats].
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        Batched 4×4 north-axis rotation matrices, one per heliostat.
        Tensor of shape [number_of_heliostats, 4, 4].
    """
    device = get_device(device=device)

    cos_n = torch.cos(n)
    sin_n = torch.sin(n)
    ones = torch.ones(n.shape, device=device)

    matrix = torch.zeros(
        n.shape[index_mapping.heliostat_dimension], 4, 4, device=device
    )

    matrix[:, index_mapping.e, index_mapping.e] = cos_n
    matrix[:, index_mapping.e, index_mapping.u] = -sin_n
    matrix[:, index_mapping.n, index_mapping.n] = ones
    matrix[:, index_mapping.u, index_mapping.e] = sin_n
    matrix[:, index_mapping.u, index_mapping.u] = cos_n
    matrix[
        :, index_mapping.transform_homogenous, index_mapping.transform_homogenous
    ] = ones

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
        Tensor of shape [number_of_heliostats].
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        Batched 4×4 up-axis rotation matrices, one per heliostat.
        Tensor of shape [number_of_heliostats, 4, 4].
    """
    device = get_device(device=device)

    cos_u = torch.cos(u)
    sin_u = torch.sin(u)
    ones = torch.ones(u.shape, device=device)

    matrix = torch.zeros(
        u.shape[index_mapping.heliostat_dimension], 4, 4, device=device
    )

    matrix[:, index_mapping.e, index_mapping.e] = cos_u
    matrix[:, index_mapping.e, index_mapping.n] = -sin_u
    matrix[:, index_mapping.n, index_mapping.e] = sin_u
    matrix[:, index_mapping.n, index_mapping.n] = cos_u
    matrix[:, index_mapping.u, index_mapping.u] = ones
    matrix[
        :, index_mapping.transform_homogenous, index_mapping.transform_homogenous
    ] = ones

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
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Raises
    ------
    ValueError
        If the sizes of the input tensors do not match.

    Returns
    -------
    torch.Tensor
        Batched 4×4 translation matrices, one per heliostat.
        Tensor of shape [number_of_heliostats, 4, 4].
    """
    device = get_device(device=device)

    if not (e.shape == u.shape == n.shape):
        raise ValueError(
            "The three tensors containing the east, north, and up translations must have the same shape."
        )

    ones = torch.ones(e.shape, device=device)

    matrix = torch.zeros(
        e.shape[index_mapping.heliostat_dimension], 4, 4, device=device
    )

    matrix[:, index_mapping.e, index_mapping.e] = ones
    matrix[:, index_mapping.e, index_mapping.transform_homogenous] = e
    matrix[:, index_mapping.n, index_mapping.n] = ones
    matrix[:, index_mapping.n, index_mapping.transform_homogenous] = n
    matrix[:, index_mapping.u, index_mapping.u] = ones
    matrix[:, index_mapping.u, index_mapping.transform_homogenous] = u
    matrix[
        :, index_mapping.transform_homogenous, index_mapping.transform_homogenous
    ] = ones

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
        Tensor of shape [..., 3]. The tensor may have arbitrary many batch dimensions, but the last shape dimension must be 3.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Raises
    ------
    ValueError
        If the input is not 3D.

    Returns
    -------
    torch.Tensor
        Point vector with ones appended at the last dimension.
        Tensor of shape [..., 4].
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
        Tensor of shape [..., 3]. The tensor may have arbitrary many batch dimensions, but the last shape dimension must be 3.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Raises
    ------
    ValueError
        If the input is not 3D.

    Returns
    -------
    torch.Tensor
        Direction vectors with ones appended at the last dimension.
        Tensor of shape [..., 4].
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
    Normalize each column of a 2D tensor to the open interval (0, 1).

    Parameters
    ----------
    points : torch.Tensor
        A tensor containing points to be normalized.
        Tensor of shape [number_of_points, 2].

    Returns
    -------
    torch.Tensor
        The normalized points.
        Tensor of shape [number_of_points, 2].
    """
    # Since we want the open interval (0,1), a small offset is required to also exclude the boundaries.
    min_vals = torch.min(points, dim=index_mapping.unbatched_tensor_values).values
    point_range = points - min_vals
    max_vals = torch.max(
        point_range + 2e-5, dim=index_mapping.unbatched_tensor_values
    ).values
    normalized = (point_range + 1e-5) / max_vals

    return normalized


def decompose_rotations(
    initial_vector: torch.Tensor,
    target_vector: torch.Tensor,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get the individual angles along the east-, north- and up-axis, to rotate an initial vector into a target vector.

    Parameters
    ----------
    initial_vector : torch.Tensor
        The initial vector.
        Tensor of shape [number_of_heliostats, 4].
    target_vector : torch.Tensor
        The target vector to rotate toward.
        Tensor of shape [4].
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
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
    initial_vector = torch.nn.functional.normalize(
        initial_vector[:, : index_mapping.slice_fourth_dimension]
    )
    target_vector = torch.nn.functional.normalize(
        target_vector[: index_mapping.slice_fourth_dimension],
        dim=index_mapping.unbatched_tensor_values,
    ).unsqueeze(index_mapping.unbatched_tensor_values)

    # Compute the cross product (rotation axis).
    r = torch.linalg.cross(initial_vector, target_vector)

    # Normalize the rotation axis.
    r_normalized = torch.nn.functional.normalize(r)

    # Compute the angle between the vectors.
    theta = torch.arccos(torch.clamp(initial_vector @ target_vector.T, -1.0, 1.0))

    # Decompose the angle along each axis.
    theta_components = theta * r_normalized

    return theta_components[:, 0], theta_components[:, 1], theta_components[:, 2]


def rotation_angle_and_axis(
    from_orientation: torch.Tensor,
    to_orientation: torch.Tensor,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the rotation axis and angle between two orientations.

    Parameters
    ----------
    from_orientation : torch.Tensor
        The original orientation.
        Tensor of shape [4].
    to_orientation : torch.Tensor
        The rotated orientation.
        Tensor of shape [4].
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The rotation axis.
        Tensor of shape [3].
    torch.Tensor
        The angle of the rotation.
        Tensor of shape [1].
    """
    device = get_device(device=device)

    from_orientation = from_orientation[:3] / torch.norm(from_orientation[:3])
    to_orientation = to_orientation[:3] / torch.norm(to_orientation[:3])

    dot = torch.clamp(torch.dot(from_orientation, to_orientation), -1.0, 1.0)
    angle = torch.acos(dot)

    axis = torch.linalg.cross(from_orientation, to_orientation)
    axis_norm = torch.norm(axis)

    # Parallel vectors.
    if axis_norm < 1e-6 and dot > 0:
        return torch.tensor([1.0, 0.0, 0.0], device=device), torch.tensor(
            0.0, device=device
        )

    # Inverse vectors.
    if axis_norm < 1e-6 and dot < 0:
        if abs(from_orientation[index_mapping.e]) < abs(
            from_orientation[index_mapping.n]
        ):
            orthogonal = torch.tensor([1.0, 0.0, 0.0], device=device)
        else:
            orthogonal = torch.tensor([0.0, 1.0, 0.0], device=device)
        axis = torch.linalg.cross(from_orientation, orthogonal)
        axis = axis / torch.norm(axis)
        return axis, torch.tensor(torch.pi, device=device)

    axis = axis / axis_norm

    return axis, angle


def get_center_of_mass(
    bitmaps: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Calculate the coordinates of the flux density center of mass.

    First determine the indices of the bitmap center of mass.
    Next determine the position (coordinates) of the center of mass on the target.

    Returns (0.0, 0.0) for empty fluxes.

    Parameters
    ----------
    bitmaps : torch.Tensor
        The flux densities in form of bitmaps.
        Tensor of shape [number_of_active_heliostats, bitmap_resolution_e, bitmap_resolution_u].
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        Bitmap coordinates of the flux density centers of mass (x pixel, y pixel).
        Tensor of shape [number_of_active_heliostats, 2].
    """
    device = get_device(device=device)

    number_of_flux_distributions, image_height, image_width = bitmaps.shape

    # Compute center of mass.
    normalized_bitmaps = bitmaps / (
        bitmaps.sum(
            dim=(index_mapping.batched_bitmap_e, index_mapping.batched_bitmap_u),
            keepdim=True,
        )
        + 1e-8
    )

    y_coordinates = torch.linspace(0, image_height - 1, image_height, device=device)
    x_coordinates = torch.linspace(0, image_width - 1, image_width, device=device)
    y_grid, x_grid = torch.meshgrid(y_coordinates, x_coordinates, indexing="ij")
    x_grid = x_grid.expand(number_of_flux_distributions, -1, -1)
    y_grid = y_grid.expand(number_of_flux_distributions, -1, -1)

    x_center_of_mass = (x_grid * normalized_bitmaps).sum(
        dim=(index_mapping.batched_bitmap_e, index_mapping.batched_bitmap_u)
    )
    y_center_of_mass = (y_grid * normalized_bitmaps).sum(
        dim=(index_mapping.batched_bitmap_e, index_mapping.batched_bitmap_u)
    )
    return torch.stack([x_center_of_mass, y_center_of_mass], dim=1)


def bitmap_coordinates_to_target_coordinates(
    bitmap_coordinates: torch.Tensor,
    bitmap_resolution: torch.Tensor,
    solar_tower: SolarTower,
    target_area_indices: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Convert bitmap pixel coordinates to 3D world coordinates on the target surface.

    For planar target areas the pixel coordinates are mapped linearly to target plane coordinates.
    For cylindrical target areas the pixel coordinates are mapped to cylindrical surface coordinates
    using the cylinder's radius, opening angle, height, axis, and normal.

    Parameters
    ----------
    bitmap_coordinates : torch.Tensor
        Pixel coordinates in the bitmap for each heliostat, as (x, y) pairs.
        Tensor of shape [number_of_active_heliostats, 2].
    bitmap_resolution : torch.Tensor
        The resolution of the bitmap (width, height) in pixels.
        Tensor of shape [2].
    solar_tower : SolarTower
        The solar tower containing all target area definitions (planar and cylindrical).
    target_area_indices : torch.Tensor
        Global target area index for each heliostat (planar indices first, cylindrical second).
        Tensor of shape [number_of_active_heliostats].
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        World coordinates on the target surface in homogeneous format.
        Tensor of shape [number_of_active_heliostats, 4].
    """
    device = get_device(device=device)

    center_coordinates = torch.zeros((target_area_indices.numel(), 4), device=device)
    center_coordinates[:, -1] = 1.0

    planar_mask = (
        target_area_indices
        < solar_tower.number_of_target_areas_per_type[index_mapping.planar_target_areas]
    )
    if target_area_indices[planar_mask].numel() > 0:
        planar_indices = target_area_indices[planar_mask]

        resolution = torch.tensor(
            [
                bitmap_resolution[index_mapping.unbatched_bitmap_e],
                1e-8,
                bitmap_resolution[index_mapping.unbatched_bitmap_u],
                1e-8,
            ],
            device=device,
        )
        coordinates = torch.zeros((planar_indices.numel(), 4), device=device)
        coordinates[:, index_mapping.e] = bitmap_coordinates[
            planar_mask, index_mapping.unbatched_bitmap_e
        ]
        coordinates[:, index_mapping.u] = bitmap_coordinates[
            planar_mask, index_mapping.unbatched_bitmap_u
        ]

        # Account for flips from bitmap to target coordinates.
        target_dimensions = torch.zeros((planar_mask.sum(), 4), device=device)
        planar: TowerTargetAreasPlanar = solar_tower.target_areas[
            index_mapping.planar_target_areas
        ]  # type: ignore[assignment]
        target_dimensions[:, index_mapping.e] = -planar.dimensions[
            planar_indices, index_mapping.target_dimensions_width
        ]
        target_dimensions[:, index_mapping.u] = -planar.dimensions[
            planar_indices, index_mapping.target_dimensions_height
        ]

        center_coordinates[planar_mask] = (
            planar.centers[planar_indices]
            - 0.5 * target_dimensions
            + coordinates / (resolution - 1) * target_dimensions
        )

    if target_area_indices[~planar_mask].numel() > 0:
        cylinder_indices = (
            target_area_indices[~planar_mask]
            - solar_tower.number_of_target_areas_per_type[
                index_mapping.planar_target_areas
            ]
        )

        cylindrical: TowerTargetAreasCylindrical = solar_tower.target_areas[
            index_mapping.cylindrical_target_areas
        ]  # type: ignore[assignment]
        cylinder_normals = cylindrical.normals[cylinder_indices][:, :3]
        cylinder_axes = cylindrical.axes[cylinder_indices][:, :3]
        cylinder_centers = cylindrical.centers[cylinder_indices]
        radii = cylindrical.radii[cylinder_indices].flatten()

        theta = (
            bitmap_coordinates[~planar_mask, index_mapping.unbatched_bitmap_e]
            / (bitmap_resolution[index_mapping.unbatched_bitmap_e] - 1)
            - 0.5
        ) * cylindrical.opening_angles[cylinder_indices].flatten()
        z = (
            (
                (bitmap_resolution[index_mapping.unbatched_bitmap_e] - 1)
                - bitmap_coordinates[~planar_mask, index_mapping.unbatched_bitmap_u]
            )
            / (bitmap_resolution[index_mapping.unbatched_bitmap_u] - 1)
            - 0.5
        ) * cylindrical.heights[cylinder_indices].flatten()

        v = torch.cross(cylinder_axes, cylinder_normals, dim=-1)

        center_coordinates[~planar_mask, :3] = cylinder_centers[:, :3] + (
            radii[:, None] * torch.cos(theta)[:, None] * cylinder_normals
            + radii[:, None] * torch.sin(theta)[:, None] * v
            + z[:, None] * cylinder_axes
        )

    return center_coordinates


def create_nurbs_evaluation_grid(
    number_of_evaluation_points: torch.Tensor,
    epsilon: float = 1e-7,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Create a grid of evaluation points for a NURBS surface.

    Parameters
    ----------
    number_of_evaluation_points : torch.Tensor
        The number of nurbs evaluation points in east and north direction.
        Tensor of shape [2].
    epsilon : float
        Offset for the NURBS evaluation points (default is 1e-7).
        NURBS are defined in the interval of [0, 1] but have numerical instabilities at their endpoints
        therefore the evaluation points need a small offset from the endpoints.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The evaluation points.
        Tensor of shape [number_of_evaluation_points_e * number_of_evaluation_points_n, 2].
    """
    device = get_device(device=device)

    evaluation_points_e = torch.linspace(
        epsilon,
        1 - epsilon,
        number_of_evaluation_points[index_mapping.evaluation_points_e],
        device=device,
    )
    evaluation_points_n = torch.linspace(
        epsilon,
        1 - epsilon,
        number_of_evaluation_points[index_mapping.evaluation_points_n],
        device=device,
    )
    evaluation_points = torch.cartesian_prod(evaluation_points_e, evaluation_points_n)

    return evaluation_points


def create_ideal_canted_nurbs_control_points(
    number_of_control_points: torch.Tensor,
    canting: torch.Tensor,
    facet_translation_vectors: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Create ideal, canted and translated control points for each facet.

    Parameters
    ----------
    number_of_control_points : torch.Tensor
        The number of NURBS control points.
        Tensor of shape [2].
    canting : torch.Tensor
        The canting vector for each facet.
        Tensor of shape [number_of_facets, 2, 4].
    facet_translation_vectors : torch.Tensor
        The facet translation vector for each facet.
        Tensor of shape [number_of_facets, 4].
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The canted and translated ideal NURBS control points.
        Tensor of shape [number_of_facets, number_of_control_points_u_direction, number_of_control_points_v_direction, 3].
    """
    device = get_device(device=device)

    number_of_facets = facet_translation_vectors.shape[
        index_mapping.facet_index_unbatched
    ]

    control_points = torch.zeros(
        (
            number_of_facets,
            number_of_control_points[index_mapping.nurbs_u],
            number_of_control_points[index_mapping.nurbs_v],
            3,
        ),
        device=device,
    )

    offsets_e = torch.linspace(
        0,
        1,
        control_points.shape[index_mapping.control_points_u_facet_batched],
        device=device,
    )
    offsets_n = torch.linspace(
        0,
        1,
        control_points.shape[index_mapping.control_points_v_facet_batched],
        device=device,
    )
    start = -torch.norm(canting, dim=index_mapping.canting)
    end = torch.norm(canting, dim=index_mapping.canting)
    origin_offsets_e = (
        start[:, index_mapping.e, None]
        + (end - start)[:, index_mapping.e, None] * offsets_e[None, :]
    )
    origin_offsets_n = (
        start[:, index_mapping.n, None]
        + (end - start)[:, index_mapping.n, None] * offsets_n[None, :]
    )

    control_points_e = origin_offsets_e[:, :, None].expand(
        -1, -1, number_of_control_points[index_mapping.nurbs_u]
    )
    control_points_n = origin_offsets_n[:, None, :].expand(
        -1, number_of_control_points[index_mapping.nurbs_v], -1
    )

    control_points[:, :, :, index_mapping.e] = control_points_e
    control_points[:, :, :, index_mapping.n] = control_points_n
    control_points[:, :, :, index_mapping.u] = 0

    return control_points


def perform_canting(
    canting_angles: torch.Tensor,
    data: torch.Tensor,
    inverse: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Perform canting (rotation) on data like surface points or surface normals.

    Parameters
    ----------
    canting_angles : torch.Tensor
        Canting angles.
        Tensor of shape [number_of_surfaces, number_of_facets, 2, 4].
    data : torch.Tensor
        Data to be canted.
        Tensor of shape [number_of_surfaces, number_of_facets, number_of_points_per_facet, 4].
    inverse : bool
        Indicates the direction of the rotation. Use ``inverse=False`` for canting and ``inverse=True`` for decanting (default is False).
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The (de-)canted data.
        Tensor of shape [number_of_surfaces, number_of_facets, number_of_points_per_facet, 4].
    """
    number_of_surfaces = data.shape[index_mapping.heliostat_dimension]
    number_of_facets_per_surface = data.shape[index_mapping.facet_dimension]
    rotation_matrix = torch.zeros(
        (number_of_surfaces, number_of_facets_per_surface, 4, 4), device=device
    )

    e = canting_angles[:, :, index_mapping.e, : index_mapping.slice_fourth_dimension]
    n = canting_angles[:, :, index_mapping.n, : index_mapping.slice_fourth_dimension]
    u = torch.linalg.cross(e, n, dim=2)

    rotation_matrix[:, :, : index_mapping.slice_fourth_dimension, index_mapping.e] = (
        torch.nn.functional.normalize(e, dim=-1)
    )
    rotation_matrix[:, :, : index_mapping.slice_fourth_dimension, index_mapping.n] = (
        torch.nn.functional.normalize(n, dim=-1)
    )
    rotation_matrix[:, :, : index_mapping.slice_fourth_dimension, index_mapping.u] = (
        torch.nn.functional.normalize(u, dim=-1)
    )

    rotation_matrix[
        :, :, index_mapping.transform_homogenous, index_mapping.transform_homogenous
    ] = 1.0

    if inverse:
        canted_data = data @ rotation_matrix
    else:
        canted_data = data @ rotation_matrix.mT

    return canted_data


def trapezoid_distribution(
    total_width: int,
    slope_width: int,
    plateau_width: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Create a one dimensional trapezoid distribution.

    If the total width is less than 2 * slope_width + plateau_width, the slope is cut off.
    If total total width is greater than 2 * slope_width + plateau_width the trapezoid is
    padded with zeros on both sides.

    Parameters
    ----------
    total_width : int
        The total width of the trapezoid.
    slope_width : int
        The width of the slope of the trapezoid.
    plateau_width : int
        The width of the plateau.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The one dimensional trapezoid distribution.
        Tensor of shape [total_width].
    """
    indices = torch.arange(total_width, device=device)
    center = (total_width - 1) / 2
    half_plateau = plateau_width / 2

    # Distances from the plateau edge.
    distances = torch.abs(indices - center) - half_plateau

    trapezoid = 1 - (distances / slope_width).clamp(min=0, max=1)

    return trapezoid


def crop_flux_distributions_around_center(
    flux_distributions: torch.Tensor,
    solar_tower: SolarTower,
    target_area_indices: torch.Tensor,
    crop_width: float = config_dictionary.utis_crop_width,
    crop_height: float = config_dictionary.utis_crop_height,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Crop a centered rectangular region from grayscale intensity images based on physical dimensions.

    This function identifies the center of mass in each image and then crops a region centered around this point
    with the specified physical width and height (in meters). The cropping is applied via affine transformation,
    which accounts for the desired crop size relative to the target's physical plane dimensions.

    Parameters
    ----------
    flux_distributions : torch.Tensor
        Flux density bitmaps, one per heliostat.
        Tensor of shape [number_of_bitmaps, bitmap_height, bitmap_width].
    solar_tower : SolarTower
        The solar tower containing the physical target area dimensions.
    target_area_indices : torch.Tensor
        Global target area index for each bitmap (planar indices first, cylindrical second).
        Tensor of shape [number_of_bitmaps].
    crop_width : float
        Desired width of the cropped region in meters (default is ``config_dictionary.utis_crop_width``).
    crop_height : float
        Desired height of the cropped region in meters (default is ``config_dictionary.utis_crop_height``).
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The cropped image regions.
        Tensor of shape [number_of_bitmaps, bitmap_height, bitmap_width].
    """
    device = get_device(device=device)

    number_of_flux_distributions, image_height, image_width = flux_distributions.shape

    # Compute center of mass.
    normalized_mass_map = flux_distributions / (
        flux_distributions.sum(
            dim=(index_mapping.batched_bitmap_e, index_mapping.batched_bitmap_u),
            keepdim=True,
        )
        + 1e-8
    )

    y_coordinates = torch.linspace(-1, 1, image_height, device=device)
    x_coordinates = torch.linspace(-1, 1, image_width, device=device)
    y_grid, x_grid = torch.meshgrid(y_coordinates, x_coordinates, indexing="ij")
    x_grid = x_grid.expand(number_of_flux_distributions, -1, -1)
    y_grid = y_grid.expand(number_of_flux_distributions, -1, -1)

    x_center_of_mass = (x_grid * normalized_mass_map).sum(
        dim=(index_mapping.batched_bitmap_e, index_mapping.batched_bitmap_u)
    )
    y_center_of_mass = (y_grid * normalized_mass_map).sum(
        dim=(index_mapping.batched_bitmap_e, index_mapping.batched_bitmap_u)
    )

    target_dimensions = torch.empty((number_of_flux_distributions, 2), device=device)
    planar_mask = (
        target_area_indices
        < solar_tower.number_of_target_areas_per_type[index_mapping.planar_target_areas]
    )
    if target_area_indices[planar_mask].numel() > 0:
        planar: TowerTargetAreasPlanar = solar_tower.target_areas[
            index_mapping.planar_target_areas
        ]  # type: ignore[assignment]
        target_dimensions[planar_mask] = planar.dimensions[
            target_area_indices[planar_mask]
        ]
    if target_area_indices[~planar_mask].numel() > 0:
        cylinder_indices = (
            target_area_indices[~planar_mask]
            - solar_tower.number_of_target_areas_per_type[
                index_mapping.planar_target_areas
            ]
        )
        cylindrical: TowerTargetAreasCylindrical = solar_tower.target_areas[
            index_mapping.cylindrical_target_areas
        ]  # type: ignore[assignment]
        target_dimensions[~planar_mask, index_mapping.target_dimensions_width] = (
            cylindrical.radii[cylinder_indices]
            * cylindrical.opening_angles[cylinder_indices]
        )
        target_dimensions[~planar_mask, index_mapping.target_dimensions_height] = (
            cylindrical.heights[cylinder_indices]
        )

    # Compute scale to match desired crop size in meters.
    scale_x = crop_width / target_dimensions[:, index_mapping.target_dimensions_width]
    scale_y = crop_height / target_dimensions[:, index_mapping.target_dimensions_height]

    # Build affine transform matrices (scale and center).
    affine_matrices = torch.zeros(number_of_flux_distributions, 2, 3, device=device)
    affine_matrices[:, index_mapping.e, index_mapping.e] = scale_x
    affine_matrices[:, index_mapping.n, index_mapping.n] = scale_y
    affine_matrices[:, index_mapping.e, index_mapping.u] = x_center_of_mass
    affine_matrices[:, index_mapping.n, index_mapping.u] = y_center_of_mass

    # Apply affine transform.
    images_expanded = flux_distributions[:, None, :, :]
    sampling_grid = torch.nn.functional.affine_grid(
        affine_matrices, size=images_expanded.shape, align_corners=True
    )
    cropped_images = torch.nn.functional.grid_sample(
        images_expanded, sampling_grid, align_corners=True, padding_mode="zeros"
    )

    return cropped_images[:, 0, :, :]


def azimuth_elevation_to_enu(
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    slant_range: float = 1.0,
    degree: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Transform coordinates from azimuth and elevation to east, north and up.

    This method assumes a south-oriented azimuth-elevation coordinate system, where 0° points toward the south.

    Parameters
    ----------
    azimuth : torch.Tensor
        Azimuth, 0° points toward the south (degrees).
        Tensor of shape [number_of_samples].
    elevation : torch.Tensor
        Elevation angle above horizon, neglecting aberrations (degrees).
        Tensor of shape [number_of_samples].
    slant_range : float
        Slant range in meters (default is 1.0).
    degree : bool
        Whether input is given in degrees (default is True).
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The east, north and up (ENU) coordinates.
        Tensor of shape [number_of_samples, 3].
    """
    device = get_device(device=device)

    if degree:
        elevation = torch.deg2rad(elevation)
        azimuth = torch.deg2rad(azimuth)

    azimuth[azimuth < 0] += 2 * torch.pi

    r = slant_range * torch.cos(elevation)

    enu = torch.zeros(
        (azimuth.shape[index_mapping.unbatched_tensor_values], 3), device=device
    )

    enu[:, index_mapping.e] = r * torch.sin(azimuth)
    enu[:, index_mapping.n] = -r * torch.cos(azimuth)
    enu[:, index_mapping.u] = slant_range * torch.sin(elevation)

    return enu


def convert_wgs84_coordinates_to_local_enu(
    coordinates_to_transform: torch.Tensor,
    reference_point: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Transform coordinates from latitude, longitude and altitude (WGS84) to local east, north and up (ENU).

    This function calculates the north and east offsets in meters of a coordinate from the reference point.
    It converts the latitude and longitude to radians, calculates the radius of curvature values,
    and then computes the offsets based on the differences between the coordinate and the reference point.
    Finally, it returns a tensor containing these offsets along with the altitude difference.

    Parameters
    ----------
    coordinates_to_transform : torch.Tensor
        The coordinates in latitude, longitude, altitude that are to be transformed.
        Tensor of shape [number_of_coordinates, 3].
    reference_point : torch.Tensor
        The center of origin of the ENU coordinate system in WGS84 coordinates.
        Tensor of shape [3].
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The east offsets in meters, north offsets in meters, and the altitude differences from the reference point.
        Tensor of shape [number_of_coordinates, 3].
    """
    device = get_device(device=device)

    transformed_coordinates = torch.zeros_like(
        coordinates_to_transform, dtype=torch.float32, device=device
    )

    wgs84_a = 6378137.0  # Major axis in meters.
    wgs84_b = 6356752.314245  # Minor axis in meters.
    wgs84_e2 = (wgs84_a**2 - wgs84_b**2) / wgs84_a**2  # Eccentricity squared.

    # Convert latitude and longitude to radians.
    latitudes = torch.deg2rad(coordinates_to_transform[:, index_mapping.latitude])
    longitudes = torch.deg2rad(coordinates_to_transform[:, index_mapping.longitude])
    latitude_reference_point = torch.deg2rad(reference_point[index_mapping.latitude])
    longitude_reference_point = torch.deg2rad(reference_point[index_mapping.longitude])

    # Calculate meridional radius of curvature for the first latitude.
    sin_lat1 = torch.sin(latitudes)
    rn1 = wgs84_a / torch.sqrt(1 - wgs84_e2 * sin_lat1**2)

    # Calculate transverse radius of curvature for the first latitude.
    rm1 = (wgs84_a * (1 - wgs84_e2)) / ((1 - wgs84_e2 * sin_lat1**2) ** 1.5)

    # Calculate delta latitude and delta longitude in radians.
    dlat_rad = latitude_reference_point - latitudes
    dlon_rad = longitude_reference_point - longitudes

    # Calculate north and east offsets in meters.
    transformed_coordinates[:, index_mapping.e] = -(
        dlon_rad * rn1 * torch.cos(latitudes)
    )
    transformed_coordinates[:, index_mapping.n] = -(dlat_rad * rm1)
    transformed_coordinates[:, index_mapping.u] = (
        coordinates_to_transform[:, index_mapping.altitude]
        - reference_point[index_mapping.altitude]
    )

    return transformed_coordinates
