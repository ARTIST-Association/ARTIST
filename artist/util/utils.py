from typing import cast

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
        Shape is ``[number_of_heliostats, number_of_rays, number_of_surface_points]``.
    u : torch.Tensor
        Up rotation angles in radians.
        Shape is ``[number_of_heliostats, number_of_rays, number_of_surface_points]``.
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
        Shape is ``[number_of_heliostats, number_of_rays, number_of_surface_points, 4, 4]``.
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
    ones = torch.ones_like(e, device=device)

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
        :,
        :,
        :,
        index_mapping.transform_homogeneous,
        index_mapping.transform_homogeneous,
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
        Shape is ``[number_of_heliostats]``.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        Batched 4×4 east-axis rotation matrices, one per heliostat.
        Shape is ``[number_of_heliostats, 4, 4]``.
    """
    device = get_device(device=device)

    cos_e = torch.cos(e)
    sin_e = torch.sin(e)
    ones = torch.ones_like(e, device=device)

    matrix = torch.zeros(
        e.shape[index_mapping.heliostat_dimension], 4, 4, device=device
    )

    matrix[:, index_mapping.e, index_mapping.e] = ones
    matrix[:, index_mapping.n, index_mapping.n] = cos_e
    matrix[:, index_mapping.n, index_mapping.u] = -sin_e
    matrix[:, index_mapping.u, index_mapping.n] = sin_e
    matrix[:, index_mapping.u, index_mapping.u] = cos_e
    matrix[
        :, index_mapping.transform_homogeneous, index_mapping.transform_homogeneous
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
        Shape is ``[number_of_heliostats]``.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        Batched 4×4 north-axis rotation matrices, one per heliostat.
        Shape is ``[number_of_heliostats, 4, 4]``.
    """
    device = get_device(device=device)

    cos_n = torch.cos(n)
    sin_n = torch.sin(n)
    ones = torch.ones_like(n, device=device)

    matrix = torch.zeros(
        n.shape[index_mapping.heliostat_dimension], 4, 4, device=device
    )

    matrix[:, index_mapping.e, index_mapping.e] = cos_n
    matrix[:, index_mapping.e, index_mapping.u] = -sin_n
    matrix[:, index_mapping.n, index_mapping.n] = ones
    matrix[:, index_mapping.u, index_mapping.e] = sin_n
    matrix[:, index_mapping.u, index_mapping.u] = cos_n
    matrix[
        :, index_mapping.transform_homogeneous, index_mapping.transform_homogeneous
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
        Shape is ``[number_of_heliostats]``.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        Batched 4×4 up-axis rotation matrices, one per heliostat.
        Shape is ``[number_of_heliostats, 4, 4]``.
    """
    device = get_device(device=device)

    cos_u = torch.cos(u)
    sin_u = torch.sin(u)
    ones = torch.ones_like(u, device=device)

    matrix = torch.zeros(
        u.shape[index_mapping.heliostat_dimension], 4, 4, device=device
    )

    matrix[:, index_mapping.e, index_mapping.e] = cos_u
    matrix[:, index_mapping.e, index_mapping.n] = -sin_u
    matrix[:, index_mapping.n, index_mapping.e] = sin_u
    matrix[:, index_mapping.n, index_mapping.n] = cos_u
    matrix[:, index_mapping.u, index_mapping.u] = ones
    matrix[
        :, index_mapping.transform_homogeneous, index_mapping.transform_homogeneous
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
        East translation distances in meters.
        Shape is ``[number_of_heliostats]``.
    n : torch.Tensor
        North translation distances in meters.
        Shape is ``[number_of_heliostats]``.
    u : torch.Tensor
        Up translation distances in meters.
        Shape is ``[number_of_heliostats]``.
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
        Shape is ``[number_of_heliostats, 4, 4]``.
    """
    device = get_device(device=device)

    if not (e.shape == u.shape == n.shape):
        raise ValueError(
            "The three tensors containing the east, north, and up translations must have the same shape."
        )

    ones = torch.ones_like(e, device=device)

    matrix = torch.zeros(
        e.shape[index_mapping.heliostat_dimension], 4, 4, device=device
    )

    matrix[:, index_mapping.e, index_mapping.e] = ones
    matrix[:, index_mapping.e, index_mapping.transform_homogeneous] = e
    matrix[:, index_mapping.n, index_mapping.n] = ones
    matrix[:, index_mapping.n, index_mapping.transform_homogeneous] = n
    matrix[:, index_mapping.u, index_mapping.u] = ones
    matrix[:, index_mapping.u, index_mapping.transform_homogeneous] = u
    matrix[
        :, index_mapping.transform_homogeneous, index_mapping.transform_homogeneous
    ] = ones

    return matrix


def convert_3d_points_to_4d_format(
    points: torch.Tensor, device: torch.device | None = None
) -> torch.Tensor:
    """
    Append ones to the last dimension of 3D point vectors.

    Includes the convention that points have a 1 and directions have a 0 as 4th dimension.
    This function can handle batched tensors.

    Parameters
    ----------
    points : torch.Tensor
        Input points in a 3D format.
        Shape is ``[..., 3]``.
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
        Shape is ``[..., 4]``.
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

    Includes the convention that points have a 1 and directions have a 0 as 4th dimension.
    This function can handle batched tensors.

    Parameters
    ----------
    directions : torch.Tensor
        Input direction in a 3D format.
        Shape is ``[..., 3]``.
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
        Direction vectors with zeros appended at the last dimension.
        Shape is ``[..., 4]``.
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
        Shape is ``[number_of_points, number_of_dimensions]``.

    Returns
    -------
    torch.Tensor
        The normalized points.
        Shape is ``[number_of_points, number_of_dimensions]``.
    """
    # Since we want the open interval (0,1), a small offset is required to also exclude the boundaries.
    min_vals = torch.min(points, dim=index_mapping.unbatched_tensor_values).values
    point_range = points - min_vals
    max_vals = torch.max(
        point_range + 2e-5, dim=index_mapping.unbatched_tensor_values
    ).values
    return (point_range + 1e-5) / max_vals


def decompose_rotations(
    initial_vector: torch.Tensor,
    target_vector: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute ENU components of the axis-angle rotation vector that rotates initial vectors toward a target vector.

    This function does **not** perform an Euler-angle decomposition. Instead, it computes:
    1) the rotation axis via cross product, and
    2) the rotation magnitude via arccos of the dot product,
    then returns the Cartesian components of the rotation vector
    (`theta * axis`) in east, north, and up coordinates.

    Parameters
    ----------
    initial_vector : torch.Tensor
        Initial vectors in homogeneous coordinates.
        Shape is ``[number_of_heliostats, 4]``.
        Only the first three components (ENU) are used.
    target_vector : torch.Tensor
        Target vector in homogeneous coordinates.
        Shape is ``[4]``.
        Only the first three components (ENU) are used.

    Returns
    -------
    torch.Tensor
        East component of the axis-angle rotation vector.
        Shape is ``[number_of_heliostats]``.
    torch.Tensor
        North component of the axis-angle rotation vector.
        Shape is ``[number_of_heliostats]``.
    torch.Tensor
        Up component of the axis-angle rotation vector.
        Shape is ``[number_of_heliostats]``.
    """
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
        Shape is ``[4]``.
    to_orientation : torch.Tensor
        The rotated orientation.
        Shape is ``[4]``.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The rotation axis.
        Shape is ``[3]``.
    torch.Tensor
        The angle of the rotation as a scalar tensor.
    """
    device = get_device(device=device)
    from_orientation = from_orientation[:3] / torch.norm(from_orientation[:3])
    to_orientation = to_orientation[:3] / torch.norm(to_orientation[:3])
    dot = torch.clamp(torch.dot(from_orientation, to_orientation), -1.0, 1.0)
    angle = torch.acos(dot)
    axis = torch.linalg.cross(from_orientation, to_orientation)
    axis_norm = torch.norm(axis)
    # Parallel vectors.
    epsilon = 1e-6
    if axis_norm < epsilon and dot > 0:
        return torch.tensor([1.0, 0.0, 0.0], device=device), torch.tensor(
            0.0, device=device
        )
    # Inverse vectors.
    if axis_norm < epsilon and dot < 0:
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

    Convention
    ----------
    Resolution is conceptually defined as:
        [W, H]  # width, height

    Tensor storage follows PyTorch convention:
        [H, W]  # height, width

    Axis mapping in this function:
        e (horizontal / width / x-axis)  -> dim 2
        u (vertical   / height / y-axis) -> dim 1

    So:
        bitmaps shape = [batch, u, e] = [batch, H, W]

    First determine the indices of the bitmap center of mass.
    Next determine the position (coordinates) of the center of mass on the target.

    Returns (0.0, 0.0) for empty fluxes.

    Layer	Order	Meaning
    tensor	[u, e]	memory layout
    output	(e, u)	geometry

    Parameters
    ----------
    bitmaps : torch.Tensor
        Flux densities in form of bitmaps.
        Shape is ``[number_of_active_heliostats, bitmap_resolution_e, bitmap_resolution_u]``.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        Bitmap coordinates of the flux density centers of mass (x pixel, y pixel).
        Shape is ``[number_of_active_heliostats, 2]`` as (e, u).
    """
    device = get_device(device=device)

    batch, height_u, width_e = bitmaps.shape

    normalized_bitmaps = bitmaps / (
        bitmaps.sum(
            dim=(index_mapping.batched_bitmap_u, index_mapping.batched_bitmap_e),
            keepdim=True,
        )
        + 1e-8
    )

    # e = horizontal axis = width = dim 2
    e_coords = torch.linspace(0, width_e - 1, width_e, device=device)
    # u = vertical axis = height = dim 1
    u_coords = torch.linspace(0, height_u - 1, height_u, device=device)

    # meshgrid in (u, e) order because tensor is [u, e]
    u_grid, e_grid = torch.meshgrid(u_coords, e_coords, indexing="ij")
    u_grid = u_grid.expand(batch, -1, -1)
    e_grid = e_grid.expand(batch, -1, -1)

    # center of mass
    e_center_of_mass = (e_grid * normalized_bitmaps).sum(
        dim=(index_mapping.batched_bitmap_u, index_mapping.batched_bitmap_e)
    )
    u_center_of_mass = (u_grid * normalized_bitmaps).sum(
        dim=(index_mapping.batched_bitmap_u, index_mapping.batched_bitmap_e)
    )

    return torch.stack([e_center_of_mass, u_center_of_mass], dim=1)

def bitmap_coordinates_to_target_coordinates(
    bitmap_coordinates: torch.Tensor,
    bitmap_resolution: torch.Tensor,
    solar_tower: SolarTower,
    target_area_indices: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Map 2D bitmap pixel coordinates to 3D world coordinates on planar or cylindrical target surfaces.

    This function implements a unified projection model that converts discrete bitmap sampling
    coordinates into continuous 3D positions on the solar tower target geometry.

    The mapping supports:
        - Planar target areas (rectangular surfaces in 3D space)
        - Cylindrical target areas (partial cylindrical surfaces)

    ------------------------------------------------------------------------
    BITMAP CONVENTIONS
    ------------------------------------------------------------------------
    The bitmap is treated as a discrete image grid with resolution:

        bitmap_resolution = [width, height]

    Pixel coordinates follow image indexing conventions:

        bitmap_coordinates[..., e] ∈ [0, W-1]
        bitmap_coordinates[..., u] ∈ [0, H-1]

    where:
        e-axis → horizontal axis (width direction)
        u-axis → vertical axis (height direction)

    IMPORTANT CONVENTION:
        Pixel coordinates are interpreted as CENTERED pixels using:

            e_norm = (e + 0.5) / W
            u_norm = (u + 0.5) / H

        This ensures each pixel represents its spatial cell center rather than its corner.

    ------------------------------------------------------------------------
    WORLD / TARGET COORDINATE CONVENTIONS
    ------------------------------------------------------------------------

    PLANAR TARGETS:
        - Each plane has a center point and two fixed global axes:
              e_axis = (1, 0, 0)
              u_axis = (0, 0, 1)

        - These define a 2D coordinate system embedded in 3D space.

        - Mapping:
              e_local = (0.5 - e_norm) * width
              u_local = (0.5 - u_norm) * height

        NOTE:
            The e-axis is intentionally flipped (0.5 - e_norm) to match
            the desired bitmap orientation.

            This means:
                increasing bitmap x → decreases world x

            This convention is deliberate and consistent across inverse mapping.

    CYLINDRICAL TARGETS:
        - Each cylindrical segment is defined by:
              center
              axis (longitudinal direction)
              normal (radial reference direction)
              radius
              height
              opening angle

        - A local orthonormal basis is constructed:
              v = axis × normal

        - Angular mapping:
              theta = (e_norm - 0.5) * opening_angle

        - Axial mapping:
              z = (0.5 - u_norm) * height

        - Final mapping:
              world = center
                    + r cos(theta) * normal
                    + r sin(theta) * v
                    + z * axis

    ------------------------------------------------------------------------
    COORDINATE SYSTEM CONSISTENCY
    ------------------------------------------------------------------------

    - Bitmap space: discrete grid, centered pixel sampling
    - Planar world space: fixed global axes (hardcoded orientation)
    - Cylindrical world space: local basis derived per target

    Vertical convention:
        Both planar and cylindrical mappings use:
            u_norm ↑ → world z decreases

        i.e.:
            top of bitmap maps to +height/2
            bottom maps to -height/2

    ------------------------------------------------------------------------
    PARAMETERS
    ------------------------------------------------------------------------
    bitmap_coordinates : torch.Tensor
        Pixel coordinates per sample.
        Shape: [N, 2] where columns are (e, u)

    bitmap_resolution : torch.Tensor
        Bitmap resolution as [W, H].

    solar_tower : SolarTower
        Geometry container defining planar and cylindrical target areas.

    target_area_indices : torch.Tensor
        Index of target surface per sample.

    device : torch.device | None
        Computation device.

    ------------------------------------------------------------------------
    RETURNS
    ------------------------------------------------------------------------
    torch.Tensor
        Homogeneous world coordinates [x, y, z, 1].
        Shape: [N, 4]

    ------------------------------------------------------------------------
    NOTES
    ------------------------------------------------------------------------
    - Pixel coordinates are treated as centered (0.5 offset convention).
    - Planar axes are intentionally hardcoded to a global orientation.
    - Cylindrical mapping uses local orthonormal basis per target.
    - The mapping is designed to be invertible with the corresponding
      world → bitmap function.
    """
    device = get_device(device=device)

    N = target_area_indices.numel()
    target_coordinates = torch.zeros((N, 4), device=device)
    target_coordinates[:, -1] = 1.0

    W = bitmap_resolution[index_mapping.unbatched_bitmap_e]
    H = bitmap_resolution[index_mapping.unbatched_bitmap_u]

    e_norm = (bitmap_coordinates[:, index_mapping.unbatched_bitmap_e] + 0.5) / W
    u_norm = (bitmap_coordinates[:, index_mapping.unbatched_bitmap_u] + 0.5) / H

    planar_mask = (
        target_area_indices
        < solar_tower.number_of_target_areas_per_type[index_mapping.planar_target_areas]
    )

    if planar_mask.any():
        planar_indices = target_area_indices[planar_mask]
        
        planar = cast(
            TowerTargetAreasPlanar,
            solar_tower.target_areas[index_mapping.planar_target_areas],
        )

        centers = planar.centers[planar_indices][:, :3]
        dims = planar.dimensions[planar_indices]

        e_axis = torch.tensor([1.0, 0.0, 0.0], device=device).expand(planar_indices.numel(), 3)
        u_axis = torch.tensor([0.0, 0.0, 1.0], device=device).expand(planar_indices.numel(), 3)

        e_local = (0.5 - e_norm[planar_mask]) * dims[:, index_mapping.target_dimensions_width]
        u_local = (0.5 - u_norm[planar_mask]) * dims[:, index_mapping.target_dimensions_height]
        
        target_coordinates[planar_mask, :3] = (
            centers
            + e_local[:, None] * e_axis
            + u_local[:, None] * u_axis
        )

    if (~planar_mask).any():
        cylinder_indices = (
            target_area_indices[~planar_mask]
            - solar_tower.number_of_target_areas_per_type[index_mapping.planar_target_areas]
        )
        cylindrical = cast(
            TowerTargetAreasCylindrical,
            solar_tower.target_areas[index_mapping.cylindrical_target_areas],
        )

        centers = cylindrical.centers[cylinder_indices][:, :3]
        axes = cylindrical.axes[cylinder_indices][:, :3]
        normals = cylindrical.normals[cylinder_indices][:, :3]
        radii = cylindrical.radii[cylinder_indices].flatten()
        heights = cylindrical.heights[cylinder_indices].flatten()
        opening_angles = cylindrical.opening_angles[cylinder_indices].flatten()

        v = torch.cross(axes, normals, dim=-1)

        theta = (e_norm[~planar_mask] - 0.5) * opening_angles
        z = (0.5 - u_norm[~planar_mask]) * heights

        target_coordinates[~planar_mask, :3] = (
            centers
            + radii[:, None] * torch.cos(theta)[:, None] * normals
            + radii[:, None] * torch.sin(theta)[:, None] * v
            + z[:, None] * axes
        )

    return target_coordinates


def target_coordinates_to_bitmap_coordinates(
    world_coordinates: torch.Tensor,
    bitmap_resolution: torch.Tensor,
    solar_tower: SolarTower,
    target_area_indices: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Map 3D world coordinates on planar or cylindrical solar tower targets
    back into 2D bitmap pixel coordinates.

    This function is the exact inverse of
    `bitmap_coordinates_to_target_coordinates`, using identical geometric
    conventions and centered-pixel sampling.

    ------------------------------------------------------------------------
    BITMAP CONVENTIONS
    ------------------------------------------------------------------------
    Bitmap resolution is defined as:

        bitmap_resolution = [W, H]

    Pixel coordinates are interpreted as CENTERED samples:

        e_norm = (e + 0.5) / W
        u_norm = (u + 0.5) / H

    Inverse mapping:

        e = e_norm * W - 0.5
        u = u_norm * H - 0.5

    ------------------------------------------------------------------------
    PLANAR TARGETS
    ------------------------------------------------------------------------
    Planar targets use a fixed global basis:

        e_axis = (1, 0, 0)
        u_axis = (0, 0, 1)

    IMPORTANT SIGN CONVENTION:

        Forward mapping uses:
            e_local = (0.5 - e_norm) * width
            u_local = (0.5 - u_norm) * height

        Therefore inverse projection must include sign correction:

            e_local = -dot(world - center, e_axis)
            u_local = -dot(world - center, u_axis)

    Normalization:

        e_norm = e_local / width + 0.5
        u_norm = u_local / height + 0.5

    ------------------------------------------------------------------------
    CYLINDRICAL TARGETS
    ------------------------------------------------------------------------
    Cylindrical targets define a local orthonormal basis:

        axis   → cylinder axis (height direction)
        normal → radial reference direction
        v      → axis × normal

    Angular coordinate:

        theta = atan2(dot(rel, v), dot(rel, normal))

    Axial coordinate:

        z = dot(rel, axis)

    Normalization:

        e_norm = theta / opening_angle + 0.5
        u_norm = -z / height + 0.5

    ------------------------------------------------------------------------
    CONSISTENCY GUARANTEE
    ------------------------------------------------------------------------
    This function is the exact inverse of the forward mapping:
        bitmap → world

    under identical assumptions:
        - centered pixel sampling
        - fixed planar axes
        - consistent cylindrical basis
        - shared sign conventions

    ------------------------------------------------------------------------
    RETURNS
    ------------------------------------------------------------------------
    torch.Tensor
        Bitmap coordinates [e, u] per input world point.
        Shape: [N, 2]
    """

    device = get_device(device=device)

    world_xyz = world_coordinates[:, :3]
    N = world_xyz.shape[0]

    bitmap_coords = torch.zeros((N, 2), device=device)

    W = bitmap_resolution[index_mapping.unbatched_bitmap_e]
    H = bitmap_resolution[index_mapping.unbatched_bitmap_u]

    planar_mask = (
        target_area_indices
        < solar_tower.number_of_target_areas_per_type[index_mapping.planar_target_areas]
    )

    if planar_mask.any():
        planar_indices = target_area_indices[planar_mask]

        planar = cast(
            TowerTargetAreasPlanar,
            solar_tower.target_areas[index_mapping.planar_target_areas],
        )

        centers = planar.centers[planar_indices][:, :3]
        dims = planar.dimensions[planar_indices]

        e_axis = torch.tensor([1.0, 0.0, 0.0], device=device).expand(planar_indices.numel(), 3)
        u_axis = torch.tensor([0.0, 0.0, 1.0], device=device).expand(planar_indices.numel(), 3)

        rel = world_xyz[planar_mask] - centers

        e_local = -torch.sum(rel * e_axis, dim=-1)
        u_local = -torch.sum(rel * u_axis, dim=-1)

        e_norm = e_local / dims[:, index_mapping.target_dimensions_width] + 0.5
        u_norm = u_local / dims[:, index_mapping.target_dimensions_height] + 0.5

        bitmap_coords[planar_mask, index_mapping.unbatched_bitmap_e] = e_norm * W - 0.5
        bitmap_coords[planar_mask, index_mapping.unbatched_bitmap_u] = u_norm * H - 0.5

    if (~planar_mask).any():
        cylinder_indices = (
            target_area_indices[~planar_mask]
            - solar_tower.number_of_target_areas_per_type[index_mapping.planar_target_areas]
        )
        cylindrical = cast(
            TowerTargetAreasCylindrical,
            solar_tower.target_areas[index_mapping.cylindrical_target_areas],
        )

        centers = cylindrical.centers[cylinder_indices][:, :3]
        axes = cylindrical.axes[cylinder_indices][:, :3]
        normals = cylindrical.normals[cylinder_indices][:, :3]
        radii = cylindrical.radii[cylinder_indices].flatten()
        heights = cylindrical.heights[cylinder_indices].flatten()
        opening_angles = cylindrical.opening_angles[cylinder_indices].flatten()

        v = torch.cross(axes, normals, dim=-1)

        rel = world_xyz[~planar_mask] - centers

        z = torch.sum(rel * axes, dim=-1)

        x_n = torch.sum(rel * normals, dim=-1)
        x_v = torch.sum(rel * v, dim=-1)

        theta = torch.atan2(x_v, x_n)

        e_norm = theta / opening_angles + 0.5
        u_norm = -z / heights + 0.5

        bitmap_coords[~planar_mask, index_mapping.unbatched_bitmap_e] = e_norm * W - 0.5
        bitmap_coords[~planar_mask, index_mapping.unbatched_bitmap_u] = u_norm * H - 0.5

    return bitmap_coords


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
        Shape is ``[2]``.
    epsilon : float
        Offset for the NURBS evaluation points (default is 1e-7).
        NURBS are defined in the interval of [0, 1] but have numerical instabilities at their endpoints.
        Therefore, the evaluation points need a small offset from the endpoints.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The evaluation points.
        Shape is ``[number_of_evaluation_points_e * number_of_evaluation_points_n, 2]``.
    """
    device = get_device(device=device)

    evaluation_points_e = torch.linspace(
        epsilon,
        1 - epsilon,
        int(number_of_evaluation_points[index_mapping.evaluation_points_e].item()),
        device=device,
    )
    evaluation_points_n = torch.linspace(
        epsilon,
        1 - epsilon,
        int(number_of_evaluation_points[index_mapping.evaluation_points_n].item()),
        device=device,
    )
    return torch.cartesian_prod(evaluation_points_e, evaluation_points_n)


def create_planar_nurbs_control_points(
    number_of_control_points: torch.Tensor,
    canting: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Create planar NURBS control points for each facet.

    The generated control points form a flat, equidistant grid.
    The grid extent is derived from the norm of the canting vectors, which encode the
    dimensions of the facets.

    Parameters
    ----------
    number_of_control_points : torch.Tensor
        The number of NURBS control points.
        Shape is ``[2]``.
    canting : torch.Tensor
        The canting vector for each facet.
        Shape is ``[number_of_facets, 2, 4]``.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        Planar control point grids for each facet.
        Shape is ``[number_of_facets, number_of_control_points_u_direction, number_of_control_points_v_direction, 3]``.
    """
    device = get_device(device=device)

    number_of_control_points = number_of_control_points.to(device)
    canting = canting.to(device)

    n_u = int(number_of_control_points[index_mapping.nurbs_u].item())
    n_v = int(number_of_control_points[index_mapping.nurbs_v].item())

    number_of_facets = canting.shape[index_mapping.facet_index_unbatched]

    control_points = torch.zeros(
        (
            number_of_facets,
            n_u,
            n_v,
            3,
        ),
        device=device,
        dtype=canting.dtype,
    )

    u_lin = torch.linspace(0, 1, n_u, device=device, dtype=canting.dtype)
    v_lin = torch.linspace(0, 1, n_v, device=device, dtype=canting.dtype)

    # Per-facet extents in local in-plane directions.
    facet_dimensions = torch.norm(canting, dim=index_mapping.canting)
    u_coordinates = (
        -facet_dimensions[:, index_mapping.e, None]
        + 2 * facet_dimensions[:, index_mapping.e, None] * u_lin
    )
    v_coordinates = (
        -facet_dimensions[:, index_mapping.n, None]
        + 2 * facet_dimensions[:, index_mapping.n, None] * v_lin
    )

    control_points[..., index_mapping.nurbs_u] = u_coordinates[:, :, None]
    control_points[..., index_mapping.nurbs_v] = v_coordinates[:, None, :]

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
        Shape is ``[number_of_surfaces, number_of_facets, 2, 4]``.
    data : torch.Tensor
        Data to be canted.
        Shape is ``[number_of_surfaces, number_of_facets, number_of_points_per_facet, 4]``.
    inverse : bool
        Indicates the direction of the rotation.
        Use ``inverse=False`` for canting and ``inverse=True`` for decanting (default is False).
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The (de-)canted data.
        Shape is ``[number_of_surfaces, number_of_facets, number_of_points_per_facet, 4]``.
    """
    device = get_device(device=device)
    canting_angles = canting_angles.to(device)
    data = data.to(device)

    number_of_surfaces = data.shape[index_mapping.heliostat_dimension]
    number_of_facets_per_surface = data.shape[index_mapping.facet_dimension]

    rotation_matrix = torch.zeros(
        (number_of_surfaces, number_of_facets_per_surface, 4, 4),
        device=device,
        dtype=data.dtype,
    )

    # Extract ENU basis candidates from canting tensor (drop homogeneous component).
    e = canting_angles[:, :, index_mapping.e, : index_mapping.slice_fourth_dimension]
    n = canting_angles[:, :, index_mapping.n, : index_mapping.slice_fourth_dimension]

    # Build a numerically stable orthonormal basis:
    # 1) normalize e
    e = torch.nn.functional.normalize(e, dim=-1)
    # 2) u = normalize(e x n)
    u = torch.linalg.cross(e, n, dim=-1)
    u = torch.nn.functional.normalize(u, dim=-1, eps=1e-8)
    # 3) n_ortho = normalize(u x e)
    n_ortho = torch.linalg.cross(u, e, dim=-1)
    n_ortho = torch.nn.functional.normalize(n_ortho, dim=-1, eps=1e-8)

    # Fill rotation matrix columns with ENU basis vectors.
    rotation_matrix[:, :, : index_mapping.slice_fourth_dimension, index_mapping.e] = e
    rotation_matrix[:, :, : index_mapping.slice_fourth_dimension, index_mapping.n] = (
        n_ortho
    )
    rotation_matrix[:, :, : index_mapping.slice_fourth_dimension, index_mapping.u] = u
    rotation_matrix[
        :, :, index_mapping.transform_homogeneous, index_mapping.transform_homogeneous
    ] = 1.0

    # Data is represented as row vectors (..., 4); therefore:
    # - forward canting uses R^T
    # - inverse canting uses R
    if inverse:
        return data @ rotation_matrix
    return data @ rotation_matrix.mT


def trapezoid_distribution(
    total_width: int,
    slope_width: int,
    plateau_width: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Create a one-dimensional trapezoid distribution.

    If the total width is less than ``2 * slope_width + plateau_width``, the slope is cut off.
    If the total width is greater than ``2 * slope_width + plateau_width``, the trapezoid is
    padded with zeros on both sides.

    Parameters
    ----------
    total_width : int
        The total width of the trapezoid. Must be > 0.
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
        Shape is ``[total_width]``.
    """
    device = get_device(device=device)
    indices = torch.arange(total_width, device=device)
    center = (total_width - 1) / 2.0
    half_plateau = plateau_width / 2.0

    # Distances from the plateau edge.
    distances = torch.abs(indices - center) - half_plateau

    # Special case: no slope -> hard plateau/rectangle.
    if slope_width == 0:
        return (distances <= 0).to(dtype=torch.float32)

    return 1 - (distances / slope_width).clamp(min=0, max=1)


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
        Shape is ``[number_of_bitmaps, bitmap_height, bitmap_width]``.
    solar_tower : SolarTower
        Solar tower containing the physical target area dimensions.
    target_area_indices : torch.Tensor
        Global target area index for each bitmap (planar indices first, cylindrical second).
        Shape is ``[number_of_bitmaps]``.
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
        Cropped and centered image regions.
        Shape is ``[number_of_bitmaps, bitmap_height, bitmap_width]``.
    """
    device = get_device(device=device)

    target_area_indices = target_area_indices.to(device)
    flux_distributions = flux_distributions.to(device)

    number_of_flux_distributions, image_height, image_width = flux_distributions.shape

    # Compute center of mass in normalized image coordinates [-1, 1].
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

    # Gather target dimensions per bitmap (width, height) in meters.
    target_dimensions = torch.empty((number_of_flux_distributions, 2), device=device)
    planar_mask = (
        target_area_indices
        < solar_tower.number_of_target_areas_per_type[index_mapping.planar_target_areas]
    )
    if target_area_indices[planar_mask].numel() > 0:
        planar = cast(
            TowerTargetAreasPlanar,
            solar_tower.target_areas[index_mapping.planar_target_areas],
        )
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
        cylindrical = cast(
            TowerTargetAreasCylindrical,
            solar_tower.target_areas[index_mapping.cylindrical_target_areas],
        )
        target_dimensions[~planar_mask, index_mapping.target_dimensions_width] = (
            cylindrical.radii[cylinder_indices]
            * cylindrical.opening_angles[cylinder_indices]
        )
        target_dimensions[~planar_mask, index_mapping.target_dimensions_height] = (
            cylindrical.heights[cylinder_indices]
        )

    # Robust division for very small dimensions.
    epsilon = 1e-8
    width = target_dimensions[:, index_mapping.target_dimensions_width].clamp(
        min=epsilon
    )
    height = target_dimensions[:, index_mapping.target_dimensions_height].clamp(
        min=epsilon
    )

    # Compute scale to match desired crop size in meters.
    scale_x = crop_width / width
    scale_y = crop_height / height

    # Build affine transform matrices (scale and center).
    affine_matrices = torch.zeros(number_of_flux_distributions, 2, 3, device=device)
    affine_matrices[:, index_mapping.e, index_mapping.e] = scale_x
    affine_matrices[:, index_mapping.n, index_mapping.n] = scale_y
    affine_matrices[:, index_mapping.e, index_mapping.u] = x_center_of_mass
    affine_matrices[:, index_mapping.n, index_mapping.u] = y_center_of_mass

    # Apply affine transform.
    images_expanded = flux_distributions[:, None, :, :]
    sampling_grid = torch.nn.functional.affine_grid(
        affine_matrices, size=list(images_expanded.shape), align_corners=True
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
    Transform coordinates from azimuth and elevation to east, north, and up.

    This method assumes a south-oriented azimuth-elevation coordinate system, where 0° points toward the south.

    Parameters
    ----------
    azimuth : torch.Tensor
        Azimuth, 0° points toward the south (degrees).
        Shape is ``[number_of_samples]``.
    elevation : torch.Tensor
        Elevation angle above horizon, neglecting aberrations (degrees).
        Shape is ``[number_of_samples]``.
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
        The east, north, and up (ENU) coordinates.
        Shape is ``[number_of_samples, 3]``.
    """
    device = get_device(device=device)

    azimuth = azimuth.to(device=device, dtype=torch.float32)
    elevation = elevation.to(device=device, dtype=torch.float32)

    if azimuth.shape != elevation.shape:
        raise ValueError("``azimuth`` and ``elevation`` must have identical shapes.")

    if degree:
        elevation = torch.deg2rad(elevation)
        azimuth = torch.deg2rad(azimuth)

    # Normalize azimuth to [0, 2π).
    azimuth = torch.remainder(azimuth, 2 * torch.pi)

    r = slant_range * torch.cos(elevation)

    enu = torch.zeros(
        (azimuth.shape[index_mapping.unbatched_tensor_values], 3), device=device
    )

    enu[:, index_mapping.e] = r * torch.sin(azimuth)
    enu[:, index_mapping.n] = -r * torch.cos(
        azimuth
    )  # South-oriented azimuth convention
    enu[:, index_mapping.u] = slant_range * torch.sin(elevation)

    return enu


def convert_wgs84_coordinates_to_local_enu(
    coordinates_to_transform: torch.Tensor,
    reference_point: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Transform WGS84 coordinates (latitude, longitude, altitude) to local east, north, and up (ENU) offsets.

    This function calculates the north and east offsets in meters of a coordinate from the reference point.
    It converts the latitude and longitude to radians, calculates the radius of curvature values,
    and then computes the offsets based on the differences between the coordinate and the reference point.
    Finally, it returns a tensor containing these offsets along with the altitude difference.

    Note that this implementation uses a local differential approximation (small-distance linearization),
    not a full ECEF->ENU transform. It is most accurate for coordinates near the reference point.

    Parameters
    ----------
    coordinates_to_transform : torch.Tensor
        The coordinates in latitude, longitude, altitude that are to be transformed.
        Shape is ``[number_of_coordinates, 3]``.
    reference_point : torch.Tensor
        The center of origin of the ENU coordinate system in WGS84 coordinates.
        Shape is ``[3]``.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The east offsets in meters, north offsets in meters, and the altitude differences from the reference point.
        Shape is ``[number_of_coordinates, 3]``.
    """
    device = get_device(device=device)

    # Ensure inputs are on the target device and use consistent dtype.
    coordinates_to_transform = coordinates_to_transform.to(device=device)
    reference_point = reference_point.to(device=device)

    transformed_coordinates = torch.zeros_like(
        coordinates_to_transform, dtype=torch.float32, device=device
    )

    # WGS84 ellipsoid constants
    wgs84_a = 6378137.0  # Major axis in meters
    wgs84_b = 6356752.314245  # Minor axis in meters
    wgs84_e2 = (wgs84_a**2 - wgs84_b**2) / wgs84_a**2  # Eccentricity^2

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
