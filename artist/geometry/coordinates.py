from typing import cast

import torch

from artist.field.solar_tower import (
    SolarTower,
    TowerTargetAreasCylindrical,
    TowerTargetAreasPlanar,
)
from artist.util import index_mapping
from artist.util.environment_setup import get_device


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


def bitmap_coordinates_to_target_coordinates(
    bitmap_coordinates: torch.Tensor,
    bitmap_resolution: torch.Tensor,
    solar_tower: SolarTower,
    target_area_indices: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Convert bitmap pixel coordinates to 4D homogeneous world coordinates on the target surface.

    For planar target areas the pixel coordinates are mapped linearly to target plane coordinates.
    For cylindrical target areas the pixel coordinates are mapped to cylindrical surface coordinates
    using the cylinder's radius, opening angle, height, axis, and normal.

    Bitmaps and the resolution are conceptually defined as: [W, H] # width, height
    Tensor memory layout follows PyTorch convention: [H, W] # height, width

    The bitmap is treated as a discrete image grid with resolution:
        - bitmap_resolution = [width, height]
    Pixel coordinates follow image indexing conventions:
        - bitmap_coordinates[..., e] ∈ [0, W-1]
        - bitmap_coordinates[..., u] ∈ [0, H-1]
    They are interpreted as centered pixels:
        - (e + 0.5) / W
        - (u + 0.5) / H
    This ensures each pixel represents its spatial cell center rather than its corner.

    The e-axis is intentionally flipped (0.5 - e_norm) to match the desired bitmap orientation.
    This means: increasing bitmap e → decreases world e.

    Parameters
    ----------
    bitmap_coordinates : torch.Tensor
        Pixel coordinates in the bitmap for each heliostat, as (e, u) pairs.
        Shape is ``[number_of_active_heliostats, 2]``.
    bitmap_resolution : torch.Tensor
        Resolution of the bitmap (width, height) in pixels.
        Shape is ``[2]``.
    solar_tower : SolarTower
        Solar tower containing all target area definitions (planar and cylindrical).
    target_area_indices : torch.Tensor
        Global target area index for each heliostat (planar indices first, cylindrical second).
        Shape is ``[number_of_active_heliostats]``.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        World coordinates on the target surface in homogeneous format.
        Shape is ``[number_of_active_heliostats, 4]``.
    """
    device = get_device(device=device)

    number_of_coordinates = target_area_indices.numel()
    target_coordinates = torch.zeros((number_of_coordinates, 4), device=device)
    target_coordinates[:, -1] = 1.0

    bitmap_width = bitmap_resolution[index_mapping.unbatched_bitmap_e]
    bitmap_height = bitmap_resolution[index_mapping.unbatched_bitmap_u]

    e_norm = (
        bitmap_coordinates[:, index_mapping.unbatched_bitmap_e] + 0.5
    ) / bitmap_width
    u_norm = (
        bitmap_coordinates[:, index_mapping.unbatched_bitmap_u] + 0.5
    ) / bitmap_height

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

        e_axis = torch.tensor([1.0, 0.0, 0.0], device=device).expand(
            planar_indices.numel(), 3
        )
        u_axis = torch.tensor([0.0, 0.0, 1.0], device=device).expand(
            planar_indices.numel(), 3
        )

        e_local = (0.5 - e_norm[planar_mask]) * dims[
            :, index_mapping.target_dimensions_width
        ]
        u_local = (0.5 - u_norm[planar_mask]) * dims[
            :, index_mapping.target_dimensions_height
        ]

        target_coordinates[planar_mask, :3] = (
            centers + e_local[:, None] * e_axis + u_local[:, None] * u_axis
        )

    if (~planar_mask).any():
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
