from typing import cast

import torch

from artist.field.solar_tower import (
    SolarTower,
    TowerTargetAreasCylindrical,
    TowerTargetAreasPlanar,
)
from artist.util import indices
from artist.util.environment import get_device


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
    min_vals = torch.min(points, dim=indices.unbatched_tensor_values).values
    point_range = points - min_vals
    max_vals = torch.max(point_range + 2e-5, dim=indices.unbatched_tensor_values).values
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

    bitmap_width = bitmap_resolution[indices.unbatched_bitmap_e]
    bitmap_height = bitmap_resolution[indices.unbatched_bitmap_u]

    e_norm = (bitmap_coordinates[:, indices.unbatched_bitmap_e] + 0.5) / bitmap_width
    u_norm = (bitmap_coordinates[:, indices.unbatched_bitmap_u] + 0.5) / bitmap_height

    planar_mask = (
        target_area_indices
        < solar_tower.number_of_target_areas_per_type[indices.planar_target_areas]
    )

    if planar_mask.any():
        planar_indices = target_area_indices[planar_mask]

        planar = cast(
            TowerTargetAreasPlanar,
            solar_tower.target_areas[indices.planar_target_areas],
        )

        centers = planar.centers[planar_indices][:, :3]
        dims = planar.dimensions[planar_indices]

        e_axis = torch.tensor([1.0, 0.0, 0.0], device=device).expand(
            planar_indices.numel(), 3
        )
        u_axis = torch.tensor([0.0, 0.0, 1.0], device=device).expand(
            planar_indices.numel(), 3
        )

        e_local = (0.5 - e_norm[planar_mask]) * dims[:, indices.target_dimensions_width]
        u_local = (0.5 - u_norm[planar_mask]) * dims[
            :, indices.target_dimensions_height
        ]

        target_coordinates[planar_mask, :3] = (
            centers + e_local[:, None] * e_axis + u_local[:, None] * u_axis
        )

    if (~planar_mask).any():
        cylinder_indices = (
            target_area_indices[~planar_mask]
            - solar_tower.number_of_target_areas_per_type[indices.planar_target_areas]
        )
        cylindrical = cast(
            TowerTargetAreasCylindrical,
            solar_tower.target_areas[indices.cylindrical_target_areas],
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
        (azimuth.shape[indices.unbatched_tensor_values], 3), device=device
    )

    enu[:, indices.e] = r * torch.sin(azimuth)
    enu[:, indices.n] = -r * torch.cos(azimuth)  # South-oriented azimuth convention
    enu[:, indices.u] = slant_range * torch.sin(elevation)

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
    latitudes = torch.deg2rad(coordinates_to_transform[:, indices.latitude])
    longitudes = torch.deg2rad(coordinates_to_transform[:, indices.longitude])
    latitude_reference_point = torch.deg2rad(reference_point[indices.latitude])
    longitude_reference_point = torch.deg2rad(reference_point[indices.longitude])

    # Calculate meridional radius of curvature for the first latitude.
    sin_lat1 = torch.sin(latitudes)
    rn1 = wgs84_a / torch.sqrt(1 - wgs84_e2 * sin_lat1**2)

    # Calculate transverse radius of curvature for the first latitude.
    rm1 = (wgs84_a * (1 - wgs84_e2)) / ((1 - wgs84_e2 * sin_lat1**2) ** 1.5)

    # Calculate delta latitude and delta longitude in radians.
    dlat_rad = latitude_reference_point - latitudes
    dlon_rad = longitude_reference_point - longitudes

    # Calculate north and east offsets in meters.
    transformed_coordinates[:, indices.e] = -(dlon_rad * rn1 * torch.cos(latitudes))
    transformed_coordinates[:, indices.n] = -(dlat_rad * rm1)
    transformed_coordinates[:, indices.u] = (
        coordinates_to_transform[:, indices.altitude]
        - reference_point[indices.altitude]
    )

    return transformed_coordinates
