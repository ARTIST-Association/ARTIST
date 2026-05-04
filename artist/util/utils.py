from typing import cast

import torch

from artist.field.solar_tower import SolarTower
from artist.field.tower_target_areas_cylindrical import TowerTargetAreasCylindrical
from artist.field.tower_target_areas_planar import TowerTargetAreasPlanar
from artist.util import config_dictionary, index_mapping
from artist.util.environment_setup import get_device


def get_center_of_mass(
    bitmaps: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Calculate the coordinates of the flux density center of mass.

    Bitmaps and the resolution are conceptually defined as: [W, H] # width, height
    Tensor memory layout follows PyTorch convention: [H, W] # height, width

    First determine the indices of the bitmap center of mass.
    Next determine the position (coordinates) of the center of mass on the target.

    Returns (0.0, 0.0) for empty fluxes.

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
        Bitmap coordinates of the flux density centers of mass (e pixel, u pixel).
        Shape is ``[number_of_active_heliostats, 2]``.
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

    e_coords = torch.linspace(0, width_e - 1, width_e, device=device)
    u_coords = torch.linspace(0, height_u - 1, height_u, device=device)

    # meshgrid in (u, e) order because tensor is [u, e].
    u_grid, e_grid = torch.meshgrid(u_coords, e_coords, indexing="ij")
    u_grid = u_grid.expand(batch, -1, -1)
    e_grid = e_grid.expand(batch, -1, -1)

    # Center of mass.
    e_center_of_mass = (e_grid * normalized_bitmaps).sum(
        dim=(index_mapping.batched_bitmap_u, index_mapping.batched_bitmap_e)
    )
    u_center_of_mass = (u_grid * normalized_bitmaps).sum(
        dim=(index_mapping.batched_bitmap_u, index_mapping.batched_bitmap_e)
    )

    return torch.stack([e_center_of_mass, u_center_of_mass], dim=1)


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
