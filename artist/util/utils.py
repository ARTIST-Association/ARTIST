import torch

from artist.util import index_mapping
from artist.util.environment_setup import get_device


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
