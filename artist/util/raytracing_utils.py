import torch

from artist.field.tower_target_areas import TowerTargetAreas
from artist.scene.rays import Rays
from artist.util import index_mapping
from artist.util.environment_setup import get_device


def reflect(
    incident_ray_directions: torch.Tensor, reflection_surface_normals: torch.Tensor
) -> torch.Tensor:
    """
    Reflect incoming rays given the normals of reflective surfaces.

    Parameters
    ----------
    incident_ray_directions : torch.Tensor
        The direction of the incident ray as seen from the heliostats.
        Tensor of shape [number_of_active_heliostats, 1, 4].
    reflection_surface_normals : torch.Tensor
        The normals of the reflective surfaces.
        Tensor of shape [number_of_active_heliostats, number_of_combined_surface_normals_all_facets, 4]

    Returns
    -------
    torch.Tensor
        The reflected rays.
        Tensor of shape [number_of_active_heliostats, number_of_combined_surface_normals_all_facets, 4]

    """
    return (
        incident_ray_directions
        - 2
        * torch.sum(
            incident_ray_directions * reflection_surface_normals,
            dim=-1,
            keepdim=True,
        )
        * reflection_surface_normals
    )


def line_plane_intersections(
    rays: Rays,
    points_at_ray_origins: torch.Tensor,
    target_areas: TowerTargetAreas,
    target_area_mask: torch.Tensor | None = None,
    epsilon: float = 1e-6,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute line-plane intersections of the rays and the (receiver) planes.

    Parameters
    ----------
    rays : Rays
        The rays.
    points_at_ray_origins : torch.Tensor
        The surface points of the ray origins.
        Tensor of shape [number_of_active_heliostats, number_of_combined_surface_points_all_facets, 4].
    target_areas : TowerTargetAreas
        All possible tower target areas with their properties.
    target_area_mask : torch.Tensor | None
        The indices of target areas corresponding to each heliostat (default is None).
        If none are provided, the first target area of the scenario will be linked to all heliostats.
        Tensor of shape [number_of_active_heliostats].
    epsilon : float
        A small value corresponding to the upper limit (default is 1e-6).
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The intersections of the lines and planes.
        Tensor of shape [number_of_active_heliostats, number_of_rays, number_of_combined_surface_points_all_facets, 4].
    torch.Tensor
        The absolute intensities of the rays hitting the target planes.
        Tensor of shape [number_of_active_heliostats, number_of_rays, number_of_combined_surface_points_all_facets].
    """
    device = get_device(device=device)

    if target_area_mask is None:
        target_area_mask = torch.zeros(
            points_at_ray_origins.shape[index_mapping.heliostat_dimension],
            dtype=torch.int32,
            device=device,
        )
    plane_normals = target_areas.normal_vectors[target_area_mask]
    plane_centers = target_areas.centers[target_area_mask]

    # Use Lambert’s Cosine Law to calculate the relative intensities of the reflected rays on the planes.
    # The relative intensities are calculated by taking the dot product (matrix multiplication) of the planes'
    # unit normal vectors and the normalized ray-direction vectors, pointing from the planes to the sources.
    # This determines how much the rays align with the plane normals.
    relative_intensities = (-rays.ray_directions * plane_normals[:, None, None, :]).sum(
        dim=-1
    )

    front_facing_mask = relative_intensities > epsilon

    # Calculate the intersections on the plane of each ray.
    # First, calculate the projections of the ray origins onto the planes' normals.
    # This indicates how far the ray origins are from the planes (along the normal directions of the planes).
    # Next, calculate the scalar distances along the ray directions from the ray origins to the intersection points on the planes.
    # This indicates how far the intersection points are along the rays' directions.
    numerator = (
        (points_at_ray_origins - plane_centers[:, None, :]) * plane_normals[:, None, :]
    ).sum(dim=-1)[:, None, :]

    intersection_distances = torch.where(
        front_facing_mask,
        numerator / torch.clamp(relative_intensities, min=epsilon),
        torch.zeros_like(relative_intensities),
    )

    intersections = (
        points_at_ray_origins[:, None, :, :]
        + rays.ray_directions * intersection_distances[:, :, :, None]
    )

    absolute_intensities = (
        rays.ray_magnitudes * relative_intensities
    )

    absolute_intensities = absolute_intensities * front_facing_mask

    return intersections, absolute_intensities
