import torch

from artist.field.tower_target_areas import TowerTargetAreas
from artist.raytracing.rays import Rays


def reflect(
    incoming_ray_direction: torch.Tensor, reflection_surface_normals: torch.Tensor
) -> torch.Tensor:
    """
    Reflect incoming rays given the normals of reflective surfaces.

    Parameters
    ----------
    incoming_ray_direction : torch.Tensor
        The direction of the incident ray as seen from the heliostat.
    reflection_surface_normals : torch.Tensor
        The normals of the reflective surfaces.

    Returns
    -------
    torch.Tensor
        The reflected rays.
    """
    return (
        incoming_ray_direction
        - 2
        * torch.sum(
            incoming_ray_direction * reflection_surface_normals, dim=-1
        ).unsqueeze(-1)
        * reflection_surface_normals
    )


def line_plane_intersections(
    rays: Rays,
    target_area_indices: torch.Tensor,
    target_areas: TowerTargetAreas,
    points_at_ray_origin: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute line-plane intersections of the rays and the (receiver) plane.

    Parameters
    ----------
    rays : Rays
        The rays.
    plane_normal_vector : torch.Tensor
        The normal vector of the plane.
    plane_center : torch.Tensor
        The center of the plane.
    points_at_ray_origin : torch.Tensor
        The surface points of the ray origin.
    epsilon : float
        A small value corresponding to the upper limit (default: 1e-6).

    Raises
    ------
    ValueError
        If there are no intersections on the front of the target plane.

    Returns
    -------
    torch.Tensor
        The intersections of the lines and plane.
    torch.Tensor
        The absolute intensities of the rays hitting the target plane.
    """
    # Use Lambert’s Cosine Law to calculate the relative intensity of the reflected rays on a plane.
    # The relative intensity is calculated by taking the dot product (matrix multiplication) of the plane's
    # unit normal vector and the normalized ray-direction vectors, pointing from the plane to the source.
    # This determines how much the ray aligns with the plane normal.
    relative_intensities = (-rays.ray_directions * target_areas.normal_vectors[target_area_indices][:, None, None, :]).sum(dim=-1)

    if (relative_intensities <= epsilon).all():
        raise ValueError("No ray intersection on the front of the target area plane.")

    # Calculate the intersections on the plane of each ray.
    # First, calculate the projection of the ray origin onto the plane’s normal.
    # This indicates how far the ray origin is from the plane (along the normal direction of the plane).
    # Next, calculate the scalar distance along the ray direction from the ray origin to the intersection point on the plane.
    # This indicates how far the intersection point is along the ray's direction.
    intersection_distances = (
        ((points_at_ray_origin - target_areas.centers[target_area_indices][:, None, :]) * target_areas.normal_vectors[target_area_indices][:, None, :]).sum(dim=-1)
    ).unsqueeze(1) / relative_intensities

    # Combine to get the intersections
    intersections = points_at_ray_origin.unsqueeze(
        1
    ) + rays.ray_directions * intersection_distances.unsqueeze(-1)

    # Calculate the absolute intensities of the rays hitting the target plane.
    # Use inverse-square law for distance attenuation from heliostat to target plane.
    distance_attenuations = (
        1 / (torch.norm(((points_at_ray_origin - target_areas.centers[target_area_indices][:, None, :])), dim=-1) ** 2)
    ).unsqueeze(1)
    absolute_intensities = (
        rays.ray_magnitudes * relative_intensities * distance_attenuations
    )

    return intersections, absolute_intensities
