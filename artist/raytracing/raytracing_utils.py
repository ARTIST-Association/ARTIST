from typing import Optional

import torch

from artist.field.tower_target_areas import TowerTargetAreas
from artist.raytracing.rays import Rays


def reflect(
    incident_ray_directions: torch.Tensor, reflection_surface_normals: torch.Tensor
) -> torch.Tensor:
    """
    Reflect incoming rays given the normals of reflective surfaces.

    Parameters
    ----------
    incident_ray_directions : torch.Tensor
        The direction of the incident ray as seen from the heliostats.
    reflection_surface_normals : torch.Tensor
        The normals of the reflective surfaces.

    Returns
    -------
    torch.Tensor
        The reflected rays.
    """
    return (
        incident_ray_directions
        - 2
        * torch.sum(
            incident_ray_directions * reflection_surface_normals, dim=-1
        ).unsqueeze(-1)
        * reflection_surface_normals
    )


def line_plane_intersections(
    rays: Rays,
    points_at_ray_origins: torch.Tensor,
    target_areas: TowerTargetAreas,
    target_area_indices: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute line-plane intersections of the rays and the (receiver) planes.

    Parameters
    ----------
    rays : Rays
        The rays.
    points_at_ray_origins : torch.Tensor
        The surface points of the ray origins.
    target_areas : TowerTargetAreas
        All possible tower target areas with their properties.
    target_area_indices : Optional[torch.Tensor]
        The indices of target areas corresponding to each heliostat (default is None).
        If none are provided, the first target area of the scenario will be linked to all heliostats.
    epsilon : float
        A small value corresponding to the upper limit (default: 1e-6).

    Raises
    ------
    ValueError
        If there are no intersections on the front of the target plane.

    Returns
    -------
    torch.Tensor
        The intersections of the lines and planes.
    torch.Tensor
        The absolute intensities of the rays hitting the target planes.
    """
    if target_area_indices is None:
        target_area_indices = torch.zeros(points_at_ray_origins.shape[0])

    # Use Lambert’s Cosine Law to calculate the relative intensities of the reflected rays on the planes.
    # The relative intensities are calculated by taking the dot product (matrix multiplication) of the planes'
    # unit normal vectors and the normalized ray-direction vectors, pointing from the planes to the sources.
    # This determines how much the rays align with the plane normals.
    relative_intensities = (
        -rays.ray_directions
        * target_areas.normal_vectors[target_area_indices][:, None, None, :]
    ).sum(dim=-1)

    if (relative_intensities <= epsilon).all():
        raise ValueError("No ray intersections on the front of the target area planes.")

    # Calculate the intersections on the plane of each ray.
    # First, calculate the projections of the ray origins onto the planes' normals.
    # This indicates how far the ray origins are from the planes (along the normal directions of the planes).
    # Next, calculate the scalar distances along the ray directions from the ray origins to the intersection points on the planes.
    # This indicates how far the intersection points are along the rays' directions.
    intersection_distances = (
        (
            (
                points_at_ray_origins
                - target_areas.centers[target_area_indices][:, None, :]
            )
            * target_areas.normal_vectors[target_area_indices][:, None, :]
        ).sum(dim=-1)
    ).unsqueeze(1) / relative_intensities

    # Combine to get the intersections
    intersections = points_at_ray_origins.unsqueeze(
        1
    ) + rays.ray_directions * intersection_distances.unsqueeze(-1)

    # Calculate the absolute intensities of the rays hitting the target planes.
    # Use the inverse-square law for distance attenuations from the heliostats to target planes.
    distance_attenuations = (
        1
        / (
            torch.norm(
                (
                    points_at_ray_origins
                    - target_areas.centers[target_area_indices][:, None, :]
                ),
                dim=-1,
            )
            ** 2
        )
    ).unsqueeze(1)
    absolute_intensities = (
        rays.ray_magnitudes * relative_intensities * distance_attenuations
    )

    return intersections, absolute_intensities
