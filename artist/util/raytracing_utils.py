import math
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F

from artist.field.tower_target_areas_cylindrical import TowerTargetAreasCylindrical
from artist.field.tower_target_areas_planar import TowerTargetAreasPlanar
from artist.scenario.scenario import Scenario
from artist.scene.rays import Rays
from artist.util import index_mapping, utils
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
    target_areas: TowerTargetAreasPlanar,
    target_area_indices: torch.Tensor | None = None,
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
    target_area_indices : torch.Tensor | None
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

    if target_area_indices is None:
        target_area_indices = torch.zeros(
            points_at_ray_origins.shape[index_mapping.heliostat_dimension],
            dtype=torch.int32,
            device=device,
        )
    plane_normals = target_areas.normals[target_area_indices]
    plane_centers = target_areas.centers[target_area_indices]

    # Use Lambert’s Cosine Law to calculate the relative intensities of the reflected rays on the planes.
    # The relative intensities are calculated by taking the dot product (matrix multiplication) of the planes'
    # unit normal vectors and the normalized ray-direction vectors, pointing from the planes to the sources.
    # This determines how much the rays align with the plane normals.
    # A positive dot product means the rays hit the front the plane, a zero dot product means a perpendicular ray
    # to the normal, which is a parallel ray to the surface, which is invalid and a negative dot product means
    # the ray hits the backside of the target plane, this is used to define a front facing mask for the rays. 
    angle_based_intensities = (-rays.ray_directions * plane_normals[:, None, None, :]).sum(
        dim=-1
    )
    front_facing_mask = angle_based_intensities > epsilon

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
        numerator / torch.clamp(angle_based_intensities, min=epsilon),
        torch.tensor(float('inf'), device=device),
    )

    intersections = (
        points_at_ray_origins[:, None, :, :]
        + rays.ray_directions * intersection_distances[:, :, :, None]
    )

    intensities = torch.where(
        front_facing_mask,
        rays.ray_magnitudes * angle_based_intensities,
        torch.tensor(float('inf'), device=device),
    )
    
    return intersections, intersection_distances, intensities 

def line_cylinder_intersections(
    rays: Rays,
    points_at_ray_origins: torch.Tensor,
    target_areas: TowerTargetAreasCylindrical,
    target_area_indices: torch.Tensor | None = None,
    bitmap_resolution: torch.Tensor = torch.tensor([256, 256]),
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    world space
      ↓
    rotate rays into receiver frame
        ↓
    cylinder aligned with z
        ↓
    intersect cylinder
        ↓
    unwrap cylinder patch
        ↓
    2D receiver grid
    """
    device = get_device(device=device)
    bitmap_resolution = bitmap_resolution.to(device)

    origins = points_at_ray_origins[:, :, :3]
    directions = rays.ray_directions[:, :, :, :3]
    number_of_heliostats = origins.shape[0]

    # Receiver definition.
    cylinder_axes = target_areas.axes[target_area_indices][:, :3]
    cylinder_centers = target_areas.centers[target_area_indices][:, :3]
    radii = target_areas.radii[target_area_indices]
    heights = target_areas.heights[target_area_indices]
    opening_angles = target_areas.opening_angles[target_area_indices]

    # Build local cylinder frame.
    u = torch.tensor([1.0, 0.0, 0.0], device=device)[None, :].expand(number_of_heliostats, 3)
    v = torch.cross(cylinder_axes, u, dim=-1)
    rotations = torch.stack([ u, v, cylinder_axes], dim=1)

    # Transform rays into local cylinder frame.
    origins_local = ((origins - cylinder_centers[:, None, :]) @ rotations.transpose(1, 2))[:, None, :, :]
    directions_local = directions @ rotations.transpose(1, 2)[:, None, :, :]

    # Cylinder intersection (aligned with z-axis).
    ox, oy, _ = origins_local[:, :, :, 0], origins_local[:, :, :, 1], origins_local[:, :, :, 2]
    dx, dy, _ = directions_local[:, :, :, 0], directions_local[:, :, :, 1], directions_local[:, :, :, 2]

    a = dx**2 + dy**2
    b = 2 * (ox*dx + oy*dy)
    c = (ox**2 + oy**2 - radii.view(-1,1,1)**2).repeat(1, dx.shape[1], 1)

    discriminant = b**2 - 4*a*c
    valid = (discriminant >= 0) & (torch.abs(a) > 1e-8)

    # If there are no intersections, return empty coordinates, intensities and distances.
    if not torch.any(valid):
        empty_tensor = torch.zeros((directions.shape[0], directions.shape[1], directions.shape[2]), device=device), torch.zeros((directions.shape[0], directions.shape[1], directions.shape[2]), device=device), torch.zeros((directions.shape[0], directions.shape[1], directions.shape[2]), device=device), torch.zeros((directions.shape[0], directions.shape[1], directions.shape[2]), device=device),
        return empty_tensor, empty_tensor, empty_tensor, empty_tensor

    sqrt_discriminant = torch.sqrt(discriminant * valid)

    a_valid = a * valid
    b_valid = b * valid

    t1 = (-b_valid - sqrt_discriminant) / (2 * a_valid)
    t2 = (-b_valid + sqrt_discriminant) / (2 * a_valid)
    t_candidates = torch.stack([t1,t2],dim=-1)
    t_candidates = torch.where(t_candidates>0, t_candidates, torch.tensor(float('inf'),device=device))
    t, _ = torch.min(t_candidates, dim=-1)     

    # Only keep hits in front of ray origins.
    valid = torch.isfinite(t)
    intersection_distances = t * valid.float() 

    if t.numel() == 0:
        empty_tensor = torch.zeros((directions.shape[0], directions.shape[1], directions.shape[2]), device=device), torch.zeros((directions.shape[0], directions.shape[1], directions.shape[2]), device=device), torch.zeros((directions.shape[0], directions.shape[1], directions.shape[2]), device=device), torch.zeros((directions.shape[0], directions.shape[1], directions.shape[2]), device=device),
        return empty_tensor, empty_tensor, empty_tensor, empty_tensor

    # Intersection points (local cylinder frame).
    intersections = origins_local + intersection_distances.unsqueeze(-1) * directions_local
    x, y, z = intersections[:, :, :, 0], intersections[:, :, :, 1], intersections[:, :, :, 2]

    # Cylinder normals (local frame).
    normals_local = torch.stack([x, y, torch.zeros_like(x)], dim=-1)
    normals_local = normals_local / torch.norm(normals_local, dim=-1, keepdim=True)

    # Lambert cosine law for ray magnitudes.
    relative_intensities = (-directions_local * normals_local).sum(dim=-1).clamp(min=0.0)
    intensities = relative_intensities * rays.ray_magnitudes

    # Height and angle of intersections.
    # We want all coordinates of the cylinder intersections to be positive for the bitmap calculation.
    # Initially the cylinder center is defined as the center of mass (halfway between top and bottom, on the cylinder axis.), therefore
    # z-values range from negative half cylinder height to positive half the cylinder height, therefore we add half the cylinder height 
    # to all z-values.
    z = z + heights.view(-1, 1, 1) / 2
    # within_height = (z >= 0) & (z <= heights.view(-1,1,1))
    # Initially angles are defined 0° towards positive east axis, we want to define 0° as positive north minus half of the opening angle. 
    angles = torch.atan2(y, x) - torch.pi/2 + opening_angles.view(-1,1,1)/2
    # within_sector = (angles >= 0) & (angles <= opening_angles.view(-1,1,1))

    # valid_hits = within_height & within_sector
    # intersection_heights = z * valid_hits
    # intersection_angles = angles * valid_hits
    # intensities = intensities * valid_hits

    bitmap_intersection_u = z / heights.view(-1,1,1) * (bitmap_resolution[1]-1)
    bitmap_intersection_e = angles / opening_angles.view(-1,1,1) * (bitmap_resolution[0] - 1)

    return bitmap_intersection_e, bitmap_intersection_u, intensities, intersection_distances
