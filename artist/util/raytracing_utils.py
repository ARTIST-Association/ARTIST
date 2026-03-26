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
    bitmap_resolution: torch.Tensor = torch.tensor([256, 256]),
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
    bitmap_resolution = bitmap_resolution.to(device)

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

    intersection_distances = numerator / torch.clamp(angle_based_intensities, min=epsilon)
    intersections = (
        points_at_ray_origins[:, None, :, :]
        + rays.ray_directions * intersection_distances[:, :, :, None]
    )

    intensities = rays.ray_magnitudes * angle_based_intensities

    # Determine the E- and U-positions of the rays' intersections with the target areas' planes, scaled to the
    # bitmap resolutions. Here, we decide that the bottom left corner of the 2D bitmap is the origin of the flux
    # image that is computed. `target_intersections_e/u` contain the intersection coordinates in meters.
    # Rays that hit the actual target have intersection coordinates ranging from 0 to `target_area.plane_e/_u`.
    plane_dimensions = target_areas.dimensions[target_area_indices]
    plane_centers = target_areas.centers[target_area_indices]

    target_intersections_e = (
        intersections[..., index_mapping.e]
        + (plane_dimensions[:, 0] / 2)[:, None, None]
        - plane_centers[:, 0][:, None, None]
    )

    target_intersections_u = (
        intersections[..., index_mapping.u]
        + (plane_dimensions[:, 1] / 2)[:, None, None]
        - plane_centers[:, 2][:, None, None]
    )

    # Scale target intersection coordinates into bitmap space.
    # The resulting `bitmap_intersections_e/u` represent continuous coordinates
    # in pixel units.
    bitmap_intersections_e = (
        (target_intersections_e / plane_dimensions[:, 0, None, None] * (bitmap_resolution[0] - 1))
    )
    bitmap_intersections_u = (
        (target_intersections_u / plane_dimensions[:, 1, None, None] * (bitmap_resolution[1] - 1))
    )

    # Filter out rays that are out of bounds of the target plane dimensions. Previously an infinite plane was considered.
    # Also filter out rays that hit the backside of the target or rays that are parallel to the target.    
    intersection_indices_on_target = (
        (0 <= bitmap_intersections_e)
        & (bitmap_intersections_e <= bitmap_resolution[0] - 1)
        & (0 <= bitmap_intersections_u)
        & (bitmap_intersections_u <= bitmap_resolution[1] - 1)
    )

    bitmap_intersections_e = bitmap_intersections_e * intersection_indices_on_target * front_facing_mask
    bitmap_intersections_u = bitmap_intersections_u * intersection_indices_on_target * front_facing_mask
    intersection_distances = intersection_distances * intersection_indices_on_target * front_facing_mask
    intensities = intensities * intersection_indices_on_target * front_facing_mask

    # The column indices need to be flipped because the more intuitive way to look at flux prediction
    # bitmaps, is to imagine oneself to stand in the heliostat field looking at the receiver.
    # This means that we look at the backside of the flux images. This corresponds to a flip of left and right,
    # i.e., subtracting the intersections from the total E-resolution to flip left and right.
    bitmap_intersections_e = (bitmap_resolution[0] - 1) - bitmap_intersections_e

    return bitmap_intersections_e, bitmap_intersections_u, intersection_distances, intensities 

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

    # Receiver definition.
    cylinder_axes = target_areas.axes[target_area_indices][:, :3]
    cylinder_normals = target_areas.normals[target_area_indices][:, :3]
    cylinder_centers = target_areas.centers[target_area_indices][:, :3]
    radii = target_areas.radii[target_area_indices]
    heights = target_areas.heights[target_area_indices]
    opening_angles = target_areas.opening_angles[target_area_indices]

    # Build local cylinder frame.
    u = torch.cross(cylinder_normals, cylinder_axes, dim=-1)
    rotations = torch.stack([ u, cylinder_normals, cylinder_axes], dim=1)

    # Transform rays into local cylinder frame.
    origins_local = ((origins - cylinder_centers[:, None, :]) @ rotations.transpose(1, 2))[:, None, :, :]
    directions_local = directions @ rotations.transpose(1, 2)[:, None, :, :]

    # Cylinder intersection (aligned with z-axis).
    ox, oy = origins_local[:, :, :, 0], origins_local[:, :, :, 1]
    dx, dy = directions_local[:, :, :, 0], directions_local[:, :, :, 1]

    a = dx**2 + dy**2
    b = 2 * (ox*dx + oy*dy)
    c = (ox**2 + oy**2 - radii.view(-1,1,1)**2).repeat(1, dx.shape[1], 1)

    discriminant = b**2 - 4*a*c
    
    # If a ray does not hit the infinite cylinder at all, the discriminant is negative and the square root cannot be computed,
    # these rays are invalid.
    mask_infinite_cylinder_hits = (discriminant >= 0) & (torch.abs(a) > 1e-8)

    # If there are no intersections, return empty coordinates, intensities and distances.
    if not torch.any(mask_infinite_cylinder_hits):
        empty_tensor = torch.zeros((directions.shape[0], directions.shape[1], directions.shape[2]), device=device), torch.zeros((directions.shape[0], directions.shape[1], directions.shape[2]), device=device), torch.zeros((directions.shape[0], directions.shape[1], directions.shape[2]), device=device), torch.zeros((directions.shape[0], directions.shape[1], directions.shape[2]), device=device),
        return empty_tensor, empty_tensor, empty_tensor, empty_tensor

    sqrt_discriminant = torch.sqrt(discriminant * mask_infinite_cylinder_hits)

    # The square root has two solutions, the minimum of the positive solutions per ray is the intersection we need.
    distance_candidates = torch.zeros((directions.shape[0], directions.shape[1], directions.shape[2], 2), device=device)
    distance_candidates[:, :, :, 0] = (-b - sqrt_discriminant) / (2 * a)
    distance_candidates[:, :, :, 1] = (-b + sqrt_discriminant) / (2 * a)

    # All invalid intersection distances are set to zero.
    intersection_distances, _ = torch.min(torch.clamp(distance_candidates, min=0.0), dim=-1)
    valid_distances = intersection_distances > 0
    
    if (intersection_distances == 0.0).all():
        empty_tensor = torch.zeros((directions.shape[0], directions.shape[1], directions.shape[2]), device=device), torch.zeros((directions.shape[0], directions.shape[1], directions.shape[2]), device=device), torch.zeros((directions.shape[0], directions.shape[1], directions.shape[2]), device=device), torch.zeros((directions.shape[0], directions.shape[1], directions.shape[2]), device=device),
        return empty_tensor, empty_tensor, empty_tensor, empty_tensor

    # Intersection points (local cylinder frame).
    intersections = origins_local + intersection_distances[:, :, :, None] * directions_local
    x, y, z = intersections[:, :, :, 0], intersections[:, :, :, 1], intersections[:, :, :, 2]

    # Cylinder normals (local frame).
    normals_local = torch.stack([x, y, torch.zeros_like(x)], dim=-1)
    normals_local = normals_local / torch.norm(normals_local, dim=-1, keepdim=True)

    # Lambert cosine law for ray magnitudes.
    angle_based_intensities = (-directions_local * normals_local).sum(dim=-1).clamp(min=0.0)

    # Height and angle of intersections.
    # We want all coordinates of the cylinder intersections to be positive for the bitmap calculation.
    # Initially the cylinder center is defined as the center of mass (halfway between top and bottom, on the cylinder axis.), therefore
    # z-values range from negative half cylinder height to positive half the cylinder height, therefore we add half the cylinder height 
    # to all z-values.
    z = z + heights.view(-1, 1, 1) / 2
    # Initially angles are defined 0° towards positive east axis, we want to define 0° as where the normal vector points towards. 
    angles = torch.atan2(y, x) - (torch.atan2(cylinder_normals[:, 1].view(-1,1,1), cylinder_normals[:, 0].view(-1,1,1)) - (opening_angles.view(-1,1,1) / 2)) 
    
    intersections_on_target = (
        (z >= 0) & (z <= heights.view(-1,1,1))
        & (angles >= 0) & (angles <= opening_angles.view(-1,1,1))
    )

    bitmap_intersections_u = z / heights.view(-1,1,1) * (bitmap_resolution[1]-1)
    bitmap_intersections_e = angles / opening_angles.view(-1,1,1) * (bitmap_resolution[0] - 1)

    # Filter out rays that are out of bounds of the target plane dimensions. Previously an infinite plane was considered.      
    bitmap_intersections_e = bitmap_intersections_e * intersections_on_target * valid_distances
    bitmap_intersections_u = bitmap_intersections_u * intersections_on_target * valid_distances
    intersection_distances = intersection_distances * intersections_on_target * valid_distances
    intensities = rays.ray_magnitudes * angle_based_intensities * intersections_on_target * valid_distances
    
    return bitmap_intersections_e, bitmap_intersections_u, intersection_distances, intensities, 
