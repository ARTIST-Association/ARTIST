from typing import Any

import torch
from torch import Tensor

from artist.field.tower_target_areas_cylindrical import TowerTargetAreasCylindrical
from artist.field.tower_target_areas_planar import TowerTargetAreasPlanar
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
    target_areas: TowerTargetAreasPlanar,
    target_area_indices: torch.Tensor | None = None,
    bitmap_resolution: torch.Tensor = torch.tensor([256, 256]),
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Compute line-plane intersections of the rays and the (receiver) planes.

    Parameters
    ----------
    rays : Rays
        Ray container with directions and magnitudes, the directions must be normalized.
    points_at_ray_origins : torch.Tensor
        Origins of the rays, which coincide with the surface points of the heliostats.
        Tensor of shape [number_of_active_heliostats, number_of_combined_surface_points_all_facets, 4].
    target_areas : TowerTargetAreas
        All planar tower target areas.
    target_area_indices : torch.Tensor | None
        Indices of target areas corresponding to each heliostat (default is None).
        If none are provided, the first target area of the scenario will be linked to all heliostats.
        Tensor of shape [number_of_active_heliostats].
    bitmap_resolution : torch.Tensor | None
        The resulting bitmap's resolution.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The east component of the bitmap intersections.
        Shape is [number_of_active_heliostats, number_of_rays, number_of_combined_surface_points_all_facets].
    torch.Tensor
        The up component of the bitmap intersections.
        Shape is [number_of_active_heliostats, number_of_rays, number_of_combined_surface_points_all_facets].
    torch.Tensor
        The intersection distances.
        Shape is [number_of_active_heliostats, number_of_rays, number_of_combined_surface_points_all_facets].
    torch.Tensor
        The absolute intensities of the rays hitting the target planes.
        Shape is [number_of_active_heliostats, number_of_rays, number_of_combined_surface_points_all_facets].
    """
    device = get_device(device=device)
    bitmap_resolution = bitmap_resolution.to(device)

    if target_area_indices is None:
        target_area_indices = torch.zeros(
            points_at_ray_origins.shape[index_mapping.heliostat_dimension],
            dtype=torch.int32,
            device=device,
        )

    # Extract 3D data.
    ray_directions = rays.ray_directions[..., :3]
    ray_origins = points_at_ray_origins[..., :3]
    plane_normals = target_areas.normals[target_area_indices][..., :3]
    plane_centers = target_areas.centers[target_area_indices][..., :3]

    # Use Lambert’s Cosine Law to calculate the relative intensities of the reflected rays on the planes.
    # Compute the alignment of the rays with the plane normals using the dot product.
    # In our raytracing process, the plane normals point away from the planes, which is usually the opposite
    # direction of the rays.
    # Therefore, a ray hitting the front side of the plane forms an angle > 90° with the normal,
    # making the dot product negative.
    # - A negative dot product indicates the ray hits the front face of the plane (valid).
    # - A zero dot product indicates the ray is parallel to the plane (invalid).
    # - A positive dot product indicates the ray hits the back face of the plane (invalid).
    # The front-facing mask selects only rays hitting the front of the plane.
    angle_based_intensities = (ray_directions * plane_normals[:, None, None, :]).sum(
        dim=-1
    )
    front_facing_mask = angle_based_intensities < 0.0

    # Calculate the intersections on the plane of each ray.
    # First, calculate the projections of the ray origins onto the planes' normals.
    # This indicates how far the ray origins are from the planes (along the normal directions of the planes).
    # Next, calculate the scalar distances along the ray directions from the ray origins to the intersection
    # points on the planes. This indicates how far the intersection points are along the rays' directions.
    numerator = (
        (plane_centers[:, None, :] - ray_origins) * plane_normals[:, None, :]
    ).sum(dim=-1)[:, None, :]

    safe_denominator = torch.where(front_facing_mask, angle_based_intensities, 1.0)
    intersection_distances = (numerator / safe_denominator) * front_facing_mask

    intersections = (
        ray_origins[:, None, :, :]
        + ray_directions * intersection_distances[:, :, :, None]
    )

    # Flip the sign of the intensities, so that valid rays have a positive intensity.
    intensities = rays.ray_magnitudes * -angle_based_intensities

    # Determine the E- and U-positions of the rays' intersections with the target areas' planes, scaled to the
    # bitmap resolutions. Here, we decide that the bottom left corner of the 2D bitmap is the origin of the flux
    # image that is computed. `target_intersections_e/u` contain the intersection coordinates in meters.
    # Rays that hit the actual target have intersection coordinates ranging from 0 to `target_area.plane_e/_u`.
    plane_dimensions = target_areas.dimensions[target_area_indices]
    plane_centers = target_areas.centers[target_area_indices]

    target_intersections_e = (
        intersections[..., index_mapping.e]
        + (plane_dimensions[:, index_mapping.target_dimensions_width] / 2)[
            :, None, None
        ]
        - plane_centers[:, index_mapping.e][:, None, None]
    )

    target_intersections_u = (
        intersections[..., index_mapping.u]
        + (plane_dimensions[:, index_mapping.target_dimensions_height] / 2)[
            :, None, None
        ]
        - plane_centers[:, index_mapping.u][:, None, None]
    )

    # Scale target intersection coordinates into bitmap space.
    # The resulting `bitmap_intersections_e/u` represent continuous coordinates
    # in pixel units. As bilinear weights assume integer indices are at pixel centers, the
    # scaling uses `(bitmap_resolution[0] - 1)` and `(bitmap_resolution[1] - 1)` so that
    # continuous coordinates map correctly to pixel centers when discretized.
    bitmap_intersections_e = (
        target_intersections_e
        / plane_dimensions[:, index_mapping.target_dimensions_width, None, None]
        * (bitmap_resolution[index_mapping.unbatched_bitmap_e] - 1)
    )
    bitmap_intersections_u = (
        target_intersections_u
        / plane_dimensions[:, index_mapping.target_dimensions_height, None, None]
        * (bitmap_resolution[index_mapping.unbatched_bitmap_u] - 1)
    )

    # Filter out rays that are out of bounds of the target plane dimensions. Previously an infinite plane was considered.
    # Also filter out rays that hit the backside of the target or rays that are parallel to the target.
    valid_mask = (
        (0 <= bitmap_intersections_e)
        & (
            bitmap_intersections_e
            <= bitmap_resolution[index_mapping.unbatched_bitmap_e] - 1
        )
        & (0 <= bitmap_intersections_u)
        & (
            bitmap_intersections_u
            <= bitmap_resolution[index_mapping.unbatched_bitmap_u] - 1
        )
        & front_facing_mask
    )

    bitmap_intersections_e = bitmap_intersections_e * valid_mask
    bitmap_intersections_u = bitmap_intersections_u * valid_mask
    intersection_distances = intersection_distances * valid_mask
    intensities = intensities * valid_mask

    # The column indices need to be flipped because the more intuitive way to look at flux prediction
    # bitmaps is to imagine oneself to stand in the heliostat field looking at the receiver.
    # This means that we look at the backside of the flux images. This corresponds to a flip of left and right,
    # i.e., subtracting the intersections from the total E-resolution to flip left and right.
    bitmap_intersections_e = (
        bitmap_resolution[index_mapping.unbatched_bitmap_e] - 1
    ) - bitmap_intersections_e

    return (
        bitmap_intersections_e,
        bitmap_intersections_u,
        intersection_distances,
        intensities,
    )


def line_cylinder_intersections(
    rays: Rays,
    points_at_ray_origins: torch.Tensor,
    target_areas: TowerTargetAreasCylindrical,
    target_area_indices: torch.Tensor | None = None,
    bitmap_resolution: torch.Tensor = torch.tensor([256, 256]),
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor] | tuple[Any, Any, Any, Any]:
    """
    Compute ray intersections with cylindrical receiver target areas and map hits into bitmap coordinates.

    This routine transforms rays from world space into each target cylinder's local frame, computes
    intersections with the infinite cylinder side surface, filters intersections to the finite receiver
    patch (height + opening angle), and returns per-ray bitmap coordinates, intersection distances,
    and Lambert-weighted intensities.

    Pipeline:
    1. Select target area per heliostat (or default to index 0).
    2. Build local cylinder frame and transform ray origins/directions into that frame.
    3. Solve quadratic for intersections with the infinite cylinder (x^2 + y^2 = r^2).
    4. Select the smallest strictly positive intersection distance per ray.
    5. Compute local intersection points, surface normals, and cosine-based intensity factor.
    6. Filter to finite cylinder patch bounds:
       - height range [0, h]
       - angle range [0, opening_angle]
    7. Convert valid local coordinates to continuous bitmap coordinates.

    Parameters
    ----------
    rays : Rays
        Ray container with directions and magnitudes, the directions must be normalized.
    points_at_ray_origins : torch.Tensor
        Ray origins in world space.
        Shape is [number_of_active_heliostats, number_of_combined_surface_points_all_facets, 4].
    target_areas : TowerTargetAreasCylindrical
        Cylindrical receiver definitions (centers, axes, normals, radii, heights, opening angles).
    target_area_indices : torch.Tensor | None, optional
        Per-heliostat target-area indices. Shape is [number_of_active_heliostats].
        If None, all heliostats use target area 0.
    bitmap_resolution : torch.Tensor, optional
        Bitmap resolution [E_res, U_res]. Default: [256, 256].
    device : torch.device | None, optional
        Compute device. If None, auto-selected.

    Returns
    -------
    torch.Tensor
        Continuous E (horizontal/unwrapped-angle) bitmap coordinates in pixel units.
        Shape is [number_of_active_heliostats, number_of_rays, number_of_combined_surface_points_all_facets].
        Invalid rays are 0.
    torch.Tensor
        Continuous U (vertical/height) bitmap coordinates in pixel units.
        Shape is [number_of_active_heliostats, number_of_rays, number_of_combined_surface_points_all_facets].
        Invalid rays are 0.
    torch.Tensor
        Ray parameter distance t to the selected cylinder intersection.
        Shape is [number_of_active_heliostats, number_of_rays, number_of_combined_surface_points_all_facets].
        Invalid rays are 0.
    torch.Tensor
        Lambert-weighted hit intensities:
        ray_magnitudes * max(0, -dot(ray_dir_local, normal_local)).
        Shape is [number_of_active_heliostats, number_of_rays, number_of_combined_surface_points_all_facets].
        Invalid rays are 0.
    """
    device = get_device(device=device)
    bitmap_resolution = bitmap_resolution.to(device)

    if target_area_indices is None:
        target_area_indices = torch.zeros(
            points_at_ray_origins.shape[index_mapping.heliostat_dimension],
            dtype=torch.int32,
            device=device,
        )

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
    rotations = torch.stack([u, cylinder_normals, cylinder_axes], dim=1)

    # Transform rays into local cylinder frame.
    origins_local = (
        (origins - cylinder_centers[:, None, :]) @ rotations.transpose(1, 2)
    )[:, None, :, :]
    directions_local = directions @ rotations.transpose(1, 2)[:, None, :, :]

    # Cylinder intersection (aligned with z-axis).
    ox, oy = origins_local[:, :, :, 0], origins_local[:, :, :, 1]
    dx, dy = directions_local[:, :, :, 0], directions_local[:, :, :, 1]

    a = dx**2 + dy**2
    b = 2 * (ox * dx + oy * dy)
    c = (ox**2 + oy**2 - radii.view(-1, 1, 1) ** 2).repeat(1, dx.shape[1], 1)

    discriminant = b**2 - 4 * a * c

    # If a ray does not hit the infinite cylinder at all, the discriminant is negative and the square root cannot be computed,
    # these rays are invalid.
    mask_infinite_cylinder_hits = (discriminant >= 0) & (torch.abs(a) > 1e-8)

    # If there are no intersections, return empty coordinates, intensities and distances.
    if not torch.any(mask_infinite_cylinder_hits):
        empty_tensor = torch.zeros(
            (directions.shape[0], directions.shape[1], directions.shape[2]),
            device=device,
        )
        return empty_tensor, empty_tensor, empty_tensor, empty_tensor

    sqrt_discriminant = torch.sqrt(discriminant * mask_infinite_cylinder_hits + 1e-12)

    # The square root has two solutions, the minimum of the positive solutions per ray is the intersection we need.
    distance_candidates = torch.zeros(
        (directions.shape[0], directions.shape[1], directions.shape[2], 2),
        device=device,
    )
    distance_candidates[:, :, :, 0] = (-b - sqrt_discriminant) / (2 * a)
    distance_candidates[:, :, :, 1] = (-b + sqrt_discriminant) / (2 * a)

    # Keep only strictly positive solutions; invalidate others with +inf.
    distance_candidates = torch.where(
        distance_candidates > 0,
        distance_candidates,
        torch.full_like(distance_candidates, torch.inf),
    )
    intersection_distances, _ = torch.min(distance_candidates, dim=-1)
    valid_distances = (
        torch.isfinite(intersection_distances) & mask_infinite_cylinder_hits
    )
    intersection_distances = torch.where(
        valid_distances,
        intersection_distances,
        torch.zeros_like(intersection_distances),
    )

    if (intersection_distances == 0.0).all():
        empty_tensor = torch.zeros(
            (directions.shape[0], directions.shape[1], directions.shape[2]),
            device=device,
        )
        return empty_tensor, empty_tensor, empty_tensor, empty_tensor

    # Intersection points (local cylinder frame).
    intersections = (
        origins_local + intersection_distances[:, :, :, None] * directions_local
    )
    x, y, z = (
        intersections[:, :, :, 0],
        intersections[:, :, :, 1],
        intersections[:, :, :, 2],
    )

    # Cylinder normals (local frame).
    normals_local = torch.stack([x, y, torch.zeros_like(x)], dim=-1)
    normals_local = normals_local / torch.norm(normals_local, dim=-1, keepdim=True)

    # Lambert cosine law for ray magnitudes.
    angle_based_intensities = (
        (-directions_local * normals_local).sum(dim=-1).clamp(min=0.0)
    )

    # Height and angle of intersections.
    # We want all coordinates of the cylinder intersections to be positive for the bitmap calculation.
    # Initially the cylinder center is defined as the center of mass (halfway between top and bottom, on the cylinder axis.), therefore
    # z-values range from negative half cylinder height to positive half the cylinder height, therefore we add half the cylinder height
    # to all z-values.
    z = z + heights.view(-1, 1, 1) / 2
    # Initially angles are defined 0° towards positive east axis, we want to define 0° as where the normal vector points towards.
    angles = torch.atan2(y, x) - (
        torch.atan2(
            cylinder_normals[:, 1].view(-1, 1, 1), cylinder_normals[:, 0].view(-1, 1, 1)
        )
        - (opening_angles.view(-1, 1, 1) / 2)
    )

    intersections_on_target = (
        (z >= 0)
        & (z <= heights.view(-1, 1, 1))
        & (angles >= 0)
        & (angles <= opening_angles.view(-1, 1, 1))
    )

    bitmap_intersections_u = (
        z
        / heights.view(-1, 1, 1)
        * (bitmap_resolution[index_mapping.unbatched_bitmap_u] - 1)
    )
    bitmap_intersections_e = (
        angles
        / opening_angles.view(-1, 1, 1)
        * (bitmap_resolution[index_mapping.unbatched_bitmap_e] - 1)
    )

    # Filter out rays that are out of bounds of the target plane dimensions. Previously an infinite plane was considered.
    bitmap_intersections_e = (
        bitmap_intersections_e * intersections_on_target * valid_distances
    )
    bitmap_intersections_u = (
        bitmap_intersections_u * intersections_on_target * valid_distances
    )
    intersection_distances = (
        intersection_distances * intersections_on_target * valid_distances
    )
    intensities = (
        rays.ray_magnitudes
        * angle_based_intensities
        * intersections_on_target
        * valid_distances
    )

    return (
        bitmap_intersections_e,
        bitmap_intersections_u,
        intersection_distances,
        intensities,
    )
