import torch
import torch.nn.functional as F

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
    plane_normals = target_areas.normal_vectors[target_area_indices]
    plane_centers = target_areas.centers[target_area_indices]

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

    absolute_intensities = rays.ray_magnitudes * relative_intensities

    absolute_intensities = absolute_intensities * front_facing_mask

    return intersections, absolute_intensities


def line_cylinder_intersections(
    rays: Rays,
    points_at_ray_origins: torch.Tensor,
    scenario: Scenario,
    epsilon: float = 1e-6,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Ray-tracing function:
    1. Transform rays into cylinder frame (rotation optional)
    2. Intersect cylinder
    3. Clip theta/z to receiver bounds (skip clipping for full cylinder)
    4. Map to module UV coordinates
    5. Splat onto module grid

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


    Step 1 — transform ray into cylinder frame
    Step 2 — ray-cylinder intersection
    Step 3 — check receiver patch bounds
    Step 4 — convert to receiver coordinates
    Example if the receiver is tilted by angle α around x:
    
    The tilt only affects: R_cyl
    Example if the receiver is tilted by angle α around x:
    R_cyl =
    [1      0        0]
    [0   cosα   -sinα]
    [0   sinα    cosα]
    This rotation is applied to the rays.

    transform rays → intersect canonical cylinder → unwrap
    """
    plant_coordinates = [
        50.913421122592574,
        6.387824755874856,
        87.0
    ]
    receiver_coordinates = {
      "center": [50.91341660151, 6.3878253047761, 142.22675],
      "receiver_outer_upper_left": [50.913427272183, 6.3878568569144, 144.805],
      "receiver_outer_upper_right": [50.9134277392519, 6.38779212125015, 144.82],
      "receiver_outer_lower_left": [50.9134054755624, 6.38785629153485, 139.596],
      "receiver_outer_lower_right": [50.9134057066037, 6.38779225067161, 139.592],
      "receiver_inner_lower_left": [50.9134065441443, 6.38785392584286, 139.86],
      "receiver_inner_lower_right": [50.9134066492965, 6.38779530140411, 139.862],
      "receiver_inner_upper_left": [50.9134264540107, 6.38785420535071, 144.592],
      "receiver_inner_upper_right": [50.9134267664737, 6.38779541198343, 144.593]
    }

    receiver_coordinates_enu = {}
    for key, coord in receiver_coordinates.items():
        wgs_tensor = torch.tensor([coord], dtype=torch.float64)
        plant_tensor = torch.tensor(plant_coordinates, dtype=torch.float64)
        enu_coord = utils.convert_3d_points_to_4d_format(utils.convert_wgs84_coordinates_to_local_enu(wgs_tensor, plant_tensor))
        receiver_coordinates_enu[key] = enu_coord.squeeze(0)
    
    cylinder_radius = 4.14
    tilt_angle = -25
    tilt_angle_radians = torch.deg2rad(torch.tensor(tilt_angle, device=device)).unsqueeze(0)
    opening_angle_cylinder_part = 60
    opening_angle_cylinder_part_radians = torch.deg2rad(torch.tensor(opening_angle_cylinder_part, device=device)).unsqueeze(0)
    arc_length_absorber = cylinder_radius * opening_angle_cylinder_part_radians
    rotation_matrix = utils.rotate_e(e=tilt_angle_radians)

    point_a = receiver_coordinates_enu["receiver_inner_lower_left"]
    point_b = receiver_coordinates_enu["receiver_inner_lower_right"]
    point_c = receiver_coordinates_enu["receiver_inner_upper_left"]
    point_d = receiver_coordinates_enu["receiver_inner_upper_right"]

    midpoint_lower = (point_a + point_b) / 2
    midpoint_upper = (point_c + point_d) / 2
    axis_cylinder = torch.nn.functional.normalize(torch.abs(midpoint_lower-midpoint_upper), dim=0)
    normal_vector = rotation_matrix @ torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)
    
    # Chord vector
    chord_lower = point_b - point_a
    chord_upper = point_d - point_c
    # Distance from midpoint to center
    distance_to_center_lower = torch.sqrt(cylinder_radius**2 - (torch.norm(chord_lower[:3])/2)**2)
    distance_to_center_upper = torch.sqrt(cylinder_radius**2 - (torch.norm(chord_upper[:3])/2)**2)
    center_lower = midpoint_lower + distance_to_center_lower * -normal_vector
    center_upper = midpoint_upper + distance_to_center_upper * -normal_vector
    
    center_cylinder = (center_lower + center_upper) / 2
    height_cylinder = torch.norm(center_lower-center_upper)
    
    theta_min = - opening_angle_cylinder_part_radians/2
    theta_max = opening_angle_cylinder_part_radians/2
    z_min = -height_cylinder/2
    z_max = +height_cylinder/2


    # In reality in the cylinder is tilted and its center is on top of the solar tower.
    # For raytracing to be efficient, calculating intersections with an axis aligned cylinder with the z-axis and center at the origin is a lot easier.
    # Rotating the cylinder makes the ray intersection equation harder to solve and gradients would be more at risk to be instable.
    # That is why we rotate the world around it instead.
    # translate ray origins by center of cylinder, then rotate 
    
    local_ray_origins = (
        points_at_ray_origins 
        @ utils.translate_enu(e=-center_cylinder[:, 0], n=-center_cylinder[:, 1], u=-center_cylinder[:, 2]).transpose(1,2) 
        @ rotation_matrix.transpose(1,2)
    ) # transpose the matrix because we need the inverse directions.
    local_ray_directions = rays.ray_directions @ rotation_matrix.transpose(1,2)

    local_ray_origins_x, local_ray_origins_y, local_ray_origins_z = local_ray_origins[:, None, :, 0], local_ray_origins[:, None, :, 1], local_ray_origins[:, None, :, 2]
    local_ray_directions_x, local_ray_directions_y, local_ray_directions_z = local_ray_directions[..., 0], local_ray_directions[..., 1], local_ray_directions[..., 2]

    # theta = torch.atan2(local_ray_origins_y, local_ray_origins_x)
    # r = torch.sqrt(local_ray_origins_x**2 + local_ray_origins_y**2)

    # Intersections.
    a = local_ray_directions_x ** 2 + local_ray_directions_y ** 2
    b = 2 * (local_ray_origins_x * local_ray_directions_x + local_ray_origins_y * local_ray_directions_y)
    c = local_ray_origins_x ** 2 + local_ray_origins_y ** 2 - cylinder_radius ** 2

    discriminant = b**2 - 4*a*c
    valid_mask = discriminant >= 0
    sqrt_disc = torch.sqrt(torch.clamp(discriminant, min=0.0))
    t1 = (-b - sqrt_disc) / (2 * a + epsilon)
    t2 = (-b + sqrt_disc) / (2 * a + epsilon)

    # For better numerical stability, you can use the “stable quadratic formula”:
    # q = -0.5 * (b + torch.sign(b) * sqrt_disc)
    # t1 = q / a_safe
    # t2 = c / q

    t1_valid = t1 > 0
    t2_valid = t2 > 0
    t = torch.where(
        t1_valid & t2_valid,
        torch.minimum(t1, t2),
        torch.where(
            t1_valid,
            t1,
            torch.where(t2_valid, t2, torch.zeros_like(t1))
        )
    )
    t = torch.where(valid_mask, t, torch.zeros_like(t))
    p_hit = local_ray_origins[:, None, :, :] + t[..., None] * local_ray_directions

    theta_hit = torch.atan2(p_hit[..., 1], p_hit[..., 0])
    z_hit = p_hit[..., 2]
    hit_mask = valid_mask & (t > 0)

    hit_mask &= (theta_min <= theta_hit) & (theta_hit <= theta_max)
    hit_mask &= (z_min <= z_hit) & (z_hit <= z_max)
                
    # def angle_in_range(theta, theta_min, theta_max):
    #     return (theta - theta_min) % (2 * torch.pi) <= (theta_max - theta_min)
    # hit_mask &= angle_in_range(theta_hit, theta_min, theta_max)

    u = (theta_hit - theta_min) / (theta_max - theta_min)
    v =  (z_hit - z_min) / (z_max - z_min)