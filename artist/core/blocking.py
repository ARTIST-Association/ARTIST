import math
import warnings

import torch

from artist.util import config_dictionary
from artist.util.environment_setup import get_device


def create_blocking_primitives_rectangle(
    blocking_heliostats_surface_points: torch.Tensor,
    blocking_heliostats_active_surface_points: torch.Tensor,
    epsilon: float = 0.05,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a representation of a rectangular heliostat blocking plane, by interpolating its corner points.

    The blocking plane for rectangular heliostats is represented by its four
    corner points, and its normal vector. The corner points are indexed
    counterclockwise. The lower left corner point of a heliostat is indexed
    by 0, and so on. Overview of corner points and their indices:

    3 | 2
    -----
    0 | 1

    Assumptions:
    - The heliostat is rectangular.
    - The heliostat is oriented to the south if it is not aligned.

    Parameters
    ----------
    blocking_heliostats_surface_points : torch.Tensor
        The unaligned surface points of all heliostats that might block other heliostats.
        Tensor of shape [number_of_heliostats, number_of_combined_surface_points_all_facets, 4].
    blocking_heliostats_active_surface_points : torch.Tensor
        The aligned surface points of all heliostats that might block other heliostats.
        Tensor of shape [number_of_heliostats, number_of_combined_surface_points_all_facets, 4].
    epsilon : float
        A small value (default is 0.05).
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The blocking plane corners.
        Tensor of shape [number_of_heliostats, 4, 4].
    torch.Tensor
        The blocking plane spans in u and v direction.
        Tensor of shape [number_of_heliostats, 2, 4].
    torch.Tensor
        The blocking plane normals.
        Tensor of shape [number_of_heliostats, 3].
    """
    device = get_device(device=device)

    number_of_surfaces = blocking_heliostats_active_surface_points.shape[0]

    min_e = blocking_heliostats_surface_points[:, :, 0].min(dim=1).values
    max_e = blocking_heliostats_surface_points[:, :, 0].max(dim=1).values
    min_n = blocking_heliostats_surface_points[:, :, 1].min(dim=1).values
    max_n = blocking_heliostats_surface_points[:, :, 1].max(dim=1).values

    min_max_values = torch.stack(
        [
            torch.stack([min_e, min_n], dim=1),
            torch.stack([max_e, min_n], dim=1),
            torch.stack([max_e, max_n], dim=1),
            torch.stack([min_e, max_n], dim=1),
        ],
        dim=1,
    )

    surface_points_2d = blocking_heliostats_surface_points[:, :, :2]
    distances_to_surface_points = torch.abs(
        surface_points_2d[:, :, None, :] - min_max_values[:, None, :, :]
    )
    mask = (distances_to_surface_points < epsilon).all(-1)

    corner_points_indices = mask.float().argmax(dim=1)
    surface_indices = torch.arange(number_of_surfaces, device=device)[:, None]
    corners = blocking_heliostats_active_surface_points[
        surface_indices, corner_points_indices
    ]

    spans = torch.zeros((number_of_surfaces, 2, 4), device=device)
    spans[:, 0] = corners[:, 1] - corners[:, 0]
    spans[:, 1] = corners[:, 3] - corners[:, 0]

    plane_normals = torch.nn.functional.normalize(
        torch.cross(spans[:, 0, :3], spans[:, 1, :3], dim=-1), dim=-1
    )

    return corners, spans, plane_normals


def create_blocking_primitives_rectangles_by_index(
    blocking_heliostats_active_surface_points: torch.Tensor,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a representation of a rectangular heliostat blocking plane, by the known indices of its corner points.

    The blocking plane for rectangular heliostats is represented by its four
    corner points, and its normal vector. The corner points are indexed
    counterclockwise. The lower left corner point of a heliostat is indexed
    by 0, and so on. Overview of corner points and their indices:

    3 | 2
    -----
    0 | 1

    Assumptions:
    - The heliostat is rectangular in shape, each facet is also rectangular.
    - There are four facets ordered in two columns and two rows.
    - Each facet has an equal amount of surface points -> number_of_surface_points / 4
    - Each facet has an equal amount of points along its width and its height -> math.sqrt(number_of_surface_points / 4)

    Parameters
    ----------
    blocking_heliostats_active_surface_points : torch.Tensor
        The aligned surface points of all heliostats that might block other heliostats.
        Tensor of shape [number_of_heliostats, number_of_combined_surface_points_all_facets, 4].
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The blocking plane corners.
        Tensor of shape [number_of_heliostats, 4, 4].
    torch.Tensor
        The blocking plane spans in u and v direction.
        Tensor of shape [number_of_heliostats, 2, 4].
    torch.Tensor
        The blocking plane normals.
        Tensor of shape [number_of_heliostats, 3].
    """
    device = get_device(device=device)

    number_of_surfaces, number_of_surface_points, _ = (
        blocking_heliostats_active_surface_points.shape
    )

    corners = torch.zeros((number_of_surfaces, 4, 4), device=device)

    # Lower left.
    corners[:, 0] = blocking_heliostats_active_surface_points[
        :, int(number_of_surface_points / 2)
    ]
    # Lower right.
    corners[:, 3] = blocking_heliostats_active_surface_points[
        :, int(number_of_surface_points - math.sqrt(number_of_surface_points / 4))
    ]
    # Upper right.
    corners[:, 2] = blocking_heliostats_active_surface_points[
        :, int((number_of_surface_points / 2) - 1)
    ]
    # Upper left.
    corners[:, 1] = blocking_heliostats_active_surface_points[
        :, int(math.sqrt(number_of_surface_points / 4) - 1)
    ]

    spans = torch.zeros((number_of_surfaces, 2, 4), device=device)
    spans[:, 0] = corners[:, 1] - corners[:, 0]
    spans[:, 1] = corners[:, 3] - corners[:, 0]

    plane_normals = torch.nn.functional.normalize(
        torch.cross(spans[:, 0, :3], spans[:, 1, :3], dim=-1), dim=-1
    )

    return corners, spans, plane_normals


def soft_ray_blocking_mask(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    blocking_primitives_corners: torch.Tensor,
    blocking_primitives_spans: torch.Tensor,
    blocking_primitives_normals: torch.Tensor,
    distances_to_target: torch.Tensor,
    epsilon: float = 1e-6,
    softness: float = 50.0,
) -> torch.Tensor:
    r"""
    Compute a mask indicating which rays are blocked, using a soft, differentiable approach.

    Calculate ray plane intersections and the distances of the intersection from the ray origin.
    Depending on the intersections and the distances, rays are blocked if they cannot reach the target.
    The blocking is made differentiable by using sigmoid functions to approximate binary transitions
    with soft boundaries.
    For each ray and each blocking plane the intersection point and distance is computed by solving the
    plane equation:

    .. math::

        (\mathbf{p} - \mathbf{p_0}) \cdot \mathbf{n} = 0

        \mathbf{p} = \mathbf{l_0} + \mathbf{l} d

        ((\mathbf{l_0} + \mathbf{l} d) - \mathbf{p_0}) \cdot \mathbf{n} = 0

        d = \frac{(\mathbf{p_0}-\mathbf{l_0})\cdot \mathbf{n}}{\mathbf{l}\cdot \mathbf{n}}

        \mathbf{p_intersection} = \mathbf{l_0} + \mathbf{l}d

    where :math:`\mathbf{p}` are the points on the plane (ray_origins), :math:`\mathbf{p_0}` is a single point on the plane
    (corner_0), :math:`\mathbf{n}` is the normal vector of the plane (blocking_planes_normals), :math:`\mathbf{l}` is the unit
    vector describing the direction of the line (ray_directions), :math:`\mathbf{l_0}` is a point on the line (ray_origins),
    :math:`d` is the distance from the ray origin to the point of intersection.
    In the final output of this method values near 0 mean no blocking and values near 1 mean full blocking (there is at least
    one blocking primitive in front of the heliostat).

    Parameters
    ----------
    ray_origins : torch.Tensor
        The origin points of the rays, i.e. the surface points.
        Tensor of shape [number_of_heliostats, number_of_combined_surface_points_all_facets, 4].
    ray_directions : torch.Tensor
        The ray directions.
        Tensor of shape [number_of_heliostats, number_of_rays, number_of_combined_surface_normals_all_facets, 4].
    blocking_primitives_corners : torch.Tensor
        The blocking primitives corner points.
        Tensor of shape [number_of_blocking_primitives, 4, 4].
    blocking_primitives_spans: torch.Tensor
        The blocking primitives spans in u and v direction.
        Tensor of shape [number_of_blocking_primitives, 2, 4].
    blocking_primitives_normals : torch.Tensor
        The blocking primitives normals.
        Tensor of shape [number_of_blocking_primitives, 3]
    distances_to_target : torch.Tensor
        Tensor of shape [number_of_heliostats, number_of_rays, number_of_combined_surface_normals_all_facets].
    epsilon : float
        A small value (default is 1e-6).
    softness : float
        Controls how soft the sigmoid approximates the blocking (default is 50.0).

    Returns
    -------
    torch.Tensor
        A soft blocking mask.
        Tensor of shape [number_of_blocking_primitives, number_of_rays, number_of_combined_surface_points_all_facets].
    """
    ray_origins = ray_origins[:, None, :, None, :3]
    ray_directions = ray_directions[:, :, :, None, :3]

    corner_0 = blocking_primitives_corners[None, None, None, :, 0, :3]
    span_u = blocking_primitives_spans[None, None, None, :, 0, :3]
    span_v = blocking_primitives_spans[None, None, None, :, 1, :3]
    blocking_primitives_normals = blocking_primitives_normals[None, None, None, :, :3]

    denominator = torch.sum(ray_directions * blocking_primitives_normals, dim=-1)
    distances_to_blocking_planes = torch.sum(
        (corner_0 - ray_origins) * blocking_primitives_normals, dim=-1
    ) / (denominator + epsilon)
    blocking_planes_in_front_of_heliostats = torch.sigmoid(
        softness * (distances_to_blocking_planes - 1e-3)
    )

    intersection_points = (
        ray_origins + distances_to_blocking_planes[..., None] * ray_directions
    )
    intersection_offset_from_corner = intersection_points - corner_0

    # Compute point of intersection in local plane coordinates.
    span_u_squared_norm = torch.sum(span_u * span_u, dim=-1)
    span_v_squared_norm = torch.sum(span_v * span_v, dim=-1)
    span_uv_dot = torch.sum(span_u * span_v, dim=-1)
    offset_projection_u = torch.sum(intersection_offset_from_corner * span_u, dim=-1)
    offset_projection_v = torch.sum(intersection_offset_from_corner * span_v, dim=-1)
    det = (
        span_u_squared_norm * span_v_squared_norm - span_uv_dot * span_uv_dot + epsilon
    )
    u_coordinate_on_plane = (
        offset_projection_u * span_v_squared_norm - offset_projection_v * span_uv_dot
    ) / det
    v_coordinate_on_plane = (
        offset_projection_v * span_u_squared_norm - offset_projection_u * span_uv_dot
    ) / det

    # Mask values near 1 if intersection within parallelogram (plane), mask values near 0, if intersection outside plane boundaries.
    blocking_within_plane = (
        torch.sigmoid(softness * u_coordinate_on_plane)
        * torch.sigmoid(softness * (1 - u_coordinate_on_plane))
        * torch.sigmoid(softness * v_coordinate_on_plane)
        * torch.sigmoid(softness * (1 - v_coordinate_on_plane))
    )

    # Mask values near 1 if blocking plane in front of target, mask values near 0, if blocking plane behind target.
    blocking_planes_in_front_of_target = torch.sigmoid(
        softness * (distances_to_target.unsqueeze(-1) - distances_to_blocking_planes)
    )

    blocking_mask_per_plane = (
        blocking_within_plane
        * blocking_planes_in_front_of_heliostats
        * blocking_planes_in_front_of_target
    )
    blocked = 1 - torch.prod(1 - blocking_mask_per_plane, dim=-1)

    return blocked


def expand_bits(integers: torch.Tensor) -> torch.Tensor:
    """
    Expand the lower 10 bits of an integer into 30 bits by inserting 2 zero bits between each original bit.

    Parameters
    ----------
    integers : torch.Tensor
        Integer coordinates with values in [0, 1023] (10 bits).
        Tensor of shape [number_of_blocking_planes].

    Returns
    -------
    torch.Tensor
        Integer coordinates expanded from 10 bits to 30 bits.
        Tensor of shape [number_of_blocking_planes].
    """
    # Keep only the lower 10 bits.
    expanded_integers = integers & 0b1111111111
    # Spread and mask bits to achieve pattern with two 0 bits in between.
    expanded_integers = (
        expanded_integers | (expanded_integers << 16)
    ) & 0b111000000000000001111111
    expanded_integers = (
        expanded_integers | (expanded_integers << 8)
    ) & 0b111000001111000000001111
    expanded_integers = (
        expanded_integers | (expanded_integers << 4)
    ) & 0b110000110000110000110011
    expanded_integers = (
        expanded_integers | (expanded_integers << 2)
    ) & 0b1001001001001001001001001001

    return expanded_integers.to(torch.int64)


def morton_codes(
    coordinates: torch.Tensor, epsilon: float = 1e-6, device: torch.device | None = None
) -> torch.Tensor:
    """
    Map 3D points to a single integer value corresponding to its Morton Code.

    Spatially nearby points have similar Morton codes. Morton codes are also sometimes referred to as
    Z-order curve codes. They are computed by bit-interleaving the binary representations of the 3D
    x, y, z coordinates.
    The padding around the bounding boxes is necessary to avoid divisions by zero and integer
    overflows. The relative padding scales with the field size.

    Reference: Morton, G.M. (1966) A Computer Oriented Geodetic Data Base and a New Technique in File
    Sequencing. IBM Ltd., Ottawa.

    Parameters
    ----------
    coordinates : torch.Tensor
        The coordinates to transform into Morton codes.
        Tensor of shape [number_of_blocking_planes, 3].
    epsilon : float
        A small epsilon value (default is 1e-6).
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The converted integers in Morton code.
        Tensor of shape [number_of_blocking_planes].
    """
    device = get_device(device=device)

    # The 10 bits per axis should not be changed. 10 bits per axis means 1024 discrete positions along
    # each dimension and 30 bits in total. This is the maximum amount of bits per axis fitting into a
    # single 32-bit integer and is enough even for scenes with more than hundred thousand blocking planes.
    bits = 10

    # Compute bounding box around all coordinates.
    mins = coordinates.min(dim=0).values
    maxs = coordinates.max(dim=0).values
    padding = (maxs - mins) * epsilon + epsilon
    bounding_box_min = mins - padding
    bounding_box_max = maxs + padding

    # Normalize coordinates to [0,1 - epsilon).
    spans = bounding_box_max - bounding_box_min
    spans[spans == 0] = 1.0
    norm = (coordinates - bounding_box_min[None, :]) / spans[None, :]
    norm = norm.clamp(0.0, 1.0 - epsilon)

    # Determine number of discrete positions along each axis (1024).
    scale = float(1 << bits)

    # Scale normalized coordinates to integer values from 0 to 1024.
    qi = (norm * scale).to(torch.int64)
    xi = qi[:, 0].to(torch.int64)
    yi = qi[:, 1].to(torch.int64)
    zi = qi[:, 2].to(torch.int64)

    # Prepare the interleaving.
    # Spread 10 bits into 30 bits with 2 zero bits between each bit.
    xx = expand_bits(xi)
    # Spread with additional shift to the left for y.
    yy = expand_bits(yi) << 1
    # Spread with 2 additional shifts to the left for z.
    zz = expand_bits(zi) << 2

    code = (xx | yy | zz).to(torch.int64)

    return code


def most_significant_differing_bit(
    differing_bits: torch.Tensor, device: torch.device | None = None
) -> torch.Tensor:
    """
    Compute the most significant bit (MSB) indices.

    The MSB index is the position of the highest set bit in the binary representation
    of the integer value. The bit positions start at 0, which is the least significant bit.
    For x = 0, the MSB is undefined and -1 will be returned. This method uses a float-based
    log2, combined with the floor operation as a fast and safe MSB implementation. This works
    for positive integers only and and also only for Morton codes up to 30 bits, as the
    torch.log2() is safe for float32.

    Parameters
    ----------
    differing_bits : torch.Tensor
        Integer values.
        Tensor of shape [number_of_blocking_planes].
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        Most significant bits.
        Tensor of shape [number_of_blocking_planes].
    """
    device = get_device(device=device)

    differing_bits = differing_bits.to(torch.float32)

    nonzero_mask = differing_bits != 0
    most_significant_bits = torch.full_like(
        differing_bits, -1, dtype=torch.int64, device=device
    )

    if nonzero_mask.any():
        msb = torch.floor(torch.log2(differing_bits[nonzero_mask])).to(torch.int64)
        most_significant_bits[nonzero_mask] = msb

    return most_significant_bits


def longest_common_prefix(
    codes: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    total_bits: int = 30,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Compute the longest common prefix (LCP) between pairs of Morton codes.

    The longest common prefix (LCP) indicates how similar two Morton codes are and therefore also
    indicates how close (spatially) two blocking objects are. The LCP is the number of highest-order
    bits that are identical in two Morton codes.

    Parameters
    ----------
    codes : torch.Tensor
        Sorted Morton codes as int64.
        Tensor of shape [number_of_blocking_planes].
    i : torch.Tensor
        Lower indices selecting the first Morton codes for the comparison.
        Tensor of shape [number_of_blocking_planes].
    j : torch.Tensor
        Upper indices selecting the second Morton codes for the comparison.
        Tensor of shape [number_of_blocking_planes].
    total_bits : int
        Total number of bits used in the Morton codes (default is 30).
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The longest common prefixes in the range from 0 to total_bits.
        Tensor of shape [number_of_blocking_planes].
    """
    device = get_device(device=device)

    differing_bits = codes[i] ^ codes[j]
    most_significant_differing_bits = most_significant_differing_bit(
        differing_bits, device=device
    )
    longest_common_prefixes = torch.where(
        differing_bits == 0,
        torch.full_like(
            most_significant_differing_bits,
            total_bits,
            dtype=torch.int64,
            device=device,
        ),
        (total_bits - 1) - most_significant_differing_bits,
    )
    return longest_common_prefixes


def range_to_node_id(
    start_indices: torch.Tensor, end_indices: torch.Tensor, leaf_offset: int
) -> torch.Tensor:
    """
    Convert a range of sorted primitives into node indices.

    When the start index is equal to the end index there it will be a leaf node with the id: leaf offset + start index
    Otherwise it will be an internal node with the minimum of the start and end index as id.

    Parameters
    ----------
    start_indices : torch.Tensor
        Start indices of the node ranges.
        Tensor of shape [number_of_blocking_planes].
    end_indices : torch.Tensor
        End indices of the node ranges.
        Tensor of shape [number_of_blocking_planes].
    leaf_offset : int
        Offset index in the node array where leaf nodes start.

    Returns
    -------
    torch.Tensor
        Node indices corresponding to the given ranges.
    """
    leaf_node = (leaf_offset + start_indices).to(torch.int32)
    internal_node = torch.minimum(start_indices, end_indices).to(torch.int32)

    node_indices = torch.where(start_indices == end_indices, leaf_node, internal_node)

    return node_indices


@torch.no_grad()
def build_linear_bounding_volume_hierarchies(
    blocking_primitives_corners: torch.Tensor, device: torch.device | None = None
) -> dict[str, torch.Tensor]:
    """
    Build linear bounding volume heirachies (LBVHs).

    Reference: Tero Karras. Maximizing Parallelism in the Construction of BVHs, Octrees, and k‑d Trees.
    In Proceedings of the Fourth ACM SIGGRAPH / Eurographics Symposium on High‑Performance Graphics (HPG 2012)

    Parameters
    ----------
    blocking_primitives_corners : torch.Tensor
        Corner points of each blocking primitive.
        Tensor of shape [number_of_blocking_primitives, 4, 4].
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    dict[str, torch.Tensor]
        - left, right: Indices of the left and right child of each LBVH node (-1 if leave).
        - aabb_min, aabb_max: axis aligned bounding boxes.
        - is_leaf: boolean, indicating whether a node is a leaf node.
        - primitive_index: indicates which primitives are contained.
    """
    device = get_device(device=device)

    number_of_blocking_primitives = blocking_primitives_corners.shape[0]
    blocker_ids = torch.arange(number_of_blocking_primitives, device=device)

    if number_of_blocking_primitives == 0:
        return {
            config_dictionary.left_node: torch.empty(
                (0,), dtype=torch.int32, device=device
            ),
            config_dictionary.right_node: torch.empty(
                (0,), dtype=torch.int32, device=device
            ),
            config_dictionary.aabb_min: torch.empty((0, 3), device=device),
            config_dictionary.aabb_max: torch.empty((0, 3), device=device),
            config_dictionary.is_leaf: torch.empty(
                (0,), dtype=torch.bool, device=device
            ),
            config_dictionary.primitive_index: torch.empty(
                (0,), dtype=torch.int32, device=device
            ),
        }

    # Compute sorted Morton code representations for each blocking primitive.
    primitive_mins = blocking_primitives_corners.min(dim=1).values
    primitive_maxs = blocking_primitives_corners.max(dim=1).values
    centroids = blocking_primitives_corners.mean(dim=1)

    codes = morton_codes(coordinates=centroids, epsilon=1e-6, device=device)
    sorted_codes, sorted_primitive_indices = torch.sort(codes)

    # Analyse similarities between Morton codes and determine the direction to the more similar Morton codes, in the sorted array: -1 = to the left, +1 = to the right.
    # The similarity is evaluated by computing leading common prefix lengths for all neighboring pairs of Morton codes.
    if number_of_blocking_primitives > 1:
        lcp_right = longest_common_prefix(
            codes=sorted_codes,
            i=blocker_ids,
            j=torch.clamp(blocker_ids + 1, max=number_of_blocking_primitives - 1),
            device=device,
        )
        lcp_left = longest_common_prefix(
            codes=sorted_codes,
            i=torch.clamp(blocker_ids - 1, min=0),
            j=blocker_ids,
            device=device,
        )
        lcp_right[-1] = -1
        lcp_left[0] = -1
    else:
        lcp_right = torch.tensor([-1], dtype=torch.int64, device=device)
        lcp_left = torch.tensor([-1], dtype=torch.int64, device=device)

    direction_to_similar_codes = torch.where(
        lcp_right > lcp_left,
        torch.ones(number_of_blocking_primitives, dtype=torch.int64, device=device),
        -torch.ones(number_of_blocking_primitives, dtype=torch.int64, device=device),
    )

    # Find threshold (delta_min) for node expansion by determining how similar the next Morton code in the chosen direction is.
    # Find the range of blocking primitives that share a common prefix larger than delta_min.
    # Find the contiguous range of Morton codes that belong together.
    # In the exponential search (the step size doubles in each iteration), find the farthest index j along direction d[i] where LCP > delta_min[i].
    neighbor_indices = blocker_ids - direction_to_similar_codes
    mask_out_of_bounds = (neighbor_indices >= 0) & (
        neighbor_indices < number_of_blocking_primitives
    )
    neighbor_indices = torch.clamp(
        neighbor_indices, 0, number_of_blocking_primitives - 1
    )

    delta_min = longest_common_prefix(
        codes=sorted_codes, i=blocker_ids, j=neighbor_indices, device=device
    )
    delta_min = torch.where(
        mask_out_of_bounds, delta_min, torch.full_like(delta_min, -1, device=device)
    )

    max = (
        math.ceil(math.log2(number_of_blocking_primitives))
        if number_of_blocking_primitives > 1
        else 1
    )
    farthest_expansion = torch.zeros(
        number_of_blocking_primitives, dtype=torch.int64, device=device
    )

    for k in range(0, max + 1):
        step = 1 << k
        candidate_indices = blocker_ids + direction_to_similar_codes * (
            farthest_expansion + step
        )
        mask_out_of_bounds_candidates = (candidate_indices >= 0) & (
            candidate_indices < number_of_blocking_primitives
        )
        candidate_indices = torch.clamp(
            candidate_indices, 0, number_of_blocking_primitives - 1
        )
        candidates_lcp = longest_common_prefix(
            sorted_codes, blocker_ids, candidate_indices, device=device
        )
        mask = mask_out_of_bounds_candidates & (candidates_lcp > delta_min)
        farthest_expansion = torch.where(
            mask, farthest_expansion + step, farthest_expansion
        )

    farthest_index = blocker_ids + direction_to_similar_codes * farthest_expansion
    farthest_index = torch.clamp(farthest_index, 0, number_of_blocking_primitives - 1)

    # Construct binary radix tree.
    # The range [first[i], last[i]] corresponds to the spatial cluster of blocking primitives that share a common prefix in Morton code.
    # Compute splits to build LBVH tree, each internal node is assigned two children.
    min_index = torch.minimum(blocker_ids, farthest_index)
    max_index = torch.maximum(blocker_ids, farthest_index)
    split = min_index.clone()
    span = max_index - min_index
    max_span = span.max().item() if span.numel() > 0 else 0
    if max_span < 1:
        pass
    else:
        max_k = math.floor(math.log2(max_span)) if max_span > 0 else 0
        for k in range(max_k, -1, -1):
            step_k = 1 << k
            candidate_indices = split + step_k
            valid = candidate_indices < max_index
            candidates_indices = torch.clamp(
                candidate_indices, 0, number_of_blocking_primitives - 1
            )
            candidates_incremented_indices = torch.clamp(
                candidate_indices + 1, 0, number_of_blocking_primitives - 1
            )
            candidates_lcp = longest_common_prefix(
                codes=sorted_codes, i=min_index, j=candidates_indices, device=device
            )
            candidates_incremented_lcp = longest_common_prefix(
                codes=sorted_codes,
                i=min_index,
                j=candidates_incremented_indices,
                device=device,
            )
            mask = valid & (candidates_lcp > candidates_incremented_lcp)
            split = torch.where(mask, split + step_k, split)

    # LBVH:
    # left, right: Indices of the left and right child of each node (-1 if not set).
    # aabb_min, aabb_max: axis aligned bounding box of the node.
    # is_leaf: boolean, indicating whether a node is a leaf node.
    # primitive_index: indicates which primitive is contained, -1 for internal nodes.
    total_nodes = 2 * number_of_blocking_primitives - 1
    left = torch.full((total_nodes,), -1, dtype=torch.int32, device=device)
    right = torch.full((total_nodes,), -1, dtype=torch.int32, device=device)
    aabb_min = torch.zeros((total_nodes, 3), dtype=torch.float32, device=device)
    aabb_max = torch.zeros((total_nodes, 3), dtype=torch.float32, device=device)
    is_leaf = torch.zeros((total_nodes,), dtype=torch.bool, device=device)
    primitive_index = torch.full((total_nodes,), -1, dtype=torch.int32, device=device)

    # In the Karras LBVH approach the leaf nodes are stored at the end of the node array.
    leaf_offset = number_of_blocking_primitives - 1
    internal_count = number_of_blocking_primitives - 1
    internal_nodes_indices = torch.arange(
        0, internal_count, dtype=torch.int64, device=device
    )

    # Map the original primitive index via the sorted_primitive_indices.
    aabb_min[leaf_offset : leaf_offset + number_of_blocking_primitives] = (
        primitive_mins[sorted_primitive_indices]
    )
    aabb_max[leaf_offset : leaf_offset + number_of_blocking_primitives] = (
        primitive_maxs[sorted_primitive_indices]
    )
    is_leaf[leaf_offset : leaf_offset + number_of_blocking_primitives] = True
    primitive_index[leaf_offset : leaf_offset + number_of_blocking_primitives] = (
        sorted_primitive_indices.to(torch.int32)
    )

    # left child node id corresponds to range [first[i], split[i]].
    # right child node id corresponds to range [split[i]+1, last[i]].
    left_child_nodes = range_to_node_id(
        start_indices=min_index[:internal_count],
        end_indices=split[:internal_count],
        leaf_offset=leaf_offset,
    )
    right_child_nodes = range_to_node_id(
        start_indices=split[:internal_count] + 1,
        end_indices=max_index[:internal_count],
        leaf_offset=leaf_offset,
    )

    # Detect cycles and replace by leaves.
    left_child_ids = torch.where(
        left_child_nodes == internal_nodes_indices,
        leaf_offset + min_index[:internal_count],
        left_child_nodes,
    )
    right_child_ids = torch.where(
        right_child_nodes == internal_nodes_indices,
        leaf_offset + max_index[:internal_count],
        right_child_nodes,
    )
    left[internal_nodes_indices] = left_child_ids.to(dtype=torch.int32, device=device)
    right[internal_nodes_indices] = right_child_ids.to(dtype=torch.int32, device=device)
    is_leaf[internal_nodes_indices] = False

    # Compute axis aligned bounding boxes (AABB) for internal nodes by combining child boxes.
    # The Karras mapping ensures internal nodes form a DAG that can be evaluated in ascending order.
    nodes_with_complete_aabb = torch.zeros(
        internal_count, dtype=torch.bool, device=device
    )
    left_internal = left[:internal_count].to(dtype=torch.int64, device=device)
    right_internal = right[:internal_count].to(dtype=torch.int64, device=device)
    rounds = 0
    while not nodes_with_complete_aabb.all() and rounds < internal_count:
        left_is_internal = left_internal < leaf_offset
        internal_mask = (
            left_is_internal & (left_internal >= 0) & (left_internal < internal_count)
        )
        left_done = torch.ones_like(left_is_internal, dtype=torch.bool, device=device)
        left_done[internal_mask] = nodes_with_complete_aabb[
            left_internal[internal_mask]
        ]

        right_is_internal = right_internal < leaf_offset
        internal_mask = (
            right_is_internal
            & (right_internal >= 0)
            & (right_internal < internal_count)
        )
        right_done = torch.ones_like(right_is_internal, dtype=torch.bool, device=device)
        right_done[internal_mask] = nodes_with_complete_aabb[
            right_internal[internal_mask]
        ]

        nodes_to_be_computed_next = (~nodes_with_complete_aabb) & left_done & right_done
        if not nodes_to_be_computed_next.any():
            break

        next_nodes_indices = torch.nonzero(nodes_to_be_computed_next, as_tuple=True)[
            0
        ].to(device)
        left_index = left_internal[next_nodes_indices]
        right_index = right_internal[next_nodes_indices]

        mins = torch.minimum(aabb_min[left_index], aabb_min[right_index])
        maxs = torch.maximum(aabb_max[left_index], aabb_max[right_index])
        aabb_min[next_nodes_indices] = mins
        aabb_max[next_nodes_indices] = maxs
        nodes_with_complete_aabb[next_nodes_indices] = True
        rounds += 1

    # Slow fallback logic if some axis aligned bounding boxes have not been computed.
    if not nodes_with_complete_aabb.all():
        incomplete = torch.nonzero(~nodes_with_complete_aabb, as_tuple=True)[0]
        warnings.warn(
            f"LBVH AABB fallback computation (very slow): {incomplete.numel()} internal nodes did not receive AABBs via DAG propagation.",
            RuntimeWarning,
        )
        for node in incomplete.tolist():
            min = int(min_index[node].item())
            max = int(max_index[node].item())
            leaf_nodes_slice = (
                torch.arange(min, max + 1, device=device, dtype=torch.int64)
                + leaf_offset
            )
            aabb_min[node] = torch.min(aabb_min[leaf_nodes_slice], dim=0).values
            aabb_max[node] = torch.max(aabb_max[leaf_nodes_slice], dim=0).values

    return {
        config_dictionary.left_node: left,
        config_dictionary.right_node: right,
        config_dictionary.aabb_min: aabb_min,
        config_dictionary.aabb_max: aabb_max,
        config_dictionary.is_leaf: is_leaf,
        config_dictionary.primitive_index: primitive_index,
    }


def ray_aabb_intersect(
    ray_origins: torch.Tensor,
    inverse_ray_directions: torch.Tensor,
    aabb_min: torch.Tensor,
    aabb_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute intersection distances between rays and axis aligned bounding boxes (AABBs).

    This method uses the slab method and the inverse ray direction for more efficient computation.

    Parameters
    ----------
    ray_origins : torch.Tensor
        Ray origins.
        Tensor of shape [total_number_of_rays, 3].
    inverse_ray_directions : torch.Tensor
        Precomputed inverse ray directions.
        Tensor of shape [total_number_of_rays, 3].
    aabb_min : torch.Tensor
        Minimum corner points of the AABBs.
        Tensor of shape [total_number_of_rays, 3].
    aabb_max : torch.Tensor
        Maximum corner points of the AABBs.
        Tensor of shape [total_number_of_rays, 3].

    Returns
    -------
    entry_distance_to_aabb : torch.Tensor
        Entry distance along each ray to the AABBs.
        Tensor of shape [total_number_of_rays].
    exit_distance_to_aabb : torch.Tensor
        Exit distance along each ray to the AABBs.
        Tensor of shape [total_number_of_rays].
    """
    min_distance = (aabb_min - ray_origins) * inverse_ray_directions
    max_distance = (aabb_max - ray_origins) * inverse_ray_directions
    entry_distance_to_aabb = torch.minimum(min_distance, max_distance).amax(dim=-1)
    exit_distance_to_aabb = torch.maximum(min_distance, max_distance).amin(dim=-1)
    return entry_distance_to_aabb, exit_distance_to_aabb


@torch.no_grad()
def lbvh_filter_blocking_planes(
    points_at_ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    blocking_primitives_corners: torch.Tensor,
    ray_to_heliostat_mapping: torch.Tensor,
    max_stack_size: int = 128,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Apply the LBVH filter to filter out blocking planes that are not hit.

    Parameters
    ----------
    points_at_ray_origins : torch.Tensor
        Origin points of the rays, i.e. the surface points, expanded in the ray dimension.
        Tensor of shape [number_of_heliostats, number_of_rays, number_of_combined_surface_normals_all_facets, 3].
    ray_directions : torch.Tensor
        The ray directions.
        Tensor of shape [number_of_heliostats, number_of_rays, number_of_combined_surface_normals_all_facets, 3].
    blocking_primitives_corners : torch.Tensor
        The blocking primitives corner points.
        Tensor of shape [number_of_blocking_planes, 4, 3].
    ray_to_heliostat_mapping : torch.Tensor
        Mapping indicating which ray is reflected by which heliostat.
        Tensor of shape [total_number_of_rays].
    max_stack_size : int
        Maximum stack size for the depth-first LBVH traversal (default is 128).
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The indices of the blocking primitives that are hit.
        Tensor of shape [number_of_hit_blocking_planes].
    """
    device = get_device(device=device)

    lbvh = build_linear_bounding_volume_hierarchies(
        blocking_primitives_corners=blocking_primitives_corners, device=device
    )

    left = lbvh[config_dictionary.left_node]
    right = lbvh[config_dictionary.right_node]
    aabb_min = lbvh[config_dictionary.aabb_min]
    aabb_max = lbvh[config_dictionary.aabb_max]
    is_leaf = lbvh[config_dictionary.is_leaf]
    primitive_index = lbvh[config_dictionary.primitive_index]

    ray_origins = points_at_ray_origins.reshape(-1, 3)
    ray_directions = ray_directions.reshape(-1, 3)
    total_number_of_rays = ray_origins.shape[0]
    number_of_primitives = blocking_primitives_corners.shape[0]

    node_traversal_stack = torch.full(
        (total_number_of_rays, max_stack_size),
        -1,
        dtype=torch.int32,
        device=device,
    )
    node_traversal_stack[:, 0] = 0
    stack_pointer = torch.ones(total_number_of_rays, dtype=torch.int32, device=device)

    mask_hits_per_ray = torch.zeros(
        (total_number_of_rays, number_of_primitives), dtype=torch.bool, device=device
    )
    inverse_directions = 1.0 / (ray_directions + 1e-12)
    active_rays = torch.arange(total_number_of_rays, device=device)

    # LBVH Traversal (Depth-first, per-ray stack-based traversal of the LBVH).
    while active_rays.numel() > 0:
        top_index = stack_pointer[active_rays] - 1
        nodes = node_traversal_stack[active_rays, top_index]
        stack_pointer[active_rays] -= 1

        # Filter out rays that miss the AABBs.
        entry_distance_to_aabb, exit_distance_to_aabb = ray_aabb_intersect(
            ray_origins[active_rays],
            inverse_directions[active_rays],
            aabb_min[nodes],
            aabb_max[nodes],
        )
        mask_hit = (exit_distance_to_aabb >= entry_distance_to_aabb) & (
            exit_distance_to_aabb > 1e-6
        )

        if mask_hit.any():
            hit_rays = active_rays[mask_hit]
            hit_nodes = nodes[mask_hit]
            leaf_mask = is_leaf[hit_nodes]

            if leaf_mask.any():
                leaf_rays = hit_rays[leaf_mask]
                leaf_nodes = hit_nodes[leaf_mask]
                leaf_prims = primitive_index[leaf_nodes]
                mask_hits_per_ray[leaf_rays, leaf_prims] = True

            if (~leaf_mask).any():
                internal_rays = hit_rays[~leaf_mask]
                internal_nodes = hit_nodes[~leaf_mask]

                left_child_nodes = left[internal_nodes]
                right_child_nodes = right[internal_nodes]

                has_left = left_child_nodes >= 0
                has_right = right_child_nodes >= 0

                index_for_children = stack_pointer[internal_rays].clone()
                index_for_left_children = index_for_children.clone()
                index_for_left_children[~has_left] = -1
                index_for_right_children = index_for_children + has_left.to(torch.int32)
                index_for_right_children[~has_right] = -1

                if (
                    (index_for_left_children >= max_stack_size)
                    & (index_for_left_children != -1)
                ).any() or (
                    (index_for_right_children >= max_stack_size)
                    & (index_for_right_children != -1)
                ).any():
                    raise RuntimeError(
                        "Stack overflow in LBVH traversal, max_stack too small."
                    )

                if has_left.any():
                    rows = internal_rays[has_left]
                    cols = index_for_left_children[has_left]
                    values = left_child_nodes[has_left]
                    node_traversal_stack[rows, cols] = values
                    stack_pointer[rows] = cols + 1

                if has_right.any():
                    rows = internal_rays[has_right]
                    cols = index_for_right_children[has_right]
                    values = right_child_nodes[has_right]
                    node_traversal_stack[rows, cols] = values
                    stack_pointer[rows] = cols + 1

        active_rays = torch.nonzero(stack_pointer > 0, as_tuple=True)[0]

    # Remove self-hits (ray hits its the blocking primitive from which it originates).
    primitive_owner = torch.arange(number_of_primitives, device=device).view(1, -1)
    ray_owner = ray_to_heliostat_mapping.view(-1, 1)
    non_self = mask_hits_per_ray & (ray_owner != primitive_owner)
    filtered_blocking_primitive_indices = torch.nonzero(
        non_self.any(dim=0), as_tuple=True
    )[0]

    return filtered_blocking_primitive_indices
