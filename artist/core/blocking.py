import logging
import math

import torch

from artist.util import config_dictionary
from artist.util.environment_setup import get_device

log = logging.getLogger(__name__)
"""A logger for blocking."""


def create_blocking_primitives_rectangle(
    blocking_heliostats_surface_points: torch.Tensor,
    blocking_heliostats_active_surface_points: torch.Tensor,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct a rectangular blocking plane representation for heliostats by interpolating their corner points.

    Instead of keeping many surface samples, each heliostat is reduced to its blocking plane via:
    - its four corner points
    - two spanning vectors (rectangle axes)
    - the plane normal

    The corner points are indexed clockwise. The lower left corner point of a heliostat is indexed
    by 0, and so on. Overview of corner points and their indices::

        1 | 2
        -----
        0 | 3

    Assumptions:

    - The heliostat is rectangular.
    - The heliostat is oriented to the south if it is not aligned.

    Parameters
    ----------
    blocking_heliostats_surface_points : torch.Tensor
        The unaligned surface points of all heliostats that might block other heliostats.
        Shape is ``[number_of_heliostats, number_of_combined_surface_points_all_facets, 4]``.
    blocking_heliostats_active_surface_points : torch.Tensor
        The aligned surface points of all heliostats that might block other heliostats.
        Shape is ``[number_of_heliostats, number_of_combined_surface_points_all_facets, 4]``.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The blocking plane corners.
        Shape is ``[number_of_heliostats, 4, 4]``.
    torch.Tensor
        The blocking plane spans in u and v direction.
        Shape is ``[number_of_heliostats, 2, 4]``.
    torch.Tensor
        The blocking plane normals.
        Shape is ``[number_of_heliostats, 4]``.
    """
    device = get_device(device=device)

    number_of_surfaces = blocking_heliostats_active_surface_points.shape[0]

    # Determine bounding rectangle in EN space. The indices of the corner points
    # are determined from the unaligned heliostat surface points, while their actual
    # positions are extracted from the aligned surfaces later on.
    # First retrieve the minimum east and north coordinates of unaligned heliostat surface points.
    # Unaligned heliostats are oriented horizontally, their normals point straight upwards.
    min_e = blocking_heliostats_surface_points[:, :, 0].min(dim=1).values
    max_e = blocking_heliostats_surface_points[:, :, 0].max(dim=1).values
    min_n = blocking_heliostats_surface_points[:, :, 1].min(dim=1).values
    max_n = blocking_heliostats_surface_points[:, :, 1].max(dim=1).values

    # Combine the minimum east and north values to form the expected four rectangle corner point
    # coordinates.
    # min_e and min_n form the lower left corner of the ASCII-diagram in the docstring
    # indexed by 0, min_e and max_n forms the corner indexed by 1, etc.
    min_max_values = torch.stack(
        [
            torch.stack([min_e, min_n], dim=1),
            torch.stack([min_e, max_n], dim=1),
            torch.stack([max_e, max_n], dim=1),
            torch.stack([max_e, min_n], dim=1),
        ],
        dim=1,
    )

    # Find points in the unaligned surface points tensor that are closest to
    # the four expected rectangle corner point coordinates saved in ``min_max_values``.
    surface_points_2d = blocking_heliostats_surface_points[:, :, :2]
    # Compute distances between all real surface points and expected rectangle corners.
    distances_to_corner = torch.norm(
        surface_points_2d[:, :, None, :] - min_max_values[:, None, :, :], dim=-1
    )
    corner_points_indices = distances_to_corner.argmin(dim=1)
    surface_indices = torch.arange(number_of_surfaces, device=device)[:, None]
    # Extract corners from aligned heliostat surface points.
    corners = blocking_heliostats_active_surface_points[
        surface_indices, corner_points_indices
    ]

    # Compute rectangle spans and normals.
    spans = torch.zeros((number_of_surfaces, 2, 4), device=device)
    spans[:, 0] = corners[:, 1] - corners[:, 0]
    spans[:, 1] = corners[:, 3] - corners[:, 0]

    plane_normals = torch.cat(
        [
            torch.nn.functional.normalize(
                torch.cross(spans[:, 0, :3], spans[:, 1, :3], dim=-1), dim=-1
            ),
            torch.zeros((number_of_surfaces, 1), device=device),
        ],
        dim=-1,
    )

    return corners, spans, plane_normals


def create_blocking_primitives_rectangles_by_index(
    blocking_heliostats_active_surface_points: torch.Tensor,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct a rectangular blocking plane representation for heliostats by the known indices of their corner points.

    The blocking plane for rectangular heliostats is represented by its four
    corner points, and its normal vector. The corner points are indexed
    clockwise. The lower left corner point of a heliostat is indexed
    by 0, and so on. Overview of corner points and their indices::

        1 | 2
        -----
        0 | 3

    Assumptions:

    - The heliostat is rectangular, each facet is also rectangular.
    - There are four facets ordered in two columns and two rows.
    - Each facet has the same number of surface points -> ``number_of_surface_points / 4``
    - Each facet has the same number of points along its width and height -> ``math.sqrt(number_of_surface_points / 4)``
    - Surface points are arranged in a structured, grid-like order and indexed in row-major fashion,
      analogous to a 2D tensor, ensuring consistent traversal.

    Parameters
    ----------
    blocking_heliostats_active_surface_points : torch.Tensor
        The aligned surface points of all heliostats that might block other heliostats.
        Shape is ``[number_of_heliostats, number_of_combined_surface_points_all_facets, 4]``.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The blocking plane corners.
        Shape is ``[number_of_heliostats, 4, 4]``.
    torch.Tensor
        The blocking plane spans in u and v direction.
        Shape is ``[number_of_heliostats, 2, 4]``.
    torch.Tensor
        The blocking plane normals.
        Shape is ``[number_of_heliostats, 4]``.
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

    plane_normals = torch.cat(
        [
            torch.nn.functional.normalize(
                torch.cross(spans[:, 0, :3], spans[:, 1, :3], dim=-1), dim=-1
            ),
            torch.zeros((number_of_surfaces, 1), device=device),
        ],
        dim=-1,
    )

    return corners, spans, plane_normals


def soft_ray_blocking_mask(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    blocking_primitives_corners: torch.Tensor,
    blocking_primitives_spans: torch.Tensor,
    blocking_primitives_normals: torch.Tensor,
    epsilon: float = 1e-12,
    softness: float = 1000.0,
    alpha: float = 100.0,
    ray_origin_offset: float = 0.05,
) -> torch.Tensor:
    r"""
    Compute a mask indicating which rays are blocked, using a soft differentiable approach.

    Calculate ray-plane intersections and the distances of the intersection from the ray origin.
    Depending on the intersections and the distances, rays are blocked if they cannot reach the target.
    The blocking is made differentiable by using sigmoid functions to approximate binary transitions
    with soft boundaries.
    For each ray and each blocking plane, the intersection point and distance is computed by solving the
    plane equation:

    .. math::

        (\mathbf{p} - \mathbf{p_0}) \cdot \mathbf{n} = 0

        \mathbf{p} = \mathbf{l_0} + \mathbf{l} d

        ((\mathbf{l_0} + \mathbf{l} d) - \mathbf{p_0}) \cdot \mathbf{n} = 0

        d = \frac{(\mathbf{p_0}-\mathbf{l_0})\cdot \mathbf{n}}{\mathbf{l}\cdot \mathbf{n}}

        \mathbf{p_intersection} = \mathbf{l_0} + \mathbf{l}d

    where :math:`\mathbf{p}` are the points on the plane (`ray_origins`), :math:`\mathbf{p_0}` is a single point on the
    plane (`corner_0`), :math:`\mathbf{n}` is the normal vector of the plane (`blocking_planes_normals`),
    :math:`\mathbf{l}` is the unit vector describing the direction of the line (`ray_directions`),
    :math:`\mathbf{l_0}` is a point on the line (`ray_origins`), and :math:`d` is the distance from the ray origin to
    the point of intersection.
    In the final output of this method, values near 0 mean no blocking and values near 1 mean full blocking (there is
    at least one blocking primitive in front of the heliostat).

    Parameters
    ----------
    ray_origins : torch.Tensor
        The origin points of the rays, i.e., the surface points.
        Shape is ``[number_of_heliostats, number_of_combined_surface_points_all_facets, 4]``.
    ray_directions : torch.Tensor
        The ray directions.
        Shape is ``[number_of_heliostats, number_of_rays, number_of_combined_surface_normals_all_facets, 4]``.
    blocking_primitives_corners : torch.Tensor
        The blocking primitives corner points.
        Shape is ``[number_of_blocking_primitives, 4, 4]``.
    blocking_primitives_spans: torch.Tensor
        The blocking primitives spans in u and v direction.
        Shape is ``[number_of_blocking_primitives, 2, 4]``.
    blocking_primitives_normals : torch.Tensor
        The blocking primitives normals.
        Shape is ``[number_of_blocking_primitives, 4]``.
    epsilon : float
        A small value used to avoid division by zero in plane-ray intersection (default is 1e-12).
    softness : float
        Controls how sharply the sigmoid approximates the hard blocking boundary (default is 1000.0).
        Higher values produce a steeper, more binary transition.
    alpha : float
        Optical depth scale factor for Beer–Lambert accumulation across blocking primitives (default is 100.0).
    ray_origin_offset : float
        Shift the ray origins a slight distance away from the heliostat planes to avoid self intersections
        (default is 0.05). The distance is measured in meters, so the default offset pushes the ray origins
        5 cm along the ray direction.

    Returns
    -------
    torch.Tensor
        A soft blocking mask, where values near 0 indicate no blocking and values near 1 indicate full blocking.
        Shape is ``[number_of_blocking_primitives, number_of_rays, number_of_combined_surface_points_all_facets]``.
    """
    # Dimensions [#heliostats, #rays, #surface_points, #blocking_primitives, 3D coordinates].
    ray_origins = ray_origins[:, None, :, None, :3]
    ray_directions = ray_directions[:, :, :, None, :3]

    corner_0 = blocking_primitives_corners[None, None, None, :, 0, :3]
    span_u = blocking_primitives_spans[None, None, None, :, 0, :3]
    span_v = blocking_primitives_spans[None, None, None, :, 1, :3]
    blocking_primitives_normals = blocking_primitives_normals[None, None, None, :, :3]

    # Solve the plane equation.
    denominator = torch.sum(ray_directions * blocking_primitives_normals, dim=-1)
    denominator_safe = torch.where(
        denominator.abs() < epsilon,
        torch.where(denominator >= 0, epsilon, -epsilon),
        denominator,
    )
    distances_to_blocking_planes = (
        torch.sum((corner_0 - ray_origins) * blocking_primitives_normals, dim=-1)
        / denominator_safe
    )
    blocking_planes_in_front_of_heliostats = torch.sigmoid(
        softness * (distances_to_blocking_planes - ray_origin_offset)
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
    det = span_u_squared_norm * span_v_squared_norm - span_uv_dot * span_uv_dot
    det_safe = torch.where(det.abs() < epsilon, torch.sign(det) * epsilon, det)
    u_coordinate_on_plane = (
        offset_projection_u * span_v_squared_norm - offset_projection_v * span_uv_dot
    ) / det_safe
    v_coordinate_on_plane = (
        offset_projection_v * span_u_squared_norm - offset_projection_u * span_uv_dot
    ) / det_safe

    # Mask values are near 1 if intersection within parallelogram (plane),
    # mask values are near 0, if intersection outside plane boundaries.
    # Beer–Lambert accumulation models occlusion as exponential attenuation through a continuous density field.
    # Each blocking primitive contributes a non-negative soft density along the ray rather than a hard binary mask.
    # These contributions are summed to form an optical depth, representing total accumulated occlusion.
    # The final visibility is computed as an exponential decay of this optical depth, ensuring stable, differentiable aggregation.
    inside_u = torch.sigmoid(softness * u_coordinate_on_plane) * torch.sigmoid(
        softness * (1 - u_coordinate_on_plane)
    )
    inside_v = torch.sigmoid(softness * v_coordinate_on_plane) * torch.sigmoid(
        softness * (1 - v_coordinate_on_plane)
    )

    inside_plane = inside_u * inside_v

    sigma = inside_plane * blocking_planes_in_front_of_heliostats
    sigma = sigma.clamp(0.0, 1.0)

    optical_depth = alpha * torch.sum(sigma, dim=-1)
    transmittance = torch.exp(-optical_depth)
    blocked = 1.0 - transmittance

    return blocked


def expand_bits(integers: torch.Tensor) -> torch.Tensor:
    """
    Expand the lower ten bits of an integer into 30 bits by inserting two zero bits between each original bit.

    This method is safe and conceptualized for scenarios with up to 2^30 blocking planes represented by 30-bit Morton
    codes using torch.int32.

    Parameters
    ----------
    integers : torch.Tensor
        Integer coordinates with values in ``[0, 1023]`` (10 bits).
        Shape is ``[number_of_blocking_planes]``.

    Returns
    -------
    torch.Tensor
        Integer coordinates expanded from 10 bits to 30 bits.
        Shape is ``[number_of_blocking_planes]``.
    """
    # Validate inputs.
    if (integers < 0).any() or (integers > 1023).any():
        raise ValueError("Input integers must be in [0, 1023].")
    if integers.dtype is not torch.int32:
        raise TypeError("Input integers must have dtype torch.int32.")
    # Keep only the lower 10 bits.
    expanded_integers = integers & 0x000003FF
    # Spread and mask bits to achieve pattern with two 0 bits in between.
    masks = [0x030000FF, 0x0300F00F, 0x030C30C3, 0x09249249]
    shifts = [16, 8, 4, 2]
    for shift, mask in zip(shifts, masks):
        expanded_integers = (expanded_integers | (expanded_integers << shift)) & mask

    return expanded_integers


def morton_codes(coordinates: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Map 3D points to a single integer value corresponding to its Morton Code.

    Spatially nearby points have similar Morton codes. Morton codes are also sometimes referred to as
    Z-order curve codes. They are computed by bit-interleaving the binary representations of the 3D
    x, y, z coordinates.
    The bits are interleaved such that y bits are given the highest priority, then x, then z. This is
    derived from heliostat field layouts, where heliostats commonly share a similar up component and
    blocking is determined by the east and north component.

    This method is safe and conceptualized for scenarios with up to 2^30 blocking planes represented by 30-bit Morton codes using torch.int32.

    Reference: Morton, G.M. (1966) A Computer Oriented Geodetic Data Base and a New Technique in File
    Sequencing. IBM Ltd., Ottawa.

    Parameters
    ----------
    coordinates : torch.Tensor
        The coordinates to transform into Morton codes.
        Shape is ``[number_of_blocking_planes, 3]``.
    epsilon : float
        A small epsilon value to avoid division by zero (default is 1e-6).

    Returns
    -------
    torch.Tensor
        The converted integers in Morton code.
        Shape is ``[number_of_blocking_planes]``.
    """
    # The 10 bits per axis should not be changed. 10 bits per axis means 1024 discrete positions along
    # each dimension and 30 bits in total. This is the maximum amount of bits per axis fitting into a
    # single 32-bit integer and is enough even for scenes with more than hundred thousand blocking planes.
    bits = 10

    mins = coordinates.min(dim=0).values
    maxs = coordinates.max(dim=0).values
    shifted = coordinates - mins
    scale = (1 << bits) - 1
    scaled_coordinates = (shifted * (scale / ((maxs - mins).max() + epsilon))).to(
        torch.int32
    )

    # Prepare the interleaving.
    # Spread 10 bits into 30 bits with 2 zero bits between each bit.
    u = expand_bits(scaled_coordinates[:, 2])
    # Spread with additional shift to the left for e.
    e = expand_bits(scaled_coordinates[:, 0]) << 1
    # Spread with 2 additional shifts to the left for n.
    n = expand_bits(scaled_coordinates[:, 1]) << 2

    return n | e | u


def longest_common_prefix(
    codes: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Compute the longest common prefix (LCP) between pairs of Morton codes.

    The longest common prefix (LCP) indicates how similar two Morton codes are and therefore also
    indicates how close (spatially) two blocking objects are. The LCP is the number of highest-order
    bits that are identical in two Morton codes.

    This method is safe and conceptualized for scenarios with up to 2^30 blocking planes represented by 30-bit Morton codes using torch.int32.

    Parameters
    ----------
    codes : torch.Tensor
        Sorted Morton codes as int64.
        Shape is ``[number_of_blocking_planes]``.
    i : torch.Tensor
        Lower indices selecting the first Morton codes for the comparison.
        Shape is ``[number_of_blocking_planes]``.
    j : torch.Tensor
        Upper indices selecting the second Morton codes for the comparison.
        Shape is ``[number_of_blocking_planes]``.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The longest common prefixes in the range from 0 to ``total_bits``.
        Shape is ``[number_of_blocking_planes]``.
    """
    device = get_device(device=device)
    valid = (j >= 0) & (j < codes.shape[0])

    lcp = torch.full_like(i, -1, device=device)
    if valid.any():
        differing_bits = torch.zeros_like(i, device=device)
        differing_bits[valid] = codes[i[valid]] ^ codes[j[valid]]

        nonzero_mask = differing_bits != 0
        msb = torch.full_like(i, -1, device=device)
        if nonzero_mask.any():
            leading_zeros = torch.zeros_like(
                differing_bits[nonzero_mask], dtype=torch.int32, device=device
            )
            for shift in [16, 8, 4, 2, 1]:
                mask = differing_bits[nonzero_mask] >> (32 - shift) == 0
                leading_zeros = leading_zeros + shift * mask.to(torch.int32)
                differing_bits[nonzero_mask] = torch.where(
                    mask,
                    differing_bits[nonzero_mask] << shift,
                    differing_bits[nonzero_mask],
                )

            msb[nonzero_mask] = 31 - leading_zeros

        lcp[valid] = torch.where(differing_bits[valid] == 0, 30, (30 - 1) - msb[valid])

    return lcp


@torch.no_grad()
def build_linear_bounding_volume_hierarchies(
    blocking_primitives_corners: torch.Tensor, device: torch.device | None = None
) -> dict[str, torch.Tensor]:
    """
    Build linear bounding volume hierarchies (LBVHs).

    This method is safe and conceptualized for scenarios with up to 2^30 blocking planes represented by
    30-bit Morton codes using ``torch.int32``.

    Reference: Tero Karras. Maximizing Parallelism in the Construction of BVHs, Octrees, and k‑d Trees.
    In Proceedings of the Fourth ACM SIGGRAPH / Eurographics Symposium on High‑Performance Graphics (HPG 2012)

    Parameters
    ----------
    blocking_primitives_corners : torch.Tensor
        Corner points of each blocking primitive.
        Shape is ``[number_of_blocking_primitives, 4, 4]``.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    dict[str, torch.Tensor]
        - ``left``, ``right``: Indices of the left and right child of each LBVH node (-1 if leaf).
        - ``aabb_min``, ``aabb_max``: Axis-aligned bounding boxes.
        - ``is_leaf``: Boolean, indicating whether a node is a leaf node.
        - ``primitive_index``: Indicates which primitives are contained.
    """
    device = get_device(device=device)

    number_of_blocking_primitives = blocking_primitives_corners.shape[0]
    blocker_ids = torch.arange(
        number_of_blocking_primitives, dtype=torch.int32, device=device
    )

    if number_of_blocking_primitives == 0:
        log.warning(
            "No blocking primitives provided, returning empty tensors for the linear bounding volume hierarchies."
        )
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

    # Compute sorted Morton code representations for each blocking primitive based on the centroid of the blocking planes.
    blocking_primitives_corners = blocking_primitives_corners[..., :3]
    primitive_mins = blocking_primitives_corners.min(dim=1).values
    primitive_maxs = blocking_primitives_corners.max(dim=1).values
    centroids = blocking_primitives_corners.mean(dim=1)

    codes = morton_codes(coordinates=centroids, epsilon=1e-6)
    sorted_codes, sorted_primitive_indices = torch.sort(codes)

    # Analyze similarities between Morton codes and determine the direction to the more similar Morton codes,
    # in the sorted array: -1 = to the left, +1 = to the right. The similarity is evaluated by computing leading
    # common prefix lengths for all neighboring pairs of Morton codes.
    if number_of_blocking_primitives > 1:
        lcp_right = longest_common_prefix(
            codes=sorted_codes,
            i=blocker_ids,
            j=blocker_ids + 1,
            device=device,
        )
        lcp_left = longest_common_prefix(
            codes=sorted_codes,
            i=blocker_ids,
            j=blocker_ids - 1,
            device=device,
        )
    else:
        lcp_right = torch.tensor([-1], device=device)
        lcp_left = torch.tensor([-1], device=device)

    direction_to_similar_codes = (lcp_right > lcp_left).to(torch.int32) * 2 - 1

    # Find the contiguous range of Morton codes that belong together.
    # Find threshold (delta_min) for node expansion by determining how different the next Morton code in the direction of the less similar neighbor is.
    # Find the range of blocking primitives that share a common prefix larger than delta_min.
    delta_min = torch.min(lcp_left, lcp_right)

    # Find the farthest index j along direction d[i] where LCP > delta_min[i].
    # Start at l_max=2 and double until delta(i, i + l_max * d) <= delta_min.
    l_max = torch.full((number_of_blocking_primitives,), 2, device=device)
    while True:
        candidates_lcp = longest_common_prefix(
            codes=sorted_codes,
            i=blocker_ids,
            j=blocker_ids + l_max * direction_to_similar_codes,
            device=device,
        )
        still_expanding = candidates_lcp > delta_min
        if not still_expanding.any():
            break
        l_max = torch.where(still_expanding, l_max * 2, l_max)

    l_ = torch.zeros(number_of_blocking_primitives, dtype=torch.int32, device=device)
    t = l_max // 2
    while t.max() >= 1:
        candidates_lcp = longest_common_prefix(
            codes=sorted_codes,
            i=blocker_ids,
            j=blocker_ids + (l_ + t) * direction_to_similar_codes,
            device=device,
        )
        mask = candidates_lcp > delta_min
        l_ = torch.where(mask, l_ + t, l_)
        t = t // 2

    farthest_index = blocker_ids + l_ * direction_to_similar_codes

    # Construct binary radix tree.
    # The range [first[i], last[i]] corresponds to the spatial cluster of blocking primitives that share a common prefix in Morton code.
    # Compute splits to build LBVH tree, each internal node is assigned two children.
    delta_node = longest_common_prefix(
        codes=sorted_codes, i=blocker_ids, j=farthest_index, device=device
    )
    split = torch.zeros(number_of_blocking_primitives, dtype=torch.int32, device=device)

    t = (l_ + 1) // 2
    while t.max() >= 1:
        candidates_lcp = longest_common_prefix(
            codes=sorted_codes,
            i=blocker_ids,
            j=blocker_ids + (split + t) * direction_to_similar_codes,
            device=device,
        )
        mask = candidates_lcp > delta_node
        split = torch.where(mask, split + t, split)
        t = t // 2

    gamma = (
        blocker_ids
        + split * direction_to_similar_codes
        + torch.clamp(direction_to_similar_codes, max=0)
    )

    # LBVH:
    # left, right: Indices of the left and right child of each node (-1 if not set).
    # aabb_min, aabb_max: axis-aligned bounding box of the node.
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
    aabb_min[leaf_offset:] = primitive_mins[sorted_primitive_indices]
    aabb_max[leaf_offset:] = primitive_maxs[sorted_primitive_indices]
    is_leaf[leaf_offset:] = True
    primitive_index[leaf_offset:] = sorted_primitive_indices.to(torch.int32)

    # The lower and upper bound for the contiguous range of Morton codes that belong
    # to the cluster of a node.
    min_index = torch.minimum(blocker_ids, farthest_index)
    max_index = torch.maximum(blocker_ids, farthest_index)

    left_internal = torch.where(
        min_index[:internal_count] == gamma[:internal_count],
        (leaf_offset + gamma[:internal_count]).to(torch.int32),
        gamma[:internal_count].to(torch.int32),
    )

    right_internal = torch.where(
        max_index[:internal_count] == gamma[:internal_count] + 1,
        (leaf_offset + gamma[:internal_count] + 1).to(torch.int32),
        (gamma[:internal_count] + 1).to(torch.int32),
    )

    left[internal_nodes_indices] = left_internal.to(dtype=torch.int32, device=device)
    right[internal_nodes_indices] = right_internal.to(dtype=torch.int32, device=device)
    is_leaf[internal_nodes_indices] = False

    # Compute axis-aligned bounding boxes (AABB) for internal nodes by combining child boxes.
    # The Karras mapping ensures internal nodes form a DAG that can be evaluated in ascending order.
    nodes_with_complete_aabb = torch.zeros(
        internal_count, dtype=torch.bool, device=device
    )

    rounds = 0
    left_done = torch.ones_like(left_internal, dtype=torch.bool, device=device)
    left_mask = (left_internal >= 0) & (left_internal < internal_count)
    right_done = torch.ones_like(right_internal, dtype=torch.bool, device=device)
    right_mask = (right_internal >= 0) & (right_internal < internal_count)
    while not nodes_with_complete_aabb.all() and rounds < internal_count * 2:
        left_done[left_mask] = nodes_with_complete_aabb[left_internal[left_mask]]
        right_done[right_mask] = nodes_with_complete_aabb[right_internal[right_mask]]

        nodes_to_compute = (~nodes_with_complete_aabb) & left_done & right_done
        if not nodes_to_compute.any():
            break

        index = torch.nonzero(nodes_to_compute, as_tuple=True)[0].to(device)

        aabb_min[index] = torch.minimum(
            aabb_min[left_internal[index]], aabb_min[right_internal[index]]
        )
        aabb_max[index] = torch.maximum(
            aabb_max[left_internal[index]], aabb_max[right_internal[index]]
        )
        nodes_with_complete_aabb[index] = True
        rounds += 1

    # Warning, if some axis-aligned bounding boxes have not been computed.
    if not nodes_with_complete_aabb.all():
        log.warning(
            "Some internal nodes did not receive AABBs via DAG propagation. This means the tree was built incorrectly.",
            RuntimeWarning,
        )

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
    Compute intersection distances between rays and axis-aligned bounding boxes (AABBs).

    This method uses the slab method and the inverse ray direction for more efficient computation.

    Parameters
    ----------
    ray_origins : torch.Tensor
        Ray origins.
        Shape is ``[total_number_of_rays, 3]``.
    inverse_ray_directions : torch.Tensor
        Precomputed inverse ray directions.
        Shape is ``[total_number_of_rays, 3]``.
    aabb_min : torch.Tensor
        Minimum corner points of the AABBs.
        Shape is ``[total_number_of_rays, 3]``.
    aabb_max : torch.Tensor
        Maximum corner points of the AABBs.
        Shape is ``[total_number_of_rays, 3]``.

    Returns
    -------
    entry_distance_to_aabb : torch.Tensor
        Entry distance along each ray to the AABBs.
        Shape is ``[total_number_of_rays]``.
    exit_distance_to_aabb : torch.Tensor
        Exit distance along each ray to the AABBs.
        Shape is ``[total_number_of_rays]``.
    """
    min_distance = (aabb_min - ray_origins) * inverse_ray_directions
    max_distance = (aabb_max - ray_origins) * inverse_ray_directions
    entry_distance_to_aabb = torch.minimum(min_distance, max_distance).amax(dim=-1)
    exit_distance_to_aabb = torch.maximum(min_distance, max_distance).amin(dim=-1)
    return entry_distance_to_aabb, exit_distance_to_aabb


def compute_lbvh_max_depth(
    left: torch.Tensor,
    right: torch.Tensor,
) -> int:
    """
    Compute the maximum depth of the LBVH tree.

    Parameters
    ----------
    left : torch.Tensor
        Left child indices.
    right : torch.Tensor
        Right child indices.

    Returns
    -------
    int
        Maximum depth of the tree.
    """
    stack = [(0, 1)]
    max_depth = 1

    while stack:
        node, depth = stack.pop()
        max_depth = max(max_depth, depth)

        l_ = int(left[node].item())
        r_ = int(right[node].item())

        if l_ >= 0:
            stack.append((l_, depth + 1))
        if r_ >= 0:
            stack.append((r_, depth + 1))

    return max_depth


@torch.no_grad()
def lbvh_filter_blocking_planes(
    points_at_ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    blocking_primitives_corners: torch.Tensor,
    ray_to_heliostat_mapping: torch.Tensor,
    intersection_distances_target: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Apply the LBVH filter to filter out blocking planes that are not hit.

    Parameters
    ----------
    points_at_ray_origins : torch.Tensor
        Origin points of the rays, i.e., the surface points, expanded in the ray dimension.
        Shape is ``[number_of_heliostats, number_of_rays, number_of_combined_surface_normals_all_facets, 4]``.
    ray_directions : torch.Tensor
        Ray directions.
        Shape is ``[number_of_heliostats, number_of_rays, number_of_combined_surface_normals_all_facets, 4]``.
    blocking_primitives_corners : torch.Tensor
        Blocking primitives corner points.
        Shape is ``[number_of_blocking_planes, 4, 4]``.
    ray_to_heliostat_mapping : torch.Tensor
        Mapping indicating which ray is reflected by which heliostat.
        Shape is ``[total_number_of_rays]``.
    intersection_distances_target : torch.Tensor
        Distances from ray origins to the target.
        Shape is ``[number_of_heliostats, number_of_rays, number_of_combined_surface_normals_all_facets]``.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        Indices of the blocking primitives that are hit.
        Shape is ``[number_of_hit_blocking_planes]``.
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

    ray_origins = (
        points_at_ray_origins[:, None, :, :3]
        .expand(-1, ray_directions.shape[1], -1, -1)
        .reshape(-1, 3)
    )
    ray_directions = ray_directions[..., :3].reshape(-1, 3)
    intersection_distances_target = intersection_distances_target.reshape(-1)
    blocking_primitives_corners = blocking_primitives_corners[..., :3]

    total_number_of_rays = ray_origins.shape[0]
    number_of_primitives = blocking_primitives_corners.shape[0]

    max_tree_depth = compute_lbvh_max_depth(left=left, right=right)

    node_traversal_stack = torch.full(
        (total_number_of_rays, max_tree_depth),
        -1,
        dtype=torch.int32,
        device=device,
    )

    node_traversal_stack[:, 0] = 0
    stack_pointer = torch.ones(total_number_of_rays, dtype=torch.int32, device=device)

    hit_primitives_flag = torch.zeros(
        number_of_primitives, dtype=torch.bool, device=device
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
        mask_hit = (
            (exit_distance_to_aabb >= entry_distance_to_aabb)
            & (exit_distance_to_aabb > 1e-6)
            & (entry_distance_to_aabb <= intersection_distances_target[active_rays])
        )

        if mask_hit.any():
            hit_rays = active_rays[mask_hit]
            hit_nodes = nodes[mask_hit]
            leaf_mask = is_leaf[hit_nodes]

            if leaf_mask.any():
                leaf_rays = hit_rays[leaf_mask]
                leaf_nodes = hit_nodes[leaf_mask]
                leaf_primitives = primitive_index[leaf_nodes]

                # Remove self-intersections.
                ray_owner_hit = ray_to_heliostat_mapping[leaf_rays]
                non_self_mask = ray_owner_hit != leaf_primitives
                valid_primitives = leaf_primitives[non_self_mask]

                # Mark hit primitives.
                hit_primitives_flag[valid_primitives] = True

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
                    (index_for_left_children >= max_tree_depth)
                    & (index_for_left_children != -1)
                ).any() or (
                    (index_for_right_children >= max_tree_depth)
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

    return torch.nonzero(hit_primitives_flag, as_tuple=True)[0]
