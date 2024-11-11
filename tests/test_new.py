import torch


def line_plane_intersections(
    ray_directions: torch.Tensor,
    plane_normal_vectors: torch.Tensor,
    plane_center: torch.Tensor,
    points_at_ray_origin: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Compute line-plane intersections of ray directions and the (receiver) plane.

    Parameters
    ----------
    ray_directions : torch.Tensor
        The direction of the rays being considered for the intersection.
    plane_normal_vectors : torch.Tensor
        The normal vectors of the plane being considered for the intersection.
    plane_center : torch.Tensor
        The center of the plane being considered for the intersection.
    points_at_ray_origin : torch.Tensor
        The surface points of the ray origin.
    epsilon : float
        A small value corresponding to the upper limit (default: 1e-6).

    Returns
    -------
    torch.Tensor
        The intersections of the lines and plane.
    """
    # Use the cosine between the ray directions and the normals to calculate the relative distribution strength of
    # the incoming rays.
    relative_distribution_strengths = ray_directions @ plane_normal_vectors
    assert (
        torch.abs(relative_distribution_strengths) >= epsilon
    ).all(), "No intersection or line is within plane."

    # Calculate the final distribution strengths.
    distribution_strengths = (
        (plane_center - points_at_ray_origin)
        @ plane_normal_vectors
        / relative_distribution_strengths
    )
    return points_at_ray_origin + ray_directions * distribution_strengths.unsqueeze(-1)


intersections = line_plane_intersections(ray_directions=torch.tensor([-0.4770, -0.4373,  0.7624,  0.0000]),
                         plane_normal_vectors=torch.tensor([0.0, 1.0, 0.0, 0.0]),
                         plane_center=torch.tensor([-17.59, -2.84, 51.98, 1.0]),
                         points_at_ray_origin=torch.tensor([13.1793, 25.0380,  1.7950,  1.0000]))

print(intersections)