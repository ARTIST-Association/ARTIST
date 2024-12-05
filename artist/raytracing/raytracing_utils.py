import torch

from artist.util import utils


def reflect(
    incoming_ray_direction: torch.Tensor, reflection_surface_normals: torch.Tensor
) -> torch.Tensor:
    """
    Reflect incoming rays given the normals of a reflective surface.

    Parameters
    ----------
    incoming_ray_direction : torch.Tensor
        The direction of the incoming rays to be reflected.
    reflection_surface_normals : torch.Tensor
        The normal of the reflective surface.

    Returns
    -------
    torch.Tensor
        The reflected rays.
    """
    return (
        incoming_ray_direction
        - 2
        * utils.batch_dot(incoming_ray_direction, reflection_surface_normals)
        * reflection_surface_normals
    )


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
    
    if (torch.abs(relative_distribution_strengths) <= epsilon).all():
        raise ValueError(
            "No intersection or line is within plane."
        )
    
    # Calculate the final distribution strengths.
    distribution_strengths = (
        (plane_center - points_at_ray_origin)
        @ plane_normal_vectors
        / relative_distribution_strengths
    )
    return points_at_ray_origin + ray_directions * distribution_strengths.unsqueeze(-1)
