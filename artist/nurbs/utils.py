import torch

from artist.util import indices
from artist.util.env import get_device


def create_nurbs_evaluation_grid(
    number_of_evaluation_points: torch.Tensor,
    epsilon: float = 1e-7,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Create a grid of evaluation points for a NURBS surface.

    Parameters
    ----------
    number_of_evaluation_points : torch.Tensor
        The number of nurbs evaluation points in east and north direction.
        Shape is ``[2]``.
    epsilon : float
        Offset for the NURBS evaluation points (default is 1e-7).
        NURBS are defined in the interval of [0, 1] but have numerical instabilities at their endpoints.
        Therefore, the evaluation points need a small offset from the endpoints.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The evaluation points.
        Shape is ``[number_of_evaluation_points_e * number_of_evaluation_points_n, 2]``.
    """
    device = get_device(device=device)

    evaluation_points_e = torch.linspace(
        epsilon,
        1 - epsilon,
        int(number_of_evaluation_points[indices.evaluation_points_e].item()),
        device=device,
    )
    evaluation_points_n = torch.linspace(
        epsilon,
        1 - epsilon,
        int(number_of_evaluation_points[indices.evaluation_points_n].item()),
        device=device,
    )
    return torch.cartesian_prod(evaluation_points_e, evaluation_points_n)


def create_planar_nurbs_control_points(
    number_of_control_points: torch.Tensor,
    canting: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Create planar NURBS control points for each facet.

    The generated control points form a flat, equidistant grid.
    The grid extent is derived from the norm of the canting vectors, which encode the
    dimensions of the facets.

    Parameters
    ----------
    number_of_control_points : torch.Tensor
        The number of NURBS control points.
        Shape is ``[2]``.
    canting : torch.Tensor
        The canting vector for each facet.
        Shape is ``[number_of_facets, 2, 4]``.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        Planar control point grids for each facet.
        Shape is ``[number_of_facets, number_of_control_points_u_direction, number_of_control_points_v_direction, 3]``.
    """
    device = get_device(device=device)

    number_of_control_points = number_of_control_points.to(device)
    canting = canting.to(device)

    n_u = int(number_of_control_points[indices.nurbs_u].item())
    n_v = int(number_of_control_points[indices.nurbs_v].item())

    number_of_facets = canting.shape[indices.facet_index_unbatched]

    control_points = torch.zeros(
        (
            number_of_facets,
            n_u,
            n_v,
            3,
        ),
        device=device,
        dtype=canting.dtype,
    )

    u_lin = torch.linspace(0, 1, n_u, device=device, dtype=canting.dtype)
    v_lin = torch.linspace(0, 1, n_v, device=device, dtype=canting.dtype)

    # Per-facet extents in local in-plane directions.
    facet_dimensions = torch.norm(canting, dim=indices.canting)
    u_coordinates = (
        -facet_dimensions[:, indices.e, None]
        + 2 * facet_dimensions[:, indices.e, None] * u_lin
    )
    v_coordinates = (
        -facet_dimensions[:, indices.n, None]
        + 2 * facet_dimensions[:, indices.n, None] * v_lin
    )

    control_points[..., indices.nurbs_u] = u_coordinates[:, :, None]
    control_points[..., indices.nurbs_v] = v_coordinates[:, None, :]

    return control_points
