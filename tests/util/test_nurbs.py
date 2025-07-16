import torch

from artist.util import utils
from artist.util.nurbs import NURBSSurfaces
from artist.util.old_nurbs import NURBSSurfaceOld


def random_surface(
    e: torch.Tensor,
    n: torch.Tensor,
    u: torch.Tensor,
    factor: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate a random surface based on provided coefficients.

    Parameters
    ----------
    e : torch.Tensor
        The east coordinates.
    n : torch.Tensor
        The north coordinates.
    u : torch.Tensor
        The up coordinates.
    factor : float
        Factor determining how deformed the surface is.
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    torch.Tensor
        Random surface generated from the coefficients.
    """
    a, b, c, d, f, g = torch.randn(6, device=device)
    return (
        factor * a * torch.sin(e)
        + factor * b * torch.sin(n)
        + factor * c * torch.sin(u[..., 0])
        + factor * d * torch.cos(e)
        + factor * f * torch.cos(n)
        + factor * g * torch.cos(u[..., 1])
    )


def test_nurbs(device: torch.device) -> None:
    """
    Test the NURBS surface only, without ray tracing.

    First, a random surface is generated, it consists of ``surface_points``.
    Then, all the NURBS parameters are initialized (evaluation points, control points, degree,...)
    Next, the NURBS surface is initialized accordingly and fitted to the random surface created in the beginning.
    The control points of the NURBS surface are the parameters of the optimizer.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    e_range = torch.linspace(-5, 5, 40, device=device)
    n_range = torch.linspace(-5, 5, 40, device=device)

    e, n = torch.meshgrid(e_range, n_range, indexing="ij")

    u_range = torch.linspace(-5, 5, 40, device=device)

    u1, u2 = torch.meshgrid(u_range, u_range, indexing="ij")

    u = torch.stack((u1, u2), dim=-1)

    factor = 0.1

    surface = random_surface(e=e, n=n, u=u, factor=factor, device=device)
    surface_points = torch.stack((e.flatten(), n.flatten(), surface.flatten()), dim=-1)
    ones = torch.ones(surface_points.shape[0], 1, device=device)
    surface_points = torch.cat((surface_points, ones), dim=1).unsqueeze(0).unsqueeze(0)

    evaluation_points = utils.create_nurbs_evaluation_grid(
        number_of_evaluation_points=torch.tensor([40, 40], device=device), device=device
    )

    degrees = torch.tensor([2, 2], device=device)

    control_points = torch.zeros((1, 1, 20, 20, 3), device=device)
    origin_offsets_e = torch.linspace(-5, 5, control_points.shape[2], device=device)
    origin_offsets_n = torch.linspace(-5, 5, control_points.shape[3], device=device)

    control_points_e, control_points_n = torch.meshgrid(
        origin_offsets_e, origin_offsets_n, indexing="ij"
    )

    control_points[:, :, :, :, 0] = control_points_e
    control_points[:, :, :, :, 1] = control_points_n
    control_points[:, :, :, :, 2] = 0

    nurbs = NURBSSurfaces(
        degrees=degrees,
        control_points=control_points,
        device=device,
    )

    optimizer = torch.optim.Adam(nurbs.parameters(), lr=5e-3)

    for epoch in range(100):
        points, normals = nurbs.calculate_surface_points_and_normals(
            evaluation_points=evaluation_points.unsqueeze(0).unsqueeze(0), 
            device=device
        )

        optimizer.zero_grad()

        loss = points - surface_points
        loss.abs().mean().backward()

        optimizer.step()

        print(loss.abs().mean())

    torch.testing.assert_close(points, surface_points, atol=1e-2, rtol=1e-2)


def test_find_span(device: torch.device):
    """
    Test the find span method for non uniform knot vectors.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    degrees = torch.tensor([3, 3], device=device)

    evaluation_points = utils.create_nurbs_evaluation_grid(torch.tensor([4, 5], device=device), device=device)
    evaluation_points[:, 0] = utils.normalize_points(evaluation_points[:, 0])
    evaluation_points[:, 1] = utils.normalize_points(evaluation_points[:, 1])

    x, y = torch.meshgrid(
        torch.linspace(1e-2, 1 - 1e-2, 6),
        torch.linspace(1e-2, 1 - 1e-2, 6),
        indexing="ij",
    )
    control_points = torch.stack([x, y], dim=-1)

    knots = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.1, 0.7, 1.0, 1.0, 1.0, 1.0], device=device
    )

    nurbs_surface = NURBSSurfaces(
        degrees=degrees,
        control_points=control_points.unsqueeze(0).unsqueeze(0),
        device=device,
    )

    span = nurbs_surface.find_spans(
        dimension="u",
        evaluation_points=evaluation_points.unsqueeze(0).unsqueeze(0),
        knot_vectors=knots.unsqueeze(0).unsqueeze(0),
        device=device,
    )

    expected = torch.tensor([[[3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5]]], device=device)

    torch.testing.assert_close(span, expected.to(device), atol=5e-4, rtol=5e-4)


def test_nurbs_forward(device: torch.device) -> None:
    """
    Test the forward method of nurbs.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    evaluation_points = torch.cartesian_prod(
        torch.linspace(1e-5, 1 - 1e-5, 2, device=device),
        torch.linspace(1e-5, 1 - 1e-5, 2, device=device),
    ).unsqueeze(0).unsqueeze(0)
    control_points = torch.tensor(
        [[[
            [
                [-5.0000, -5.0000, 0.0000],
                [-5.0000, -1.6667, 0.0000],
                [-5.0000, 1.6667, 0.0000],
                [-5.0000, 5.0000, 0.0000],
            ],
            [
                [-1.6667, -5.0000, 0.0000],
                [-1.6667, -1.6667, 0.0000],
                [-1.6667, 1.6667, 0.0000],
                [-1.6667, 5.0000, 0.0000],
            ],
            [
                [1.6667, -5.0000, 0.0000],
                [1.6667, -1.6667, 0.0000],
                [1.6667, 1.6667, 0.0000],
                [1.6667, 5.0000, 0.0000],
            ],
            [
                [5.0000, -5.0000, 0.0000],
                [5.0000, -1.6667, 0.0000],
                [5.0000, 1.6667, 0.0000],
                [5.0000, 5.0000, 0.0000],
            ],
        ]]],
        device=device,
    )

    nurbs = NURBSSurfaces(
        degrees=torch.tensor([2, 2], device=device),
        control_points=control_points,
        device=device,
    )

    surface_points, surface_normals = nurbs(evaluation_points, device)

    expected_points = torch.tensor(
        [[[
            [-4.999866008759, -4.999866008759, 0.000000000000, 0.999999880791],
            [-4.999866485596, 4.999866008759, 0.000000000000, 0.999999940395],
            [4.999866008759, -4.999866485596, 0.000000000000, 0.999999940395],
            [4.999866485596, 4.999866485596, 0.000000000000, 1.000000000000],
        ]]],
        device=device,
    )
    expected_normals = torch.tensor(
        [[[
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]]],
        device=device,
    )

    torch.testing.assert_close(surface_points, expected_points)
    torch.testing.assert_close(surface_normals, expected_normals)
