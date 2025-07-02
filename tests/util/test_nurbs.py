import torch

from artist.util.nurbs import NURBSSurface


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
    surface_points = torch.cat((surface_points, ones), dim=1)

    evaluation_points_e = torch.linspace(1e-5, 1 - 1e-5, 40, device=device)
    evaluation_points_n = torch.linspace(1e-5, 1 - 1e-5, 40, device=device)
    evaluation_points = torch.cartesian_prod(evaluation_points_e, evaluation_points_n)
    evaluation_points_e = evaluation_points[:, 0]
    evaluation_points_n = evaluation_points[:, 1]

    num_control_points_e = 20
    num_control_points_n = 20

    degree_e = 2
    degree_n = 2

    control_points_shape = (num_control_points_e, num_control_points_n)
    control_points = torch.zeros(control_points_shape + (3,), device=device)
    origin_offsets_e = torch.linspace(-5, 5, num_control_points_e, device=device)
    origin_offsets_n = torch.linspace(-5, 5, num_control_points_n, device=device)
    origin_offsets = torch.cartesian_prod(origin_offsets_e, origin_offsets_n)
    origin_offsets = torch.hstack(
        (
            origin_offsets,
            torch.zeros((len(origin_offsets), 1), device=device),
        )
    )

    control_points = origin_offsets.reshape(control_points.shape)

    nurbs = NURBSSurface(
        degree_e,
        degree_n,
        evaluation_points_e,
        evaluation_points_n,
        control_points,
        device=device,
    )

    optimizer = torch.optim.Adam(nurbs.parameters(), lr=5e-3)

    for epoch in range(100):
        points, normals = nurbs.calculate_surface_points_and_normals(device=device)

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
    degree = 3
    evaluation_points = torch.linspace(1e-5, 1 - 1e-5, steps=20)

    x, y = torch.meshgrid(
        torch.linspace(1e-2, 1 - 1e-2, 6),
        torch.linspace(1e-2, 1 - 1e-2, 6),
        indexing="ij",
    )
    control_points = torch.stack([x, y], dim=-1)

    knots = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.1, 0.7, 1.0, 1.0, 1.0, 1.0], device=device
    )

    span = NURBSSurface.find_span(
        degree=degree,
        evaluation_points=evaluation_points,
        knot_vector=knots,
        control_points=control_points,
        device=device,
    )

    expected = torch.tensor(
        [3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5], device=device
    )

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
    evaluation_grid = torch.cartesian_prod(
        torch.linspace(1e-5, 1 - 1e-5, 2, device=device),
        torch.linspace(1e-5, 1 - 1e-5, 2, device=device),
    )
    control_points = torch.tensor(
        [
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
        ],
        device=device,
    )

    nurbs = NURBSSurface(
        degree_e=2,
        degree_n=2,
        evaluation_points_e=evaluation_grid[:, 0],
        evaluation_points_n=evaluation_grid[:, 1],
        control_points=control_points,
        device=device,
    )

    surface_points, surface_normals = nurbs(device)

    expected_points = torch.tensor(
        [
            [-4.999866008759, -4.999866008759, 0.000000000000, 0.999999880791],
            [-4.999866485596, 4.999866008759, 0.000000000000, 0.999999940395],
            [4.999866008759, -4.999866485596, 0.000000000000, 0.999999940395],
            [4.999866485596, 4.999866485596, 0.000000000000, 1.000000000000],
        ],
        device=device,
    )
    expected_normals = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    )

    torch.testing.assert_close(surface_points, expected_points)
    torch.testing.assert_close(surface_normals, expected_normals)
