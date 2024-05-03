import torch

from artist.field.nurbs import NURBSSurface


def test_nurbs() -> None:
    """
    Test the NURBS surface only, without raytracing.

    First a random surface is generated, it consists of ``surface_points``.
    Then, all the NURBS parameters are initialized (evaluation points, control points, degree,...)
    Next, the NURBS surface is initialized accordingly and then it is fitted to the
    random surface that was created in the beginning.
    The control points of the NURBS surface are the parameters of the optimizer.
    """
    torch.manual_seed(7)
    x_range = torch.linspace(-5, 5, 40)
    y_range = torch.linspace(-5, 5, 40)

    x, y = torch.meshgrid(x_range, y_range, indexing="ij")

    z_range = torch.linspace(-5, 5, 40)

    z1, z2 = torch.meshgrid(z_range, z_range, indexing="ij")

    z = torch.stack((z1, z2), dim=-1)

    def generate_random_coefficients() -> torch.Tensor:
        # Generate random coefficients.
        return torch.randn(6)

    factor = 0.1

    def random_surface(x, y, z, coefficients):
        a, b, c, d, e, f = coefficients
        return (
            factor * a * torch.sin(x)
            + factor * b * torch.sin(y)
            + factor * c * torch.sin(z[..., 0])
            + factor * d * torch.cos(x)
            + factor * e * torch.cos(y)
            + factor * f * torch.cos(z[..., 1])
        )

    surface_coefficients = generate_random_coefficients()

    surface = random_surface(x, y, z, surface_coefficients)
    surface_points = torch.stack((x.flatten(), y.flatten(), surface.flatten()), dim=-1)
    ones = torch.ones(surface_points.shape[0], 1)
    surface_points = torch.cat((surface_points, ones), dim=1)

    evaluation_points_x = torch.linspace(1e-5, 1 - 1e-5, 40)
    evaluation_points_y = torch.linspace(1e-5, 1 - 1e-5, 40)
    evaluation_points = torch.cartesian_prod(evaluation_points_x, evaluation_points_y)
    evaluation_points_x = evaluation_points[:, 0]
    evaluation_points_y = evaluation_points[:, 1]

    num_control_points_x = 20
    num_control_points_y = 20

    degree_x = 2
    degree_y = 2

    control_points_shape = (num_control_points_x, num_control_points_y)
    control_points = torch.zeros(
        control_points_shape + (3,),
    )
    origin_offsets_x = torch.linspace(-5, 5, num_control_points_x)
    origin_offsets_y = torch.linspace(-5, 5, num_control_points_y)
    origin_offsets = torch.cartesian_prod(origin_offsets_x, origin_offsets_y)
    origin_offsets = torch.hstack(
        (
            origin_offsets,
            torch.zeros((len(origin_offsets), 1)),
        )
    )

    control_points = torch.nn.parameter.Parameter(
        (origin_offsets).reshape(control_points.shape)
    )

    nurbs = NURBSSurface(
        degree_x, degree_y, evaluation_points_x, evaluation_points_y, control_points
    )

    optimizer = torch.optim.Adam([control_points], lr=5e-3)

    for epoch in range(100):
        points, normals = nurbs.calculate_surface_points_and_normals()

        optimizer.zero_grad()

        loss = points - surface_points
        loss.abs().mean().backward()

        optimizer.step()

        print(loss.abs().mean())

    torch.testing.assert_close(points, surface_points, atol=1e-2, rtol=1e-2)
