import torch

from artist.field.nurbs import NURBSSurface


def test_nurbs():
    """
    Test the NURBS surface only, without raytracing.

    First a random surface is generated, it consists of surface_points.
    Then all the nurbs parameters are initialized (evaluation points, control points, degree,...)
    Next the NURBS surface is initialized accordingly and then it is fitted to the
    random surface that was created in the beginning.
    The control points of the NURBS surface are the parameters of the optimizer.
    """
    torch.manual_seed(7)
    e_range = torch.linspace(-5, 5, 40)
    n_range = torch.linspace(-5, 5, 40)

    e, n = torch.meshgrid(e_range, n_range, indexing="ij")

    u_range = torch.linspace(-5, 5, 40)

    u1, u2 = torch.meshgrid(u_range, u_range, indexing="ij")

    u = torch.stack((u1, u2), dim=-1)

    def generate_random_coefficients() -> torch.Tensor:
        """
        Generate random coefficients for the surface.

        Returns
        -------
        torch.Tensor
            Random coefficients.
        """
        return torch.randn(6)

    factor = 0.1

    def random_surface(
        e: torch.Tensor, n: torch.Tensor, u: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate a random surface based on provided coefficients.

        Parameters
        ----------
        e : torch.Tensor
            The east-coordinates.
        n : torch.Tensor
            The north-coordinates.
        u : torch.Tensor
            The up-coordinates.
        coefficients : torch.Tensor
            Coefficients to be used in generating the surface.

        Returns
        -------
        torch.Tensor
            Random surface generated from the coefficients.
        """
        a, b, c, d, e, f = coefficients
        return (
            factor * a * torch.sin(e)
            + factor * b * torch.sin(n)
            + factor * c * torch.sin(u[..., 0])
            + factor * d * torch.cos(e)
            + factor * e * torch.cos(n)
            + factor * f * torch.cos(u[..., 1])
        )

    surface_coefficients = generate_random_coefficients()

    surface = random_surface(e, n, u, surface_coefficients)
    surface_points = torch.stack((e.flatten(), n.flatten(), surface.flatten()), dim=-1)
    ones = torch.ones(surface_points.shape[0], 1)
    surface_points = torch.cat((surface_points, ones), dim=1)

    evaluation_points_e = torch.linspace(1e-5, 1 - 1e-5, 40)
    evaluation_points_n = torch.linspace(1e-5, 1 - 1e-5, 40)
    evaluation_points = torch.cartesian_prod(evaluation_points_e, evaluation_points_n)
    evaluation_points_e = evaluation_points[:, 0]
    evaluation_points_n = evaluation_points[:, 1]

    num_control_points_e = 20
    num_control_points_n = 20

    degree_e = 2
    degree_n = 2

    control_points_shape = (num_control_points_e, num_control_points_n)
    control_points = torch.zeros(
        control_points_shape + (3,),
    )
    origin_offsets_e = torch.linspace(-5, 5, num_control_points_e)
    origin_offsets_n = torch.linspace(-5, 5, num_control_points_n)
    origin_offsets = torch.cartesian_prod(origin_offsets_e, origin_offsets_n)
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
        degree_e, degree_n, evaluation_points_e, evaluation_points_n, control_points
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
