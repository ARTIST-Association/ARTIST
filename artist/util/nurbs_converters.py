import torch

from artist.field.nurbs import NURBSSurface


def deflectometry_to_nurbs(
    surface_points: torch.Tensor,
    surface_normals: torch.Tensor,
    width: torch.Tensor,
    height: torch.Tensor,
) -> NURBSSurface:
    """
    Convert deflectometry data to a NURBS surface.

    Parameters
    ----------
    surface_points : torch.Tensor
        The surface points from deflectometry data.
    surface_normals : torch.Tensor
        The surface normals from deflectometry data.
    width : torch.Tensor
        The width of the heliostat.
    height : torch.Tensor
        The height of the heliostat.

    Returns
    -------
    NURBSSurface
        A NURBS surface.
    """
    evaluation_points = surface_points.clone()
    evaluation_points[:, 2] = 0

    # normalize evaluation points and shift them so that they correspond to the knots.
    # -> The evaluation points must also lie between 0 and 1 (like the knots).
    evaluation_points_x = (
        evaluation_points[:, 0] - min(evaluation_points[:, 0]) + 1e-5
    ) / max((evaluation_points[:, 0] - min(evaluation_points[:, 0])) + 2e-5)
    evaluation_points_y = (
        evaluation_points[:, 1] - min(evaluation_points[:, 1]) + 1e-5
    ) / max((evaluation_points[:, 1] - min(evaluation_points[:, 1])) + 2e-5)

    num_control_points_x = 7
    num_control_points_y = 7

    degree_x = 2
    degree_y = 2

    control_points_shape = (num_control_points_x, num_control_points_y)
    control_points = torch.zeros(
        control_points_shape + (3,),
    )
    origin_offsets_x = torch.linspace(-width / 2, width / 2, num_control_points_x)
    origin_offsets_y = torch.linspace(-height / 2, height / 2, num_control_points_y)
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

    nurbs_surface = NURBSSurface(
        degree_x, degree_y, evaluation_points_x, evaluation_points_y, control_points
    )

    optimizer = torch.optim.Adam([control_points], lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.4,
        patience=40,
        threshold=0.0001,
        threshold_mode="abs",
        verbose=True,
    )

    for epoch in range(500):
        points, normals = nurbs_surface.calculate_surface_points_and_normals()

        optimizer.zero_grad()

        loss = points - surface_points
        loss.abs().mean().backward()

        optimizer.step()
        scheduler.step(loss.abs().mean())

        print(loss.abs().mean())

    return nurbs_surface
