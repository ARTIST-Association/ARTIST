import torch

from artist.field.nurbs import NURBSSurface


def normalize_evaluation_points_for_nurbs(points: torch.Tensor) -> torch.Tensor:
    """
    Normalize evaluation points for NURBS with minimum >0 and maximum < 1 since NURBS are not defined for the edges.

    Parameters
    ----------
    points : torch.Tensor
        Evaluation points for NURBS.

    Returns
    -------
    torch.Tensor
        Normalized evaluation points for NURBS.
    """
    points_normalized = (points[:] - min(points[:]) + 1e-5) / max(
        (points[:] - min(points[:])) + 2e-5
    )
    return points_normalized


def point_cloud_to_nurbs(
    surface_points: torch.Tensor,
    num_control_points_e: int,
    num_control_points_n: int,
    tolerance: float = 1e-7,
    max_epoch: int = 2500,
) -> NURBSSurface:
    """
    Convert point cloud to NURBS surface.

    The surface points are first normalized and shifted to the range [0,1]
    in order to be compatible with the knot vector of the NURBS surface.
    The NURBS surface is then initialized with the correct number of control
    points, degrees, and knots, and the origin of the control points is set
    based on the width and height of the point cloud.
    The control points are then fitted to the surface points using an Adam
    optimizer.
    The Adam optimizer is stopped when the loss is less than the tolerance
    or the maximum number of epochs is reached.

    Parameters
    ----------
    surface_points : torch.Tensor
        The surface points given as a (N,3) tensor.
    num_control_points_e : int
        Number of NURBS control points to be set in the E (first) direction.
    num_control_points_n : int
        Number of NURBS control points to be set in the N (second) direction.
    tolerance : float, optional
        Tolerance value for convergence criteria (default is 1e-7).
    max_epoch : int, optional
        Maximum number of epochs for optimization (default is 2500).

    Returns
    -------
    NURBSSurface
        A NURBS surface.
    """
    # Normalize evaluation points and shift them so that they correspond
    # to the knots.
    evaluation_points = surface_points.clone()
    evaluation_points[:, 2] = 0
    evaluation_points_e = normalize_evaluation_points_for_nurbs(evaluation_points[:, 0])
    evaluation_points_n = normalize_evaluation_points_for_nurbs(evaluation_points[:, 1])

    # Initialize the NURBS surface.
    degree_e = 2
    degree_n = 2
    control_points_shape = (num_control_points_e, num_control_points_n)
    control_points = torch.zeros(
        control_points_shape + (3,),
    )
    width_of_point_cloud = torch.max(evaluation_points[:, 0]) - torch.min(
        evaluation_points[:, 0]
    )
    height_of_point_cloud = torch.max(evaluation_points[:, 1]) - torch.min(
        evaluation_points[:, 1]
    )
    origin_offsets_e = torch.linspace(
        -width_of_point_cloud / 2, width_of_point_cloud / 2, num_control_points_e
    )
    origin_offsets_n = torch.linspace(
        -height_of_point_cloud / 2, height_of_point_cloud / 2, num_control_points_n
    )
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

    nurbs_surface = NURBSSurface(
        degree_e, degree_n, evaluation_points_e, evaluation_points_n, control_points
    )

    # Optimize the control points of the NURBS surface to fit the surface
    # points.
    optimizer = torch.optim.Adam([control_points], lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.2,
        patience=500,
        threshold=1e-7,
        threshold_mode="abs",
    )
    loss = torch.inf
    epoch = 0
    while loss > tolerance and epoch <= max_epoch:
        points, _ = nurbs_surface.calculate_surface_points_and_normals()

        optimizer.zero_grad()
        loss = (points - surface_points).abs().mean()
        loss.backward()

        optimizer.step()
        scheduler.step(loss.abs().mean())
        if epoch % 20 == 0:
            print(
                "Epoch:",
                epoch,
                "\t Loss:",
                loss.abs().mean().item(),
                f"\t LR: {scheduler.get_last_lr()[0]}",
            )
        epoch += 1

    return nurbs_surface


def deflectometry_to_nurbs(
    surface_points: torch.Tensor,
    surface_normals: torch.Tensor,
    num_control_points_e: int,
    num_control_points_n: int,
    tolerance: float = 1e-7,
    max_epoch: int = 2500,
) -> NURBSSurface:
    """
    Convert deflectometry to NURBS surface.

    The surface points are first normalized and shifted to the range [0,1]
    in order to be compatible with the knot vector of the NURBS surface.
    The NURBS surface is then initialized with the correct number of control
    points, degrees, and knots, and the origin of the control points is set
    based on the width and height of the point cloud.
    The control points are then fitted to the surface points using an Adam
    optimizer.
    The Adam optimizer is stopped when the loss is less than the tolerance
    or the maximum number of epochs is reached.

    Parameters
    ----------
    surface_points : torch.Tensor
        The surface points given as a (N,3) tensor.
    num_control_points_e : int
        Number of NURBS control points to be set in the E (first) direction.
    num_control_points_n : int
        Number of NURBS control points to be set in the N (second) direction.
    tolerance : float, optional
        Tolerance value for convergence criteria (default is 1e-7).
    max_epoch : int, optional
        Maximum number of epochs for optimization (default is 2500).

    Returns
    -------
    NURBSSurface
        A NURBS surface.
    """
    # Normalize evaluation points and shift them so that they correspond
    # to the knots.
    evaluation_points = surface_points.clone()
    evaluation_points[:, 2] = 0
    evaluation_points_e = normalize_evaluation_points_for_nurbs(evaluation_points[:, 0])
    evaluation_points_n = normalize_evaluation_points_for_nurbs(evaluation_points[:, 1])

    # Initialize the NURBS surface.
    degree_e = 2
    degree_n = 2
    control_points_shape = (num_control_points_e, num_control_points_n)
    control_points = torch.zeros(
        control_points_shape + (3,),
    )
    width_of_point_cloud = torch.max(evaluation_points[:, 0]) - torch.min(
        evaluation_points[:, 0]
    )
    height_of_point_cloud = torch.max(evaluation_points[:, 1]) - torch.min(
        evaluation_points[:, 1]
    )
    origin_offsets_e = torch.linspace(
        -width_of_point_cloud / 2, width_of_point_cloud / 2, num_control_points_e
    )
    origin_offsets_n = torch.linspace(
        -height_of_point_cloud / 2, height_of_point_cloud / 2, num_control_points_n
    )
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

    nurbs_surface = NURBSSurface(
        degree_e, degree_n, evaluation_points_e, evaluation_points_n, control_points
    )

    # Optimize the control points of the NURBS surface to fit the surface
    # points.
    optimizer = torch.optim.Adam([control_points], lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.2,
        patience=500,
        threshold=1e-7,
        threshold_mode="abs",
    )
    loss = torch.inf
    epoch = 0
    while loss > tolerance and epoch <= max_epoch:
        _, normals = nurbs_surface.calculate_surface_points_and_normals()

        optimizer.zero_grad()
        loss = (normals - surface_normals).abs().mean()
        loss.backward()

        optimizer.step()
        scheduler.step(loss.abs().mean())
        if epoch % 20 == 0:
            print(
                "Epoch:",
                epoch,
                "\t Loss:",
                loss.abs().mean().item(),
                f"\t LR: {scheduler.get_last_lr()[0]}",
            )
        epoch += 1

    return nurbs_surface
