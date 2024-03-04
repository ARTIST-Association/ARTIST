import torch


def batch_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Batch-wise dot product.

    Parameters
    ----------
    x : torch.Tensor
        Single tensor with dimension (1,3).
    y : torch.Tensor
        Single tensor with dimension (N,3).


    Returns
    -------
    torch.Tensor
        Dot product of x and y as a tensor with dimension (N,3).
    """
    return (x * y).sum(-1).unsqueeze(-1)


def only_rotation_matrix(
    rx=torch.tensor([0.0]), rz=torch.tensor([0.0])
) -> torch.Tensor:
    """
    Create a transformation matrix for x and z rotations only.

    Parameters
    ----------
    rx : torch.Tensor
        Angle of rotation around the x axis.
    ry : torch.Tensor
        Angle of rotation around the y axis.

    Returns
    -------
    torch.Tensor
        Rotation matrix for x and z rotation.
    """
    # Compute trigonometric functions
    rx_cos = torch.cos(rx)
    rx_sin = -torch.sin(rx)  # due to heliostat convention
    rz_cos = torch.cos(rz)
    rz_sin = torch.sin(rz)
    zeros = torch.zeros(rx.shape)
    ones = torch.ones(rx.shape)

    # Compute rotation matrix
    rot_matrix = torch.stack(
        [
            torch.stack([rz_cos, rz_sin, zeros, zeros]),
            torch.stack([-rz_sin, rx_cos * rz_cos, -rx_sin, zeros]),
            torch.stack([zeros, rx_sin, rx_cos, zeros]),
            torch.stack([zeros, zeros, zeros, ones]),
        ],
        dim=1,
    )

    return rot_matrix.permute(2, 3, 0, 1)


def general_affine_matrix(
    tx=[0.0], ty=[0.0], tz=[0.0], rx=[0.0], ry=[0.0], rz=[0.0], sx=1.0, sy=1.0, sz=1.0
):
    rx_cos = torch.cos(torch.tensor(rx))
    rx_sin = -torch.sin(torch.tensor(rx))  # due to heliostat convention
    ry_cos = torch.cos(torch.tensor(ry))
    ry_sin = torch.sin(torch.tensor(ry))
    rz_cos = torch.cos(torch.tensor(rz))
    rz_sin = torch.sin(torch.tensor(rz))

    rot_matrix = torch.stack(
        [
            torch.stack(
                [sx * ry_cos * rz_cos, rz_sin, ry_sin, torch.tensor([0.0])], dim=1
            ),
            torch.stack(
                [-rz_sin, sy * rx_cos * rz_cos, -rx_sin, torch.tensor([0.0])], dim=1
            ),
            torch.stack(
                [-ry_sin, rx_sin, sz * rx_cos * ry_cos, torch.tensor([0.0])], dim=1
            ),
            torch.stack([tx, ty, tz, torch.tensor([1.0])], dim=1),
        ],
        dim=1,
    )

    return rot_matrix
