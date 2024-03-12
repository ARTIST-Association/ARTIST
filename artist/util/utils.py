import numpy as np
import torch


def batch_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Batch-wise dot product.

    Parameters
    ----------
    x : torch.Tensor
        Single tensor with dimension (1, 4).
    y : torch.Tensor
        Single tensor with dimension (N, 4).


    Returns
    -------
    torch.Tensor
        Dot product of x and y as a tensor with dimension (N, 1).
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
    tx: torch.Tensor,
    ty: torch.Tensor,
    tz: torch.Tensor,
    rx: torch.Tensor,
    ry: torch.Tensor,
    rz: torch.Tensor,
    sx: torch.Tensor,
    sy: torch.Tensor,
    sz: torch.Tensor,
):
    rx_cos = -torch.cos(rx)  # due to heliostat convention
    rx_sin = -torch.sin(rx)  # due to heliostat convention
    ry_cos = torch.cos(ry)
    ry_sin = torch.sin(ry)
    rz_cos = torch.cos(rz)
    rz_sin = torch.sin(rz)
    zeros = torch.zeros(rx.shape)
    ones = torch.ones(rx.shape)

    rot_matrix = torch.stack(
        [
            torch.stack([sx * ry_cos * rz_cos, rz_sin, ry_sin, zeros], dim=1),
            torch.stack([-rz_sin, sy * rx_cos * rz_cos, -rx_sin, zeros], dim=1),
            torch.stack([-ry_sin, rx_sin, sz * rx_cos * ry_cos, zeros], dim=1),
            torch.stack([tx, ty, tz, ones], dim=1),
        ],
        dim=1,
    )

    return rot_matrix.squeeze(-1)


def another_random_align_function(v1, v2):
    axis = np.cross(v1, v2)
    cosA = np.dot(v1, v2)
    k = 1.0 / (1.0 + cosA)

    result = np.array(
        [
            [
                axis[0] * axis[0] * k + cosA,
                axis[1] * axis[0] * k - axis[2],
                axis[2] * axis[0] * k + axis[1],
            ],
            [
                axis[0] * axis[1] * k + axis[2],
                axis[1] * axis[1] * k + cosA,
                axis[1] * axis[2] * k + axis[0],
            ],
            [
                axis[0] * axis[2] * k - axis[1],
                axis[2] * axis[1] * k - axis[0],
                axis[2] * axis[2] * k + cosA,
            ],
        ]
    )

    return result
