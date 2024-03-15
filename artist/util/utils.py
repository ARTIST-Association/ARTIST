import math

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


# TODO: Delete this function since it can't actually work!
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


def rotate_nu(
    n: torch.Tensor,
    u: torch.Tensor,
):
    """
    Rotate around the north and then the up axis in this very order in a right-handed east north up
    coordinate system. Positive angles result in a rotation in the mathematical direction of rotation, i.e.
    counter-clockwise. Points need to be multiplied as column vectors from the right hand side with the
    resulting rotation matrix. Note that the order is fixed due to the non-commutative property of matrix-matrix
    multiplication.

    Parameters
    ----------
    n : torch.Tensor
        North rotation angle in radians
    u : torch.Tensor
        Up rotation angle in radians.

    Returns
    -------
    torch.Tensor
        Corresponding rotation matrix.
    """
    cos_n = torch.cos(n)
    sin_n = torch.sin(n)
    cos_u = torch.cos(u)
    sin_u = torch.sin(u)
    zeros = torch.zeros(n.shape)
    ones = torch.ones(n.shape)

    return torch.stack(
        [
            torch.stack([cos_u * cos_n, -sin_u, -cos_u * sin_n, zeros], dim=1),
            torch.stack([sin_u * cos_n, cos_u, -sin_u * sin_n, zeros], dim=1),
            torch.stack([sin_n, zeros, cos_n, zeros], dim=1),
            torch.stack([zeros, zeros, zeros, ones], dim=1),
        ],
        dim=1,
    )


def rotate_enu(
    e: torch.Tensor,
    n: torch.Tensor,
    u: torch.Tensor,
):
    """
    Rotate around the east, then the north, then the up axis in this very order in a right handed east north up
    coordinate system. Positive angles result in a rotation in the mathematical direction of rotation, i.e.
    counter-clockwise. Points need to be multiplied as column vectors from the right hand side with the
    resulting rotation matrix. Note that the order is fixed due to the non-commutative property of matrix-matrix
    multiplication.

    Parameters
    ----------
    e : torch.Tensor
        East rotation angle in radians.
    n : torch.Tensor
        North rotation angle in radians
    u : torch.Tensor
        Up rotation angle in radians.

    Returns
    -------
    torch.Tensor
        Corresponding rotation matrix.
    """
    cos_e = torch.cos(e)
    sin_e = torch.sin(e)
    cos_n = torch.cos(n)
    sin_n = torch.sin(n)
    cos_u = torch.cos(u)
    sin_u = torch.sin(u)
    zeros = torch.zeros(e.shape)
    ones = torch.ones(e.shape)

    return torch.stack(
        [
            torch.stack(
                [
                    cos_u * cos_n,
                    -sin_u * cos_e + cos_u * sin_n * sin_e,
                    -sin_u * sin_e - cos_u * sin_n * cos_e,
                    zeros,
                ],
                dim=1,
            ),
            torch.stack(
                [
                    sin_u * cos_n,
                    cos_u * cos_e + sin_u * sin_n * sin_e,
                    cos_u * sin_e - sin_u * sin_n * cos_e,
                    zeros,
                ],
                dim=1,
            ),
            torch.stack([sin_n, -cos_n * sin_e, cos_n * cos_e, zeros], dim=1),
            torch.stack([zeros, zeros, zeros, ones], dim=1),
        ],
        dim=1,
    )


def translate_enu(
    e: torch.Tensor,
    n: torch.Tensor,
    u: torch.Tensor,
):
    """
    Translate a given point in the east, north and up direction. Note that the point must be multiplied as a column
    vector from the right hand side of the resulting matrix.

    Parameters
    ----------
    e : torch.Tensor
        East translation.
    n : torch.Tensor
        North translation
    u : torch.Tensor
        Up translation.

    Returns
    -------
    torch.Tensor
        Corresponding rotation matrix.
    """
    zeros = torch.zeros(e.shape)
    ones = torch.ones(e.shape)

    return torch.stack(
        [
            torch.stack([ones, zeros, zeros, e], dim=1),
            torch.stack([zeros, ones, zeros, n], dim=1),
            torch.stack([zeros, zeros, ones, u], dim=1),
            torch.stack([zeros, zeros, zeros, ones], dim=1),
        ],
        dim=1,
    )
