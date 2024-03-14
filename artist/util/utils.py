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

    return rot_matrix.permute(0, 1, 2).squeeze(-1)


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
                axis[2] * axis[1] * k - axis[0],
            ],
            [
                axis[0] * axis[2] * k - axis[1],
                axis[1] * axis[2] * k + axis[0],
                axis[2] * axis[2] * k + cosA,
            ],
        ]
    )
    return result.T


def rotate_axis_angle(u, angle_radians):
    sin_a = math.sin(angle_radians)
    cos_a = math.cos(angle_radians)
    one_minus_cos_a = 1.0 - cos_a

    rot = np.array(
        [
            [
                u[0] * u[0] * one_minus_cos_a + cos_a,
                u[1] * u[0] * one_minus_cos_a - (sin_a * u[2]),
                u[2] * u[0] * one_minus_cos_a + (sin_a * u[1]),
            ],
            [
                u[0] * u[1] * one_minus_cos_a + (sin_a * u[2]),
                u[1] * u[1] * one_minus_cos_a + cos_a,
                u[2] * u[1] * one_minus_cos_a - (sin_a * u[0]),
            ],
            [
                u[0] * u[2] * one_minus_cos_a - (sin_a * u[1]),
                u[1] * u[2] * one_minus_cos_a + (sin_a * u[0]),
                u[2] * u[2] * one_minus_cos_a + cos_a,
            ],
        ]
    )

    return rot.T


def rotate_align_new(v1, v2):
    axis = np.cross(v1, v2)
    axis /= np.linalg.norm(axis)
    dot_product = np.dot(v1, v2)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_radians = np.arccos(dot_product)
    return rotate_axis_angle(axis, angle_radians)


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
