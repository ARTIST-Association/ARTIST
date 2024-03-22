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


def rotate_distortions(
    e: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """
    Rotate the distortions for the sun.

    Rotate around the up and then the east axis in this very order in a right-handed east north up
    coordinate system. Positive angles result in a rotation in the mathematical direction of rotation, i.e.
    counter-clockwise. Points need to be multiplied as column vectors from the right hand side with the
    resulting rotation matrix. Note that the order is fixed due to the non-commutative property of matrix-matrix
    multiplication.

    Parameters
    ----------
    e : torch.Tensor
        East rotation angle in radians
    u : torch.Tensor
        Up rotation angle in radians.

    Returns
    -------
    torch.Tensor
        Corresponding rotation matrix.
    """
    assert (
        e.shape == u.shape
    ), "The two tensors containing angles for the east and up rotation must have the same shape."

    cos_e = torch.cos(e)
    sin_e = -torch.sin(e)  # Heliostat convention
    cos_u = torch.cos(u)
    sin_u = torch.sin(u)
    zeros = torch.zeros(e.shape)
    ones = torch.ones(e.shape)

    return torch.stack(
        [
            torch.stack([cos_u, -sin_u, zeros, zeros], dim=1),
            torch.stack([cos_e * sin_u, cos_e * cos_u, sin_e, zeros], dim=1),
            torch.stack([-sin_e * sin_u, -sin_e * cos_u, cos_e, zeros], dim=1),
            torch.stack([zeros, zeros, zeros, ones], dim=1),
        ],
        dim=1,
    ).permute(0, 3, 1, 2)


def rotate_e(
    e: torch.Tensor,
) -> torch.Tensor:
    """
    Rotate around the east axis.

    Rotate around the east axis in a right-handed east north up coordinate system. Positive angles result in a rotation
    in the mathematical direction of rotation, i.e. counter-clockwise. Points need to be multiplied as column vectors
    from the right hand side with the resulting rotation matrix.


    Parameters
    ----------
    e : torch.Tensor
        East rotation angle in radians.

    Returns
    -------
    torch.Tensor
        Corresponding rotation matrix.
    """
    cos_e = torch.cos(e)
    sin_e = -torch.sin(e)  # Heliostat convention
    zeros = torch.zeros(e.shape)
    ones = torch.ones(e.shape)
    return torch.stack(
        [
            torch.stack([ones, zeros, zeros, zeros]),
            torch.stack([zeros, cos_e, sin_e, zeros]),
            torch.stack([zeros, -sin_e, cos_e, zeros]),
            torch.stack([zeros, zeros, zeros, ones]),
        ],
    ).squeeze(-1)


def rotate_n(
    n: torch.Tensor,
) -> torch.Tensor:
    """
    Rotate around the north axis.

    Rotate around the north axis in a right-handed east north up
    coordinate system. Positive angles result in a rotation in the mathematical direction of rotation, i.e.
    counter-clockwise. Points need to be multiplied as column vectors from the right hand side with the
    resulting rotation matrix.

    Parameters
    ----------
    n : torch.Tensor
        North rotation angle in radians.

    Returns
    -------
    torch.Tensor
        Corresponding rotation matrix.
    """
    cos_n = torch.cos(n)
    sin_n = torch.sin(n)
    zeros = torch.zeros(n.shape)
    ones = torch.ones(n.shape)

    return torch.stack(
        [
            torch.stack([cos_n, zeros, -sin_n, zeros]),
            torch.stack([zeros, ones, zeros, zeros]),
            torch.stack([sin_n, zeros, cos_n, zeros]),
            torch.stack([zeros, zeros, zeros, ones]),
        ],
    ).squeeze(-1)


def rotate_u(
    u: torch.Tensor,
):
    """
    Rotate around the up axis.

    Rotate around the up axis in a right-handed east north up
    coordinate system. Positive angles result in a rotation in the mathematical direction of rotation, i.e.
    counter-clockwise. Points need to be multiplied as column vectors from the right hand side with the
    resulting rotation matrix.

    Parameters
    ----------
    u : torch.Tensor
        Up rotation angle in radians

    Returns
    -------
    torch.Tensor
        Corresponding rotation matrix.
    """
    cos_u = torch.cos(u)
    sin_u = torch.sin(u)
    zeros = torch.zeros(u.shape)
    ones = torch.ones(u.shape)

    return torch.stack(
        [
            torch.stack([cos_u, -sin_u, zeros, zeros]),
            torch.stack([sin_u, cos_u, zeros, zeros]),
            torch.stack([zeros, zeros, ones, zeros]),
            torch.stack([zeros, zeros, zeros, ones]),
        ],
    ).squeeze(-1)


def translate_enu(
    e: torch.Tensor,
    n: torch.Tensor,
    u: torch.Tensor,
):
    """
    Trnslate in all directions.

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
    assert (
        e.shape == u.shape == n.shape
    ), "The three tensors containing the east, north and up translations must have the same shape."

    zeros = torch.zeros(e.shape)
    ones = torch.ones(e.shape)

    return torch.stack(
        [
            torch.stack([ones, zeros, zeros, e]),
            torch.stack([zeros, ones, zeros, n]),
            torch.stack([zeros, zeros, ones, u]),
            torch.stack([zeros, zeros, zeros, ones]),
        ],
    ).squeeze(-1)
