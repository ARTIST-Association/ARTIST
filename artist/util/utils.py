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
