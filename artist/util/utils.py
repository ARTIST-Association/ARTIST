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


def general_affine_matrix1(tx=0.0, ty=0.0, tz=0.0, rx=[0.0], ry=[0.0], rz=[0.0], sx=1.0, sy=1.0, sz=1.0):
    rx_cos = torch.cos(torch.tensor(rx))
    rx_sin = -torch.sin(torch.tensor(rx)) # due to heliostat convention
    ry_cos = torch.cos(torch.tensor(ry))
    ry_sin = torch.sin(torch.tensor(ry))
    rz_cos = torch.cos(torch.tensor(rz))
    rz_sin = torch.sin(torch.tensor(rz))

    rot_matrix = torch.stack(
        [
            torch.stack(
                [sx * ry_cos * rz_cos,   rz_sin,                  ry_sin,                torch.tensor([0.0])],    dim=1),
            torch.stack(
                [-rz_sin,                sy * rx_cos * rz_cos,    -rx_sin,                torch.tensor([0.0])],   dim=1),
            torch.stack(
                [-ry_sin,                rx_sin,                   sz * rx_cos * ry_cos,  torch.tensor([0.0])],   dim=1),
            torch.stack(
                [torch.tensor([tx]),     torch.tensor([ty]),       torch.tensor([tz]),    torch.tensor([1.0])],   dim=1),
        ],
    dim=1,
    )

    return rot_matrix


def general_affine_matrix(tx=[0.0], ty=[0.0], tz=[0.0], rx=[0.0], ry=[0.0], rz=[0.0], sx=1.0, sy=1.0, sz=1.0):
    # Convert input parameters to PyTorch tensors
    tx = torch.tensor(tx, dtype=torch.float32)
    ty = torch.tensor(ty, dtype=torch.float32)
    tz = torch.tensor(tz, dtype=torch.float32)
    rx = torch.tensor(rx, dtype=torch.float32)
    ry = torch.tensor(ry, dtype=torch.float32)
    rz = torch.tensor(rz, dtype=torch.float32)

    # Compute trigonometric functions
    rx_cos = torch.cos(rx)
    rx_sin = -torch.sin(rx)  # due to heliostat convention
    ry_cos = torch.cos(ry)
    ry_sin = torch.sin(ry)
    rz_cos = torch.cos(rz)
    rz_sin = torch.sin(rz)

    # Broadcast translation components to match other parameters
    num_points = max(len(tx), len(ty), len(tz), len(rx), len(ry), len(rz))
    tx = tx.expand(num_points)
    ty = ty.expand(num_points)
    tz = tz.expand(num_points)
    rx_cos = rx_cos.expand(num_points)
    rx_sin = rx_sin.expand(num_points)
    ry_cos = ry_cos.expand(num_points)
    ry_sin = ry_sin.expand(num_points)
    rz_cos = rz_cos.expand(num_points)
    rz_sin = rz_sin.expand(num_points)

    zeros = torch.tensor([0.0]).expand(num_points)
    ones = torch.tensor([1.0]).expand(num_points)

    # Compute rotation matrix
    rot_matrix = torch.stack([
        torch.stack([sx * ry_cos * rz_cos,   rz_sin,                  ry_sin,                 zeros]),
        torch.stack([-rz_sin,                sy * rx_cos * rz_cos,    -rx_sin,                zeros]),
        torch.stack([-ry_sin,                rx_sin,                  sz * rx_cos * ry_cos,   zeros]),
        torch.stack([tx,                     ty,                      tz,                     ones]),
    ], dim=1)

    return rot_matrix.squeeze(-1)


# Function to apply transformation matrix to multiple points
def transform_points(points, transform_matrix):
    # Ensure points have shape (4, num_points)
    if points.shape[1] != 4:
        points = torch.cat((points, torch.ones(points.shape[0], 1)), dim=1)
    
    # Perform batch matrix multiplication
    transformed_points = torch.matmul(points, transform_matrix)
    return transformed_points[:3].permute(2, 1, 0)  # Remove the homogeneous coordinates