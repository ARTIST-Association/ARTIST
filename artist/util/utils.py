import torch


def batch_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Batch-wise dot product.

    Parameters
    ----------
    x : torch.Tensor (1,3)
        single tensor for computing dot product
    y : torch.Tensor (N,3)
        Nx3 tensor to multiply by x


    Returns
    -------
    torch.Tensor
        (N,3) dot product of x and y.
    """
    return (x * y).sum(-1).unsqueeze(-1)
