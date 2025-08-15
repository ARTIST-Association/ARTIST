import pathlib

import matplotlib.pyplot as plt
import torch

from artist.data_loader import flux_distribution_loader


def trapezoid_1d(
    total_width: torch.Tensor,
    slope_width: torch.Tensor,
    plateau_width: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Create a one dimensional trapezoid distribution.

    Parameter
    ---------
    total_width : torch.Tensor
        The total width of the trapezoid.
    slope_width : torch.Tensor
        The width of the slope of the trapezoid.
    plateau_width : torch.Tensor
        The width of the plateau.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        The one dimensional trapezoid distribution.
    """
    indices = torch.arange(total_width, device=device)
    center = (total_width - 1) / 2
    half_plateau = plateau_width / 2

    # Distances from the plateau edge.
    distances = torch.abs(indices - center) - half_plateau

    trapezoid = 1 - (distances / slope_width).clamp(min=0, max=1)

    return trapezoid


def kl_divergence(
    p: torch.Tensor, q: torch.Tensor, epsilon: float = 1e-12
) -> torch.Tensor:
    """
    Compute D_KL(P||Q) over all pixels. P and Q can be any nonnegative tensors.

    Parameters
    ----------
    p : torch.Temsor
    q : torch.Tensor

    Returns
    -------
    torch.Tensor
        The kl divergnece.
    """
    return (p * (torch.log(((p + epsilon) / (q + epsilon))))).sum()


device = torch.device("cuda")

epsilon = 1e-12

e_trapezoid = trapezoid_1d(
    total_width=128, slope_width=2, plateau_width=120, device=device
)
u_trapezoid = trapezoid_1d(
    total_width=256, slope_width=2, plateau_width=248, device=device
)

trapezoid_2d = u_trapezoid.unsqueeze(1) * e_trapezoid.unsqueeze(0)
plt.imshow(trapezoid_2d.cpu().detach())
plt.savefig("trapezoid.png")

flux = flux_distribution_loader.load_flux_from_png(
    heliostat_flux_path_mapping=[
        (
            "test_heliostat",
            [
                pathlib.Path(
                    "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA31/125284-flux.png"
                )
            ],
        )
    ],
    heliostat_names=["test_heliostat"],
    resolution=torch.tensor([128, 256], device=device),
    device=device,
)

p_normalized = (trapezoid_2d + epsilon) / (
    trapezoid_2d.sum() + epsilon * trapezoid_2d.numel()
)
q_normalized = (flux + epsilon) / (flux.sum() + epsilon * flux.numel())

kl_div_manual = kl_divergence(p=p_normalized, q=q_normalized)
kl_div_torch = torch.nn.functional.kl_div(
    input=torch.log(q_normalized), target=p_normalized
)
pass
