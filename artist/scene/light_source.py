from typing import Tuple

import torch


class LightSource(torch.nn.Module):
    """Abstract base class for all light sources."""

    def __init__(self):
        super().__init__()

    def sample(
        self,
        num_preferred_ray_directions: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample rays from a given distribution.

        Raises
        ------
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError("Must be overridden!")

    def scatter_rays(
        self,
        ray_directions: torch.Tensor,
        distortion_u: torch.Tensor,
        distortion_e: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the scattered rays for points on a surface.

        Raises
        ------
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError("Must Be Overridden!")
