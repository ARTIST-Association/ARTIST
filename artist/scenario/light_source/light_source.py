from typing import Tuple
import torch


class ALightSource(torch.nn.Module):
    """
    Abstract base class for all light sources.

    See Also
    --------
    :class: torch.nn.Module : Reference to the parent class
    """

    def __init__(self):
        super().__init__()

    def sample(
        self,
        num_rays_on_hel: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample rays from a given distribution.

        Raises
        ------
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError("Must Be Overridden!")

    def compute_rays(
        self,
        plane_normal: torch.Tensor,
        plane_point: torch.Tensor,
        ray_directions: torch.Tensor,
        hel_in_field: torch.Tensor,
        xi: torch.Tensor,
        yi: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the scattered rays for points on a surface.

        Raises
        ------
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError("Must Be Overridden!")
