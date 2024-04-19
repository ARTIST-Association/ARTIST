from typing import Tuple

import torch


class LightSource(torch.nn.Module):
    """Abstract base class for all light sources."""

    def __init__(self):
        super().__init__()

    def get_distortions(
        self,
        number_of_points: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get distortions given the light source model.

        This function gets the distortions that are later used to model possible rays that are being generated
        from the light source. Depending on the model of the sun, the distortions are generated differently.

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
