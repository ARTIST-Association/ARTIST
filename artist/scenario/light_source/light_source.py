from typing import Tuple
import torch


class ALightSource(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def sample(self, num_rays_on_hel: int,
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Must Be Overridden!")

    def compute_rays(self,
                     planeNormal: torch.Tensor,
                     planePoint: torch.Tensor,
                     ray_directions: torch.Tensor,
                     hel_in_field: torch.Tensor,
                     xi: torch.Tensor,
                     yi: torch.Tensor,
                     ) -> torch.Tensor:
        raise NotImplementedError("Must Be Overridden!")
