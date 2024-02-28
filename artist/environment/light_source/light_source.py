from typing import Tuple

import torch


class ALightSource(torch.nn.Module):
    """
    Abstract base class for all light sources.

    See Also
    --------
    :class: torch.nn.Module : Reference to the parent class.
    """

    def __init__(self):
        super().__init__()

    def sample_distortions(
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
