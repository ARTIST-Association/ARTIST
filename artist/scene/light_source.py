from typing import Optional, Tuple

import torch


class LightSource(torch.nn.Module):
    """
    Abstract base class for all light sources.

    Methods
    -------
    get_distortions()
        Get distortions given the light source model.
    """

    def __init__(self):
        """Initialize the light source."""
        super().__init__()

    def get_distortions(
        self,
        number_of_points: int,
        number_of_heliostats: Optional[int] = 1,
        random_seed: Optional[int] = 7,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get distortions given the light source model.

        This function gets the distortions that are later used to model possible rays that are being generated
        from the light source. Depending on the model of the sun, the distortions are generated differently.

        Parameters
        ----------
        number_of_points : int
            The number of points on the heliostat.
        number_of_heliostats : Optional[int]
            The number of heliostats in the scenario.
        random_seed : Optional[int]
            The random seed to enable result replication.

        Raises
        ------
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError("Must be overridden!")
