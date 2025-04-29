"""Heliostat group in ARTIST."""

import torch


class HeliostatGroup(torch.nn.Module):
    """
    Abstract base class for all heliostat groups.

    Methods
    -------
    align()
        Align all heliostats within this group.
    forward()
        Specify the forward pass.
    """
    
    def __init__(self) -> None:
        """
        Initialize the heliostat group.

        The abstract heliostat group implements a template for the construction of inheriting heliostat groups, each
        with a specific kinematic type and specific actuator type. All heliostat groups together form the overall heliostat 
        field. The abstract base class defines an align function that all heliostat groups need to overwrite
        in order to align the heliostats within this group.
        """
        super().__init__()

    def align(
        self,
    ) -> None:
        """
        Align all heliostats within this group.

        Raises
        ------
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError("Must be overridden!")

    def forward(self) -> None:
        """
        Specify the forward pass.

        Raises
        ------
        NotImplementedError
            Whenever called.
        """
        raise NotImplementedError("Must be overridden!")
