"""Kinematic modules in ARTIST."""

import torch


class Kinematic(torch.nn.Module):
    """
    Abstract base class for all kinematic modules.

    Methods
    -------
    align()
        Align given surface points and surface normals according to an input.
    forward()
        Specify the forward pass.
    """

    def __init__(self) -> None:
        """
        Initialize the kinematic.

        The abstract kinematic implements a template for the construction of inheriting kinematics which currently
        can only be rigid body kinematics. The kinematic is concerned with the mechanics and motion of the heliostats
        and their actuators. The abstract base class defines an align function that all kinematics need to overwrite
        in order to align the heliostat surfaces.
        """
        super().__init__()

    def align(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Align given surface points and surface normals according to an input.

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
