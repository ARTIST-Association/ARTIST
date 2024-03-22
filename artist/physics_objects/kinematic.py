"""Kinematic modules in ARTIST."""

import typing

import torch


class AKinematicModule(torch.nn.Module):
    """
    Abstract base class for all kinematic modules.

    Attributes
    ----------
    position : torch.Tensor
        The position of the heliostat in the field.

    Methods
    -------
    align()
        Compute the orientation matrix to align the heliostat.
    forward()
        Implement the forward kinematics.

    See Also
    --------
    :class:`AModule` : The parent class.
    """

    def __init__(self, position: torch.Tensor) -> None:
        super().__init__()
        self.position = position

    def align(
        self,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the orientation matrix to align the heliostat.

        Raises
        ------
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError("Must be overridden!")

    def forward(self) -> torch.Tensor:
        """
        Implement the forward kinematics.

        Raises
        ------
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError("Must be overridden!")
