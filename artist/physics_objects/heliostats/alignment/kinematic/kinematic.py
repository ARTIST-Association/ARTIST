"""
Kinematic modules in ARTIST.
"""

import typing
import torch
from artist.physics_objects.module import AModule


class AKinematicModule(AModule):
    """
    Abstract base class for all kinematic modules.

    Attributes
    ----------
    _position : torch.Tensor
        The position

    Methods
    -------
    compute_orientation()
        Compute the orientation matrix to align the heliostat.
    forward()
        Implement the forward kinematics.

    See Also
    --------
    :class:`AModule` : The parent class.
    """

    def __init__(self, position: torch.Tensor) -> None:
        """
        Parameters
        ----------
        position : torch.Tensor
            The position.
        """
        super().__init__()
        self.position = position

    def compute_orientation(
        self, data_point_tensor: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the orientation matrix to align the heliostat.

        Parameters
        ----------
        data_point_tensor: torch.Tensor
            Contains the information about the heliostat, light source, receiver.

        Raises
        ------
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError("Must be overridden!")

    def forward(self, data_points: torch.Tensor) -> torch.Tensor:
        """
        Implement the forward kinematics.

        Parameters
        ----------
        data_points : torch.Tensor
            Contains the information about the heliostat, light source, receiver.

        Raises
        ------
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError("Must be overridden!")
