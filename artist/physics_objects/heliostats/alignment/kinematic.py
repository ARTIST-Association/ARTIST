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
    compute_reflection()
        Compute the aim points starting from the orientation of the heliostat.
    forward()
        Implement the forward kinematics.
    to_dict()
        Convert to dictionary.

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
        self._position = position

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

    def compute_reflection(self, data_points: torch.Tensor) -> torch.Tensor:
        """
        Compute the aim points starting from the orientation of the heliostat.

        Parameters
        ----------
        data_points : torch.Tensor
            Contains the information about the heliostat, light source, receiver.

        Returns
        -------
        torch.Tensor
            The derived aim points.
        """
        normal_vectors, normal_origins = self.compute_orientation(
            data_point_tensor=data_points
        )
        aim_vectors = 2 * (data_points[:, 2:5] @ normal_vectors) - data_points[:, 2:5]
        return (
            self._position + aim_vectors * (data_points[:, 5:] - self._position).norm()
        )  # Return aim points.

    def forward(self, data_points: torch.Tensor) -> torch.Tensor:
        """
        Implement the forward kinematics.

        Parameters
        ----------
        data_points : torch.Tensor
            Contains the information about the heliostat, light source, receiver.

        Returns
        -------
        torch.Tensor
            The derived aim points.

        """
        return self.compute_reflection(data_points)

    def to_dict(self):
        """Convert to dictionary."""
        return self.state_dict()
