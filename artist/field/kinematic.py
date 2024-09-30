"""Kinematic modules in ARTIST."""

import torch


class Kinematic(torch.nn.Module):
    """
    Abstract base class for all kinematic modules.

    Attributes
    ----------
    aim_point : torch.Tensor
        The aim point of the heliostat.
    position : torch.Tensor
        The position of the heliostat.

    Methods
    -------
    align()
        Compute the orientation matrix to align the heliostat.
    forward()
        Implement the forward kinematics.
    """

    def __init__(self, position: torch.Tensor, aim_point: torch.Tensor) -> None:
        """
        Initialize the kinematic.

        The abstract kinematic implements a template for the construction of inheriting kinematics which currently
        can only be rigid body kinematics. The kinematic is concerned with the mechanics and motion of the heliostat
        and its actuators. The abstract base class defines an align function that all kinematics need to overwrite
        in order to align the heliostat surface according to a provided aim point.

        Parameters
        ----------
        position : torch.Tensor
            The position of the heliostat.
        aim_point : torch.Tensor
            The aim point of the heliostat.
        """
        super().__init__()
        self.position = position
        self.aim_point = aim_point

    def align(
        self,
        incident_ray_direction: torch.Tensor,
        max_num_iterations: int = 2,
        min_eps: float = 0.0001,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the rotation matrix to align the concentrator along a desired orientation.

        Parameters
        ----------
        incident_ray_direction : torch.Tensor
            The direction of the incident ray as seen from the heliostat.
        max_num_iterations : int
            Maximum number of iterations (default: 2).
        min_eps : float
            Convergence criterion (default: 0.0001).

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
