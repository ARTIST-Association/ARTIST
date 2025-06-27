"""Kinematic modules in ARTIST."""

from typing import Optional

import torch


class Kinematic(torch.nn.Module):
    """
    Abstract base class for all kinematic modules.

    Methods
    -------
    incident_ray_directions_to_orientations()
        Compute orientation matrices given incident ray directions.
    motor_positions_to_orientations()
        Compute orientation matrices given the motor positions.
    forward()
        Specify the forward pass.
    """

    def __init__(self) -> None:
        """
        Initialize the kinematic.

        The abstract kinematic implements a template for the construction of inheriting kinematics which currently
        can only be rigid body kinematics. The kinematic is concerned with the mechanics and motion of the heliostats
        and their actuators. The abstract base class defines two methods to determine orientation matrices, which all
        kinematics need to overwrite.
        """
        super().__init__()

    def incident_ray_directions_to_orientations(
        self,
        incident_ray_directions: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Compute orientation matrices given incident ray directions.

        Parameters
        ----------
        incident_ray_directions : torch.Tensor
            The directions of the incident rays as seen from the heliostats.
        device : Optional[torch.device]
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA, MPS, or CPU) based on availability and OS.

        Raises
        ------
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError("Must be overridden!")

    def motor_positions_to_orientations(
        self, motor_positions: torch.Tensor, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Compute orientation matrices given the motor positions.

        Parameters
        ----------
        motor_positions : torch.Tensor
            The motor positions.
        device : Optional[torch.device]
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA, MPS, or CPU) based on availability and OS.

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
