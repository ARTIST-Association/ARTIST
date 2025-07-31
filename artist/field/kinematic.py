"""Kinematic modules in ARTIST."""

from typing import Optional

import torch

from artist.util.environment_setup import get_device


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
        Specify the forward operation of the kinematic, i.e. calculate orientation matrices given the incident ray directions.
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
        aim_points: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Compute orientation matrices given incident ray directions.

        Parameters
        ----------
        incident_ray_directions : torch.Tensor
            The directions of the incident rays as seen from the heliostats.
        aim_points : torch.Tensor
            The aim points for the active heliostats.
        device : Optional[torch.device]
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

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
            device (CUDA or CPU) based on availability and OS.

        Raises
        ------
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError("Must be overridden!")

    def forward(
        self,
        incident_ray_directions: torch.Tensor,
        aim_points: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Specify the forward operation of the kinematic, i.e. calculate orientation matrices given the incident ray directions.

        Parameters
        ----------
        incident_ray_directions : torch.Tensor
            The directions of the incident rays as seen from the heliostats.
        aim_points : torch.Tensor
            The aim points for the active heliostats.
        device : Optional[torch.device]
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The orientation matrices.
        """
        device = get_device(device=device)

        return self.incident_ray_directions_to_orientations(
            incident_ray_directions=incident_ray_directions,
            aim_points=aim_points,
            device=device,
        )
