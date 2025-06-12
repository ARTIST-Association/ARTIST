from typing import Optional

import torch

from artist.util.environment_setup import get_device


class Actuators(torch.nn.Module):
    """
    Implement the abstract behavior of actuators.

    Attributes
    ----------
    actuator_parameters : torch.Tensor
        The actuator parameters.
    active_actuator_parameters : torch.Tensor
        The active actuator parameters.

    Methods
    -------
    motor_positions_to_angles()
        Calculate the joint angles for given motor positions.
    angles_to_motor_positions()
        Calculate the motor positions for given joint angles.
    forward()
        Specify the forward pass.
    """

    def __init__(
        self, actuator_parameters: torch.Tensor, device: Optional[torch.device] = None
    ) -> None:
        """
        Initialize abstract actuators.

        The abstract actuator implements a template for the construction of inheriting actuators.
        An actuator is responsible for turning the heliostat surface in such a way that the
        heliostat reflects the incoming light onto the aim point on the tower. The abstract actuator specifies
        the functionality that must be implemented in the inheriting classes. These include one function to map
        the motor steps to angles and another one for the opposite conversion of angles to motor steps.

        Parameters
        ----------
        actuator_parameters : torch.Tensor
            The actuator parameters.
        device : Optional[torch.device]
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA, MPS, or CPU) based on availability and OS.
        """
        super().__init__()

        device = get_device(device=device)

        self.actuator_parameters = actuator_parameters

        self.active_actuator_parameters = torch.empty_like(
            self.actuator_parameters, device=device
        )

    def motor_positions_to_angles(
        self, motor_positions: torch.Tensor, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Calculate the joint angles for given motor positions.

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
            This abstract method must be overridden.
        """
        raise NotImplementedError("Must be overridden!")

    def angles_to_motor_positions(
        self, angles: torch.Tensor, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Calculate the motor positions for given joint angles.

        Parameters
        ----------
        angles : torch.Tensor
            The joint angles.
        device : Optional[torch.device]
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA, MPS, or CPU) based on availability and OS.

        Raises
        ------
        NotImplementedError
            This abstract method must be overridden.
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
