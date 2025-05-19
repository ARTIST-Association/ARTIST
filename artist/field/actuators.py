from typing import Union

import torch


class Actuators(torch.nn.Module):
    """
    Implement the abstract behavior of actuators.

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
        self,
    ) -> None:
        """
        Initialize abstract actuators.

        The abstract actuator implements a template for the construction of inheriting actuators.
        An actuator is responsible for turning the heliostat surface in such a way that the
        heliostat reflects the incoming light onto the aim point on the tower. The abstract actuator specifies
        the functionality that must be implemented in the inheriting classes. These include one function to map
        the motor steps to angles and another one for the opposite conversion of angles to motor steps.
        """
        super().__init__()

    def motor_positions_to_angles(
        self,
        active_heliostats_indices: torch.Tensor,
        motor_positions: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> torch.Tensor:
        """
        Calculate the joint angles for given motor positions.

        Parameters
        ----------
        active_heliostats_indices : torch.Tensor
            The indices of the active heliostats that will be aligned.
        motor_positions : torch.Tensor
            The motor positions.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Raises
        ------
        NotImplementedError
            This abstract method must be overridden.
        """
        raise NotImplementedError("Must be overridden!")

    def angles_to_motor_positions(
        self,
        active_heliostats_indices: torch.Tensor,
        angles: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> torch.Tensor:
        """
        Calculate the motor positions for given joint angles.

        Parameters
        ----------
        active_heliostats_indices : torch.Tensor
            The indices of the active heliostats that will be aligned.
        angles : torch.Tensor
            The joint angles.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

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
