from typing import Union

import torch

from artist.field.actuator import (
    Actuator,
)


class IdealActuator(Actuator):
    """
    Implement the behavior of an ideal actuator.

    Methods
    -------
    motor_position_to_angle()
        Calculate the joint angle for a given motor position.
    angle_to_motor_position()
        Calculate the motor position for a given angle.

    See Also
    --------
    :class:`Actuator` : The parent class.
    """

    def motor_position_to_angle(
        self, motor_position: torch.Tensor, device: Union[torch.device, str] = "cuda"
    ) -> torch.Tensor:
        """
        Calculate the joint angle for a given motor position.

        Parameters
        ----------
        motor_position : torch.Tensor
            The motor position.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        torch.Tensor
            The joint angle corresponding to the motor position.
        """
        return motor_position

    def angle_to_motor_position(
        self, angle: torch.Tensor, device: Union[torch.device, str] = "cuda"
    ) -> torch.Tensor:
        """
        Calculate the motor position for a given angle.

        Parameters
        ----------
        angle : torch.Tensor
            The joint angle.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        torch.Tensor
            The motor steps.
        """
        return angle
