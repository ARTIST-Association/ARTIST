import torch

from artist.field.actuator import (
    Actuator,
)


class IdealActuator(Actuator):
    """
    Implement the behavior of an ideal actuator.

    Methods
    -------
    motor_steps_to_angles()
        Calculate the angles given motor steps.
    angles_to_motor_steps()
        Calculate the motor steps given the angles.
    forward()
        Perform the forward kinematic.

    See Also
    --------
    :class:`Actuator` : The parent class.
    """

    def motor_steps_to_angles(
        self, motor_steps: torch.Tensor, device: torch.device = "cpu"
    ) -> torch.Tensor:
        """
        Translate motor steps to a joint angle.

        Parameters
        ----------
        motor_steps : torch.Tensor
            The motor steps.
        device : torch.device
            The device on which to initialize tensors (default is CPU).

        Returns
        -------
        torch.Tensor
            The joint angle.
        """
        return motor_steps

    def angles_to_motor_steps(
        self, angles: torch.Tensor, device: torch.device = "cpu"
    ) -> torch.Tensor:
        """
        Translate a joint angle to motor steps.

        Parameters
        ----------
        angles : torch.Tensor
            The joint angles.
        device : torch.device
            The device on which to initialize tensors (default is CPU).

        Returns
        -------
        torch.Tensor
            The motor steps.
        """
        return angles

    def forward(
        self, actuator_pos: torch.Tensor, device: torch.device = "cpu"
    ) -> torch.Tensor:
        """
        Perform the forward kinematic for an ideal actuator.

        Parameters
        ----------
        actuator_pos : torch.Tensor
            The position of the actuator.
        device : torch.device
            The device on which to initialize tensors (default is CPU).

        Returns
        -------
        torch.Tensor
            The required angles.
        """
        return actuator_pos
