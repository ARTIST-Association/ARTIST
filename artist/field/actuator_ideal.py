import torch

from artist.field.actuator import (
    Actuator,
)


class IdealActuator(Actuator):
    """
    This class implements the behavior of an ideal actuator.

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

    def motor_steps_to_angles(self, motor_steps: torch.Tensor) -> torch.Tensor:
        """
        Translate motor steps to a joint angle.

        Parameters
        ----------
        motor_steps : torch.Tensor
            The motor steps.

        Returns
        -------
        torch.Tensor
            The joint angle.
        """
        return motor_steps

    def angles_to_motor_steps(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Translate a joint angle to motor steps.

        Parameters
        ----------
        angles : torch.Tensor
            The joint angles.

        Returns
        -------
        torch.Tensor
            The motor steps.
        """
        return angles

    def forward(self, actuator_pos: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward kinematic for an ideal actuator.

        Parameters
        ----------
        actuator_pos : torch.Tensor
            The position of the actuator.

        Returns
        -------
        torch.Tensor
            The required angles.
        """
        return actuator_pos
