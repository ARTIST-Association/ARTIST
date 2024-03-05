import torch
from artist.physics_objects.heliostats.alignment.kinematic.actuators.actuator import (
    AActuatorModule,
)


class IdealActuator(AActuatorModule):

    def motor_steps_to_angles(self, motor_steps: torch.Tensor):

        return motor_steps

    def angles_to_motor_steps(self, angles: torch.Tensor):

        return angles

    def forward(self, actuator_pos: torch.Tensor) -> torch.Tensor:
        """
        The forward kinematic.

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
