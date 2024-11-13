from typing import Union

import torch

from artist.field.actuator import Actuator


class LinearActuator(Actuator):
    """
    Implement the behavior of a linear actuator.

    Attributes
    ----------
    joint_number : int
        Descriptor (number) of the joint.
    clockwise : bool
        Turning direction of the joint.
    increment : torch.Tensor
        The stroke length change per motor step.
    initial_stroke_length : torch.Tensor
        The stroke length for a motor step of 0.
    offset : torch.Tensor
        The offset between the linear actuator's pivoting point and the point
        around which the actuator is allowed to pivot.
    radius : torch.Tensor
        The actuator's pivoting radius.
    phi_0 : torch.Tensor
        The angle that the actuator introduces to the manipulated coordinate system at the initial stroke length.

    Methods
    -------
    steps_to_phi()
        Calculate phi (angle) from steps.
    motor_steps_to_angles()
        Calculate the angles given the motor steps.
    angles_to_motor_steps()
        Calculate the motor steps given the angles.
    forward()
        Perform the forward kinematic.

    See Also
    --------
    :class:`Actuator` : Reference to the parent class.
    """

    def __init__(
        self,
        joint_number: int,
        clockwise: bool,
        increment: torch.Tensor,
        initial_stroke_length: torch.Tensor,
        offset: torch.Tensor,
        radius: torch.Tensor,
        phi_0: torch.Tensor,
    ) -> None:
        """
        Initialize a linear actuator.

        A linear actuator describes movement within a 2D plane. As there can be multiple actuators for a single
        heliostat, each actuator is labeled with a joint number. The clockwise attribute describes the turning
        direction of the actuator. The linear actuator is further parametrized by five parameters. These are the
        increment, which stores the information about the stroke length change per motor step, the initial stroke
        length, and an offset that describes the difference between the linear actuator's pivoting point and the
        point around which the actuator is allowed to pivot. Next, the actuator's pivoting radius is described by
        the radius and lastly, phi_0 indicates the angle that the actuator introduces to the manipulated coordinate
        system at the initial stroke length.

        Parameters
        ----------
        joint_number : int
            Descriptor (number) of the joint.
        clockwise : bool
            Turning direction of the joint.
        increment : torch.Tensor
            The stroke length change per motor step.
        initial_stroke_length : torch.Tensor
            The stroke length for a motor step of 0.
        offset : torch.Tensor
            The offset between the linear actuator's pivoting point and the point
            around which the actuator is allowed to pivot.
        radius : torch.Tensor
            The actuator's pivoting radius.
        phi_0 : torch.Tensor
            The angle that the actuator introduces to the manipulated coordinate system at the initial stroke length.
        """
        super().__init__(
            joint_number,
            clockwise,
            increment,
            initial_stroke_length,
            offset,
            radius,
            phi_0,
        )
        self.joint_number = joint_number
        self.clockwise = clockwise
        self.increment = increment
        self.initial_stroke_length = initial_stroke_length
        self.offset = offset
        self.radius = radius
        self.phi_0 = phi_0

    def steps_to_phi(self, actuator_pos: torch.Tensor) -> torch.Tensor:
        """
        Calculate phi (angle) from steps.

        Parameters
        ----------
        actuator_pos : torch.Tensor
            The actuator position.

        Returns
        -------
        torch.Tensor
            The calculated angle.
        """
        stroke_length = actuator_pos / self.increment + self.initial_stroke_length
        calc_step_1 = self.offset**2 + self.radius**2 - stroke_length**2
        calc_step_2 = 2.0 * self.offset * self.radius
        calc_step_3 = calc_step_1 / calc_step_2
        angle = torch.arccos(calc_step_3)
        return angle

    def motor_steps_to_angles(
        self, actuator_pos: torch.Tensor, device: Union[torch.device, str] = "cuda"
    ) -> torch.Tensor:
        """
        Calculate the angles given the motor steps.

        Parameters
        ----------
        actuator_pos : torch.Tensor
            The actuator (motor) position.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        torch.Tensor
            The angles corresponding to the motor steps.
        """
        device = torch.device(device)
        phi = self.steps_to_phi(actuator_pos=actuator_pos)
        phi_0 = self.steps_to_phi(actuator_pos=torch.zeros(1, device=device))
        delta_phi = phi_0 - phi

        angles = self.phi_0 + delta_phi if self.clockwise else self.phi_0 - delta_phi
        return angles

    def angles_to_motor_steps(
        self, angles: torch.Tensor, device: Union[torch.device, str] = "cuda"
    ) -> torch.Tensor:
        """
        Calculate the motor steps given the angles.

        Parameters
        ----------
        angles : torch.Tensor
            The angles.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        torch.Tensor
            The motor steps.
        """
        device = torch.device(device)
        delta_phi = angles - self.phi_0 if self.clockwise else self.phi_0 - angles

        phi_0 = self.steps_to_phi(actuator_pos=torch.zeros(1, device=device))
        phi = phi_0 - delta_phi

        calc_step_3 = torch.cos(phi)
        calc_step_2 = 2.0 * self.offset * self.radius
        calc_step_1 = calc_step_3 * calc_step_2
        stroke_length = torch.sqrt(self.offset**2 + self.radius**2 - calc_step_1)
        actuator_steps = (stroke_length - self.initial_stroke_length) * self.increment
        return actuator_steps

    def forward(
        self, actuator_pos: torch.Tensor, device: Union[torch.device, str] = "cuda"
    ) -> torch.Tensor:
        """
        Perform the forward kinematic.

        Parameters
        ----------
        actuator_pos : torch.Tensor
            The actuator (motor) position.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        torch.Tensor
            The angles.
        """
        return self.motor_steps_to_angles(
            actuator_pos=actuator_pos, device=torch.device(device)
        )
