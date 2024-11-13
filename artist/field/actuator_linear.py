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
    motor_position_to_angle()
        Calculate the joint angle for a given motor position.
    angle_to_motor_position()
        Calculate the motor position for a given angle.

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

    def steps_to_phi(self, motor_position: torch.Tensor) -> torch.Tensor:
        """
        Calculate phi (angle) from the motor position.

        Parameters
        ----------
        motor_position : torch.Tensor
            The motor position.

        Returns
        -------
        torch.Tensor
            The calculated angle phi.
        """
        stroke_length = motor_position / self.increment + self.initial_stroke_length
        calc_step_1 = self.offset**2 + self.radius**2 - stroke_length**2
        calc_step_2 = 2.0 * self.offset * self.radius
        calc_step_3 = calc_step_1 / calc_step_2
        angle = torch.arccos(calc_step_3)
        return angle

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
        device = torch.device(device)
        phi = self.steps_to_phi(motor_position=motor_position)
        phi_0 = self.steps_to_phi(motor_position=torch.zeros(1, device=device))
        delta_phi = phi_0 - phi

        angle = self.phi_0 + delta_phi if self.clockwise else self.phi_0 - delta_phi
        return angle

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
        device = torch.device(device)
        delta_phi = angle - self.phi_0 if self.clockwise else self.phi_0 - angle

        phi_0 = self.steps_to_phi(motor_position=torch.zeros(1, device=device))
        phi = phi_0 - delta_phi

        calc_step_3 = torch.cos(phi)
        calc_step_2 = 2.0 * self.offset * self.radius
        calc_step_1 = calc_step_3 * calc_step_2
        stroke_length = torch.sqrt(self.offset**2 + self.radius**2 - calc_step_1)
        motor_position = (stroke_length - self.initial_stroke_length) * self.increment
        return motor_position