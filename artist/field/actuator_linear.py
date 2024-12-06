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
    pivot_radius : torch.Tensor
        The actuator's pivoting radius.
    initial_angle : torch.Tensor
        The angle that the actuator introduces to the manipulated coordinate system at the initial stroke length.

    Methods
    -------
    motor_position_to_angle()
        Calculate the joint angle for a given motor position.
    angle_to_motor_position()
        Calculate the motor position for a given angle.
    forward()
        Specify the forward pass.

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
        pivot_radius: torch.Tensor,
        initial_angle: torch.Tensor,
    ) -> None:
        """
        Initialize a linear actuator.

        A linear actuator describes movement within a 2D plane. As there can be multiple actuators for a single
        heliostat, each actuator is labeled with a joint number. The clockwise attribute describes the turning
        direction of the actuator. The linear actuator is further parametrized by five parameters. These are the
        increment, which stores the information about the stroke length change per motor step, the initial stroke
        length, and an offset that describes the difference between the linear actuator's pivoting point and the
        point around which the actuator is allowed to pivot. Next, the actuator's pivoting radius is described by
        the pivot radius and lastly, the initial angle indicates the angle that the actuator introduces to the
        manipulated coordinate system at the initial stroke length.

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
        pivot_radius : torch.Tensor
            The actuator's pivoting radius.
        initial_angle : torch.Tensor
            The angle that the actuator introduces to the manipulated coordinate system at the initial stroke length.
        """
        super().__init__(
            joint_number,
            clockwise,
            increment,
            initial_stroke_length,
            offset,
            pivot_radius,
            initial_angle,
        )
        self.joint_number = joint_number
        self.clockwise = clockwise
        self.increment = increment
        self.initial_stroke_length = initial_stroke_length
        self.offset = offset
        self.pivot_radius = pivot_radius
        self.initial_angle = initial_angle

    def _motor_position_to_absolute_angle(
        self, motor_position: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert motor steps into an angle using actuator geometry.

        Calculate an absolute angle based solely on the motor's current position and the geometry
        of the actuator. This gives the angle of the actuator in a global sense. It  does not
        consider the starting position of the motor.

        Parameters
        ----------
        motor_position : torch.Tensor
            The motor position.

        Returns
        -------
        torch.Tensor
            The calculated absolute angle.
        """
        stroke_length = motor_position / self.increment + self.initial_stroke_length
        calc_step_1 = self.offset**2 + self.pivot_radius**2 - stroke_length**2
        calc_step_2 = 2.0 * self.offset * self.pivot_radius
        calc_step_3 = calc_step_1 / calc_step_2
        absolute_angle = torch.arccos(calc_step_3)
        return absolute_angle

    def motor_position_to_angle(
        self, motor_position: torch.Tensor, device: Union[torch.device, str] = "cuda"
    ) -> torch.Tensor:
        """
        Calculate the joint angle for a given motor position.

        Using the absolute angle calculated with _motor_position_to_absolute_angle(), the
        absolute angle is adjusted to be relative to an initial angle. It accounts for
        the initial angle and the motor's direction (clockwise or counterclockwise).

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
        absolute_angle = self._motor_position_to_absolute_angle(
            motor_position=motor_position
        )
        absolute_initial_angle = self._motor_position_to_absolute_angle(
            motor_position=torch.zeros(1, device=device)
        )
        delta_angle = absolute_initial_angle - absolute_angle

        relative_angle = (
            self.initial_angle + delta_angle
            if self.clockwise
            else self.initial_angle - delta_angle
        )
        return relative_angle

    def angle_to_motor_position(
        self, angle: torch.Tensor, device: Union[torch.device, str] = "cuda"
    ) -> torch.Tensor:
        """
        Calculate the motor position for a given angle.

        First the relative angular change is calculated based on the given angle.
        Then the corresponding stroke length is determined using trigonometric
        relationships. This stroke length is converted into motor steps.

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
        delta_angle = (
            angle - self.initial_angle if self.clockwise else self.initial_angle - angle
        )

        absolute_initial_angle = self._motor_position_to_absolute_angle(
            motor_position=torch.zeros(1, device=device)
        )
        initial_angle = absolute_initial_angle - delta_angle

        calc_step_3 = torch.cos(initial_angle)
        calc_step_2 = 2.0 * self.offset * self.pivot_radius
        calc_step_1 = calc_step_3 * calc_step_2
        stroke_length = torch.sqrt(self.offset**2 + self.pivot_radius**2 - calc_step_1)
        motor_position = (stroke_length - self.initial_stroke_length) * self.increment
        return motor_position
    
    def forward(self) -> None:
        """
        Specify the forward pass.

        Raises
        ------
        NotImplementedError
            Whenever called.
        """
        raise NotImplementedError("Not Implemented!")
