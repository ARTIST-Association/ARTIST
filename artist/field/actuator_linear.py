from typing import Union

import torch

from artist.field.actuator import Actuators


class LinearActuators(Actuators):
    """
    Implement the behavior of a linear actuator.

    Attributes
    ----------
    joint_number : int
        Descriptor (number) of the joint.
    clockwise_axis_movement : bool
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
        clockwise_axis_movement: torch.Tensor,
        increments: torch.Tensor,
        initial_stroke_lengths: torch.Tensor,
        offsets: torch.Tensor,
        pivot_radii: torch.Tensor,
        initial_angles: torch.Tensor,
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
        clockwise_axis_movement : bool
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
        super().__init__()
        self.clockwise_axis_movement = clockwise_axis_movement
        self.increments = increments
        self.initial_stroke_lengths = initial_stroke_lengths
        self.offsets = offsets
        self.pivot_radii = pivot_radii
        self.initial_angles = initial_angles

    def _motor_positions_to_absolute_angles(
        self, motor_positions: torch.Tensor
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
        stroke_lengths = motor_positions / self.increments + self.initial_stroke_lengths
        calc_step_1 = self.offsets**2 + self.pivot_radii**2 - stroke_lengths**2
        calc_step_2 = 2.0 * self.offsets * self.pivot_radii
        calc_step_3 = calc_step_1 / calc_step_2
        absolute_angles = torch.arccos(calc_step_3)
        return absolute_angles

    def motor_positions_to_angles(
        self, motor_positions: torch.Tensor, device: Union[torch.device, str] = "cuda"
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
        absolute_angles = self._motor_positions_to_absolute_angles(
            motor_positions=motor_positions
        )
        absolute_initial_angles = self._motor_positions_to_absolute_angles(
            motor_positions=torch.zeros_like(motor_positions, device=device)
        )
        delta_angles = absolute_initial_angles - absolute_angles

        relative_angles = (
            self.initial_angles + delta_angles * (self.clockwise_axis_movement == 1) - delta_angles * (self.clockwise_axis_movement == 0)
        )
        return relative_angles

    def angles_to_motor_positions(
        self, angles: torch.Tensor, device: Union[torch.device, str] = "cuda"
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
        delta_angles = torch.where(
            self.clockwise_axis_movement == 1, 
            angles - self.initial_angles,
            self.initial_angles - angles
        )

        absolute_initial_angles = self._motor_positions_to_absolute_angles(
            motor_positions=torch.zeros_like(angles, device=device)
        )
        initial_angles = absolute_initial_angles - delta_angles

        calc_step_3 = torch.cos(initial_angles)
        calc_step_2 = 2.0 * self.offsets * self.pivot_radii
        calc_step_1 = calc_step_3 * calc_step_2
        stroke_lengths = torch.sqrt(self.offsets**2 + self.pivot_radii**2 - calc_step_1)
        motor_positions = (stroke_lengths - self.initial_stroke_lengths) * self.increments
        return motor_positions

    def forward(self) -> None:
        """
        Specify the forward pass.

        Raises
        ------
        NotImplementedError
            Whenever called.
        """
        raise NotImplementedError("Not Implemented!")
