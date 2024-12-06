from typing import Union

import torch


class Actuator(torch.nn.Module):
    """
    Implement the abstract behavior of an actuator.

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
        Initialize an abstract actuator.

        The abstract actuator implements a template for the construction of inheriting actuators which can be
        ideal or linear. An actuator is responsible for turning the heliostat surface in such a way that the
        heliostat reflects the incoming light onto the aim point on the receiver. The abstract actuator specifies
        the functionality that must be implemented in the inheriting classes. These include one function to map
        the motor steps to angles and another one for the opposite conversion of angles to motor steps.


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
        super().__init__()
        self.joint_number = joint_number
        self.clockwise = clockwise
        self.increment = increment
        self.initial_stroke_length = initial_stroke_length
        self.offset = offset
        self.pivot_radius = pivot_radius
        self.initial_angle = initial_angle

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

        Raises
        ------
        NotImplementedError
            This abstract method must be overridden.
        """
        raise NotImplementedError("Must be overridden!")

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