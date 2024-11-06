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
    radius : torch.Tensor
        The actuator's pivoting radius.
    phi_0 : torch.Tensor
        The angle that the actuator introduces to the manipulated coordinate system at the initial stroke length.

    Methods
    -------
    forward()
        The forward kinematic.
    motor_steps_to_angles()
        Translate motor steps to a joint angle.
    angles_to_motor_steps()
        Translate a joint angle to motor steps.
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
        radius : torch.Tensor
            The actuator's pivoting radius.
        phi_0 : torch.Tensor
            The angle that the actuator introduces to the manipulated coordinate system at the initial stroke length.
        """
        super().__init__()
        self.joint_number = joint_number
        self.clockwise = clockwise
        self.increment = increment
        self.initial_stroke_length = initial_stroke_length
        self.offset = offset
        self.radius = radius
        self.phi_0 = phi_0

    def motor_steps_to_angles(
        self, motor_steps: torch.Tensor, device: Union[torch.device, str] = "cuda"
    ) -> torch.Tensor:
        """
        Translate motor steps to a joint angle.

        Parameters
        ----------
        motor_steps : torch.Tensor
            The motor steps.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Raises
        ------
        NotImplementedError
            This abstract method must be overridden.
        """
        raise NotImplementedError("Must be overridden!")

    def angles_to_motor_steps(
        self, angles: torch.Tensor, device: Union[torch.device, str] = "cuda"
    ) -> torch.Tensor:
        """
        Translate a joint angle to motor steps.

        Parameters
        ----------
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

    def forward(
        self, actuator_pos: torch.Tensor, device: Union[torch.device, str] = "cuda"
    ) -> torch.Tensor:
        """
        Perform forward kinematic.

        Parameters
        ----------
        actuator_pos : torch.Tensor
            The position of the actuator.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Raises
        ------
        NotImplementedError
            This abstract method must be overridden.
        """
        raise NotImplementedError("Must be overridden!")
