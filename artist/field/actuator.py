import torch


class Actuator(torch.nn.Module):
    """
    This class implements the abstract behavior of an actuator.

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
        The offset between the linear actuator's pivoting point and the point around which the actuator is allowed to pivot.
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
            The offset between the linear actuator's pivoting point and the point around which the actuator is allowed to pivot.
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

    def motor_steps_to_angles(self, motor_steps: torch.Tensor) -> torch.Tensor:
        """
        Translate motor steps to a joint angle.

        Parameters
        ----------
        motor_steps : torch.Tensor
            The motor steps.

        Raises
        ------
        NotImplementedError
            This abstract method must be overridden.
        """
        raise NotImplementedError("Must be overridden!")

    def angles_to_motor_steps(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Translate a joint angle to motor steps.

        Parameters
        ----------
        angles : torch.Tensor
            The joint angles.

        Raises
        ------
        NotImplementedError
            This abstract method must be overridden.
        """
        raise NotImplementedError("Must be overridden!")

    def forward(self, actuator_pos: torch.Tensor) -> torch.Tensor:
        """
        Perform forward kinematic.

        Parameters
        ----------
        actuator_pos : torch.Tensor
            The position of the actuator.

        Raises
        ------
        NotImplementedError
            This abstract method must be overridden.
        """
        raise NotImplementedError("Must be overridden!")
