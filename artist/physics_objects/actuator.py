import torch


class AActuatorModule(torch.nn.Module):
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
    actuator_offset : torch.Tensor
        The offset between the linear actuator's pivoting point and the point around which the actuator is allowed to pivot.
    radius : torch.Tensor
        The actuator's pivoting radius.
    phi_0 : torch.Tensor
        The angle that the actuator introduces to the manipulated coordinate system at the initial stroke length.

    Methods
    -------
    forward()
        The forward kinematic.
    """

    def __init__(
        self,
        joint_number: int,
        clockwise: bool,
        increment: torch.Tensor,
        initial_stroke_length: torch.Tensor,
        actuator_offset: torch.Tensor,
        radius: torch.Tensor,
        phi_0: torch.Tensor,
    ) -> None:
        super().__init__()
        self.joint_number = joint_number
        self.clockwise = clockwise
        self.increment = increment
        self.initial_stroke_length = initial_stroke_length
        self.actuator_offset = actuator_offset
        self.radius = radius
        self.phi_0 = phi_0

    def forward(self) -> torch.Tensor:
        """
        Perform forward kinematic.

        Raises
        ------
        NotImplementedError
            This abstract method must be overridden.
        """
        raise NotImplementedError("Must Be Overridden!")
