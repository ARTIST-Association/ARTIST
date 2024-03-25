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
        The increment of an actuator.
    initial_stroke_length : torch.Tensor
        The initial_stroke_length.
    actuator_offset : torch.Tensor
        The actuator offset.
    joint_radius : toorch.Tensor
        The joint radius.
    phi_0 : torch.Tensor
        An initial angle.

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
        joint_radius: torch.Tensor,
        phi_0: torch.Tensor,
    ) -> None:
        super().__init__()
        self.joint_number = joint_number
        self.clockwise = clockwise
        self.increment = increment
        self.initial_stroke_length = initial_stroke_length
        self.actuator_offset = actuator_offset
        self.joint_radius = joint_radius
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
