import torch

from artist.physics_objects.module import AModule


class AActuatorModule(AModule):
    """
    This class implements the abstract behavior of an actuator.

    Attributes
    ----------
    joint_number : int
        Descriptor (number) of the joint.
    clockwise : bool
        Turning direction of the joint.

    Methods
    -------
    forward()
        The forward kinematic.

    See Also
    --------
    :class:`AModule` : The parent class.
    """

    def __init__(
        self,
        joint_number: int,
        clockwise: bool,
    ) -> None:
        super().__init__()
        self.joint_number = joint_number
        self.clockwise = clockwise

    def forward(self) -> torch.Tensor:
        """
        Perform forward kinematic.

        Raises
        ------
        NotImplementedError
            This abstract method must be Overridden.
        """
        raise NotImplementedError("Must Be Overridden!")
