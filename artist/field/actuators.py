import torch

from artist.util.environment_setup import get_device


class Actuators(torch.nn.Module):
    """
    Implement the abstract behavior of actuators.

    Attributes
    ----------
    actuator_parameters : torch.Tensor
        The actuator parameters.
    active_actuator_parameters : torch.Tensor
        The active actuator parameters.

    Methods
    -------
    motor_positions_to_angles()
        Calculate the joint angles for given motor positions.
    angles_to_motor_positions()
        Calculate the motor positions for given joint angles.
    forward()
        Specify the forward pass.
    """

    def __init__(
        self, actuator_parameters: torch.Tensor, device: torch.device | None = None
    ) -> None:
        """
        Initialize abstract actuators.

        The abstract actuator implements a template for the construction of inheriting actuators.
        An actuator is responsible for turning the heliostat surface in such a way that the
        heliostat reflects the incoming light onto the aim point on the tower. The abstract actuator specifies
        the functionality that must be implemented in the inheriting classes. These include one function to map
        the motor steps to angles and another one for the opposite conversion of angles to motor steps.

        Parameters
        ----------
        actuator_parameters : torch.Tensor
            The actuator parameters.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        super().__init__()

        device = get_device(device=device)

        self.actuator_parameters = torch.nn.Parameter(actuator_parameters)

        self.active_actuator_parameters = torch.empty_like(
            self.actuator_parameters, device=device
        )

    def motor_positions_to_angles(
        self, motor_positions: torch.Tensor, device: torch.device | None = None
    ) -> torch.Tensor:
        """
        Calculate the joint angles for given motor positions.

        Parameters
        ----------
        motor_positions : torch.Tensor
            The motor positions.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Raises
        ------
        NotImplementedError
            This abstract method must be overridden.
        """
        raise NotImplementedError("Must be overridden!")

    def angles_to_motor_positions(
        self, angles: torch.Tensor, device: torch.device | None = None
    ) -> torch.Tensor:
        """
        Calculate the motor positions for given joint angles.

        Parameters
        ----------
        angles : torch.Tensor
            The joint angles.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Raises
        ------
        NotImplementedError
            This abstract method must be overridden.
        """
        raise NotImplementedError("Must be overridden!")

    def forward(
        self, motor_positions: torch.Tensor, device: torch.device | None = None
    ) -> torch.Tensor:
        """
        Specify the forward pass.

        Parameters
        ----------
        motor_positions : torch.Tensor
            The motor positions to be converted to joint angles.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The joint angles.
        """
        return self.motor_positions_to_angles(
            motor_positions=motor_positions, device=device
        )
