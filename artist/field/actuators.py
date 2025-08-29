import torch

from artist.util.environment_setup import get_device


class Actuators(torch.nn.Module):
    """
    Implement the abstract behavior of actuators.

    Attributes
    ----------
    actuator_parameters : torch.Tensor
        The actuator parameters.
        Tensor of shape [number_of_heliostats, n, 2], where n=7 for linear actuators or n=2 for ideal actuators.
    active_actuator_parameters : torch.Tensor
        The active actuator parameters.
        Tensor of shape [number_of_active_heliostats, n, 2], where n=7 for linear actuators or n=2 for ideal actuators.

    Methods
    -------
    motor_positions_to_angles()
        Calculate the joint angles for given motor positions.
    angles_to_motor_positions()
        Calculate the motor positions for given joint angles.
    forward()
        Specify the forward operation of the actuator, i.e. calculate the angles for given the motor positions.
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
            Tensor of shape [number_of_heliostats, n, 2], where n=7 for linear actuators or n=2 for ideal actuators.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ''ARTIST'' will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        super().__init__()

        device = get_device(device=device)

        self.actuator_parameters = actuator_parameters

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
            Tensor of shape [number_of_active_heliostats, 2].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ''ARTIST'' will automatically select the most appropriate
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
            Tensor of shape [number_of_active_heliostats, 2].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ''ARTIST'' will automatically select the most appropriate
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
        Specify the forward operation of the actuator, i.e. calculate the angles for given the motor positions.

        Parameters
        ----------
        motor_positions : torch.Tensor
            The motor positions to be converted to joint angles.
            Tensor of shape [number_of_active_heliostats, 2].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ''ARTIST'' will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The joint angles.
            Tensor of shape [number_of_active_heliostats, 2].

        """
        device = get_device(device=device)

        return self.motor_positions_to_angles(
            motor_positions=motor_positions, device=device
        )
