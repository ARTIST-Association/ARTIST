import torch

from artist.field.actuators import Actuators


class IdealActuators(Actuators):
    """
    Implement the behavior of ideal actuators.

    Attributes
    ----------
    actuator_parameters : torch.Tensor
        The actuator parameters.
        Tensor of shape [number_of_heliostats, 4, 2].
    active_actuator_parameters : torch.Tensor
        The active actuator parameters.
        Tensor of shape [number_of_active_heliostats, 4, 2].


    Methods
    -------
    motor_positions_to_angles()
        Calculate the joint angles for given motor positions.
    angles_to_motor_positions()
        Calculate the motor positions for given joint angles.

    See Also
    --------
    :class:`Actuator` : Reference to the parent class.
    """

    def __init__(
        self, actuator_parameters: torch.Tensor, device: torch.device | None = None
    ) -> None:
        """
        Initialize ideal actuators.

        Parameters
        ----------
        actuator_parameters : torch.Tensor
            The four actuator parameters.
            Tensor of shape [number_of_heliostats, 4, 2].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ''ARTIST'' will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        super().__init__(actuator_parameters=actuator_parameters, device=device)

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

        Returns
        -------
        torch.Tensor
            The joint angles corresponding to the motor positions.
        """
        return motor_positions

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

        Returns
        -------
        torch.Tensor
            The motor steps.
            Tensor of shape [number_of_active_heliostats, 2].
        """
        return angles
