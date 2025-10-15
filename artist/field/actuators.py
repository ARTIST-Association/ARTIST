import torch

from artist.util.environment_setup import get_device


class Actuators(torch.nn.Module):
    """
    Implement the abstract behavior of actuators.

    Attributes
    ----------
    non_optimizable_parameters : torch.Tensor
        The non-optimizable actuator parameters, describing actuator geometry.
        Tensor of shape [number_of_heliostats, 7, 2] for linear actuators or [number_of_heliostats, 4, 2] for ideal actuators.
    optimizable_parameters : torch.Tensor
        The two optimizable actuator parameters, describing the initial actuator configuration.
        Tensor of shape [number_of_heliostats, 2, 2] for linear actuators or [] for ideal actuators.
    active_non_optimizable_parameters : torch.Tensor
        Active non-optimizable geometry parameters.
        Tensor of shape [number_of_active_heliostats, 7, 2] for linear actuators or [number_of_active_heliostats, 4, 2] for ideal actuators.
    active_optimizable_parameters : torch.Tensor
        Active optimizable parameters.
        Tensor of shape [number_of_active_heliostats, 2, 2] for linear actuators or [] for ideal actuators.

    Methods
    -------
    motor_positions_to_angles()
        Calculate the joint angles for given motor positions.
    angles_to_motor_positions()
        Calculate the motor positions for given joint angles.
    forward()
        Specify the forward operation of the actuator, i.e., calculate the angles for given the motor positions.
    """

    def __init__(
        self,
        non_optimizable_parameters: torch.Tensor,
        optimizable_parameters: torch.Tensor = torch.tensor([], requires_grad=True),
        device: torch.device | None = None,
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
        non_optimizable_parameters : torch.Tensor
            The non-optimizable actuator parameters, describing actuator geometry.
            Tensor of shape [number_of_heliostats, 7, 2] for linear actuators or [number_of_heliostats, 4, 2] for ideal actuators.
        optimizable_parameters : torch.Tensor
            The two optimizable actuator parameters, describing the initial actuator configuration.
            Tensor of shape [number_of_heliostats, 2, 2] for linear actuators or [] for ideal actuators (default is torch.Tensor([])).
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        super().__init__()

        self.non_optimizable_parameters = non_optimizable_parameters
        self.optimizable_parameters = optimizable_parameters

        self.active_non_optimizable_parameters = torch.empty_like(
            self.non_optimizable_parameters, device=device
        )
        self.active_optimizable_parameters = torch.empty_like(
            self.optimizable_parameters, device=device
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
            If None, ``ARTIST`` will automatically select the most appropriate
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
            If None, ``ARTIST`` will automatically select the most appropriate
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
        Specify the forward operation of the actuator, i.e., calculate the angles for given the motor positions.

        Parameters
        ----------
        motor_positions : torch.Tensor
            The motor positions to be converted to joint angles.
            Tensor of shape [number_of_active_heliostats, 2].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
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
