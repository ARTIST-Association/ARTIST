import torch

from artist.field.actuators import Actuators


class IdealActuators(Actuators):
    """
    Implement the behavior of ideal actuators.

    Attributes
    ----------
    non_optimizable_parameters : torch.Tensor
        The four non-optimizable actuator parameters, describing actuator geometry.
        Shape is ``[number_of_heliostats, 4, 2]``.
    optimizable_parameters : torch.Tensor
        The ideal actuators do not have optimizable parameters, this tensor is therefore empty.
        Shape is ``[]``.
    active_non_optimizable_parameters : torch.Tensor
        Active non-optimizable geometry parameters.
        Shape is ``[number_of_active_heliostats, 4, 2]``.
    active_optimizable_parameters : torch.Tensor
        The ideal actuators do not have optimizable parameters, this tensor is therefore empty.
        Shape is ``[]``.

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
        self,
        non_optimizable_parameters: torch.Tensor,
        optimizable_parameters: torch.Tensor = torch.tensor([], requires_grad=True),
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize ideal actuators.

        Parameters
        ----------
        non_optimizable_parameters : torch.Tensor
            The four non-optimizable actuator parameters, describing actuator geometry.
            Shape is ``[number_of_heliostats, 4, 2]``.
        optimizable_parameters : torch.Tensor
            The ideal actuators do not have optimizable parameters, this tensor is therefore empty
            (default is ``torch.tensor([])``).
            Shape is ``[]``.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        super().__init__(
            non_optimizable_parameters=non_optimizable_parameters,
            optimizable_parameters=optimizable_parameters,
            device=device,
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
            Shape is ``[number_of_active_heliostats, 2]``.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
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
            Shape is ``[number_of_active_heliostats, 2]``.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The motor steps.
            Shape is ``[number_of_active_heliostats, 2]``.
        """
        return angles
