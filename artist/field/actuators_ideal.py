from typing import Optional

import torch

from artist.field.actuators import Actuators


class IdealActuators(Actuators):
    """
    Implement the behavior of ideal actuators.

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

    See Also
    --------
    :class:`Actuator` : Reference to the parent class.
    """

    def __init__(
        self, actuator_parameters: torch.Tensor, device: Optional[torch.device] = None
    ) -> None:
        """
        Initialize ideal actuators.

        Parameters
        ----------
        actuator_parameters : torch.Tensor
            The two actuator parameters.
        device : Optional[torch.device]
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA, MPS, or CPU) based on availability and OS.
        """
        super().__init__(actuator_parameters=actuator_parameters, device=device)

    def motor_positions_to_angles(
        self, motor_positions: torch.Tensor, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Calculate the joint angles for given motor positions.

        Parameters
        ----------
        motor_positions : torch.Tensor
            The motor positions.
        device : Optional[torch.device]
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA, MPS, or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The joint angles corresponding to the motor positions.
        """
        return motor_positions

    def angles_to_motor_positions(
        self, angles: torch.Tensor, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Calculate the motor positions for given joint angles.

        Parameters
        ----------
        angles : torch.Tensor
            The joint angles.
        device : Optional[torch.device]
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA, MPS, or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The motor steps.
        """
        return angles

    def forward(self) -> None:
        """
        Specify the forward pass.

        Raises
        ------
        NotImplementedError
            Whenever called.
        """
        raise NotImplementedError("Not Implemented!")
