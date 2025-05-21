from typing import Optional, Union

import torch

from artist.field.actuators import Actuators


class IdealActuators(Actuators):
    """
    Implement the behavior of ideal actuators.

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

    def __init__(self, actuator_parameters: torch.Tensor) -> None:
        pass

    def motor_positions_to_angles(
        self,
        motor_positions: torch.Tensor,
        active_heliostats_indices: Optional[torch.Tensor] = None,
        device: Union[torch.device, str] = "cuda",
    ) -> torch.Tensor:
        """
        Calculate the joint angles for given motor positions.

        Parameters
        ----------
        motor_positions : torch.Tensor
            The motor positions.
        active_heliostats_indices : Optional[torch.Tensor]
            The indices of the active heliostats that will be aligned (default is None).
            If none are provided, all will be selected.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        torch.Tensor
            The joint angles corresponding to the motor positions.
        """
        return motor_positions

    def angles_to_motor_positions(
        self,
        angles: torch.Tensor,
        active_heliostats_indices: Optional[torch.Tensor] = None,
        device: Union[torch.device, str] = "cuda",
    ) -> torch.Tensor:
        """
        Calculate the motor positions for given joint angles.

        Parameters
        ----------
        angles : torch.Tensor
            The joint angles.
        active_heliostats_indices : Optional[torch.Tensor]
            The indices of the active heliostats that will be aligned (default is None).
            If none are provided, all will be selected.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

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
