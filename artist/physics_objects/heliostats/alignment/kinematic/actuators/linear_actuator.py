"""
Implementation of linear actuators in ARTIST.
"""
import torch

from artist.physics_objects.heliostats.alignment.kinematic.actuators.actuator import AActuatorModule

class LinearActuator(AActuatorModule):
    """
    This class contains the implementation of linear actuators.

    Attributes
    ----------
    joint_number : int
        Descriptor (number) of the joint.
    clockwise : bool
        Turning direction of the joint.
    params : dict
        The parameters that can be optimized.

    Methods
    -------
    _steps_to_phi()

    _steps_to_angles()
        Translate the actuator steps to angles.
    _angles_to_steps()
        Translate the angles to actuator steps.
    forward()
        The forward kinematic.
    
    See Also
    --------
    :class:`AActuatorModule` : The parent class.
    """
    def __init__(
        self, joint_number: int, clockwise: bool, params: dict, **deviations
    ) -> None:
        """
        Parameters
        ----------
        joint_number : int
            Descriptor (number) of the joint.
        clockwise : bool
            Turning direction of the joint.
        params : dict
            The parameters that can be optimized.
        """
        super().__init__()
        self.joint_number = joint_number
        self.clockwise = clockwise
        self.params = params
        self.deviations = deviations
 
    def _steps_to_phi(self, actuator_pos: torch.Tensor) -> torch.Tensor:
        # Access actuator_pos via joint number: items in actuator_pos list have to be ordered by number
        stroke_length = (
            actuator_pos[self.joint_number - 1, :] / self._increment()
            + self._initial_stroke_length()
        )
        calc_step_1 = (
            self._actuator_offset() ** 2
            + self._joint_radius() ** 2
            - stroke_length**2
        )
        calc_step_2 = 2.0 * self._actuator_offset() * self._joint_radius()
        calc_step_3 = calc_step_1 / calc_step_2
        angle = torch.arccos(calc_step_3)
        return angle

    def _steps_to_angles(self, actuator_pos: torch.Tensor) -> torch.Tensor:
        """
        Translate the actuator steps to angles.

        Parameters
        ----------
        actuator_pos : torch.Tensor
            The current position of the actuator.
        
        Returns
        -------
            The angles corresponding to the actuator steps.
        """
        phi = self._steps_to_phi(actuator_pos=actuator_pos)
        phi_0 = self._steps_to_phi(actuator_pos=torch.zeros(actuator_pos.shape))
        delta_phi = phi_0 - phi

        angles = (
            self._phi_0() + delta_phi if self.clockwise else self._phi_0() - delta_phi
        )
        return angles

    def _angles_to_steps(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Translate the angles to actuator steps.

        Parameters
        ----------
        angles : torch.Tensor
            The angles that are to be converted.
        
        Returns
        -------
        torch.Tensor
            The actuator steps corresponding to the given angles.
        """
        delta_phi = angles - self._phi_0() if self.clockwise else self._phi_0() - angles

        phi_0 = self._steps_to_phi(actuator_pos=torch.zeros(angles.shape[0], 2))
        phi = phi_0 - delta_phi

        calc_step_3 = torch.cos(phi)
        calc_step_2 = 2.0 * self._actuator_offset() * self._joint_radius()
        calc_step_1 = calc_step_3 * calc_step_2
        stroke_length = torch.sqrt(
            self._actuator_offset() ** 2 + self._joint_radius() ** 2 - calc_step_1
        )
        actuator_steps = (
            stroke_length - self._initial_stroke_length()
        ) * self._increment()
        return actuator_steps

    def forward(self, actuator_pos: torch.Tensor) -> torch.Tensor:
        """
        The forward kinematic.

        Parameters
        ----------
        actuator_pos : torch.Tensor
            The position of the actuator.
        
        Returns
        -------
        torch.Tensor
            The required angles.
        """
        # Access actuator_pos via joint number: items in actuator_pos list have to be ordered by number
        return self._steps_to_angles(actuator_pos=actuator_pos)