import torch

from artist.field.actuators import Actuators
from artist.util.environment_setup import get_device


class LinearActuators(Actuators):
    """
    Implement the behavior of linear actuators.

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

    See Also
    --------
    :class:`Actuator` : Reference to the parent class.
    """

    def __init__(
        self, actuator_parameters: torch.Tensor, device: torch.device | None = None
    ) -> None:
        """
        Initialize linear actuators.

        A linear actuator describes movement within a 2D plane. One linear actuator has seven parameters.
        Ordered by index, the first parameter describes the type of the actuator, i.e. linear, the second parameter
        describes the turning direction of the actuator. The next five parameters are the increment, which stores
        the information about the stroke length change per motor step, the initial stroke length, and an offset
        that describes the difference between the linear actuator's pivoting point and the point around which the
        actuator is allowed to pivot. Next, the actuator's pivoting radius is described by the pivot radius and
        lastly, the initial angle indicates the angle that the actuator introduces to the manipulated coordinate
        system at the initial stroke length.

        Parameters
        ----------
        actuator_parameters : torch.Tensor
            The seven actuator parameters.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        super().__init__(actuator_parameters=actuator_parameters, device=device)

    def _motor_positions_to_absolute_angles(
        self, motor_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert motor steps into angles using actuator geometries.

        Calculate absolute angles based solely on the motors' current positions and the geometries
        of the actuators. This gives the angles of the actuators in a global sense. It does not
        consider the starting positions of the motors.

        Parameters
        ----------
        motor_positions : torch.Tensor
            The motor positions.

        Returns
        -------
        torch.Tensor
            The calculated absolute angles.
        """
        stroke_lengths = (
            motor_positions / self.active_actuator_parameters[:, 2]
            + self.active_actuator_parameters[:, 3]
        )
        calc_step_1 = (
            self.active_actuator_parameters[:, 4] ** 2
            + self.active_actuator_parameters[:, 5] ** 2
            - stroke_lengths**2
        )
        calc_step_2 = (
            2.0
            * self.active_actuator_parameters[:, 4]
            * self.active_actuator_parameters[:, 5]
        )
        calc_step_3 = calc_step_1 / calc_step_2
        absolute_angles = torch.arccos(torch.clamp(calc_step_3, min=-1.0, max=1.0))
        return absolute_angles

    def motor_positions_to_angles(
        self, motor_positions: torch.Tensor, device: torch.device | None = None
    ) -> torch.Tensor:
        """
        Calculate the joint angles for given motor positions.

        The absolute angles are adjusted to be relative to the initial angles. This accounts for
        the initial angles and the motors' directions (clockwise or counterclockwise).

        Parameters
        ----------
        motor_positions : torch.Tensor
            The motor positions.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The joint angles corresponding to the motor positions.
        """
        device = get_device(device=device)

        absolute_angles = self._motor_positions_to_absolute_angles(
            motor_positions=motor_positions,
        )
        absolute_initial_angles = self._motor_positions_to_absolute_angles(
            motor_positions=torch.zeros_like(motor_positions, device=device),
        )
        delta_angles = absolute_initial_angles - absolute_angles

        relative_angles = (
            self.active_actuator_parameters[:, 6]
            + delta_angles * (self.active_actuator_parameters[:, 1] == 1)
            - delta_angles * (self.active_actuator_parameters[:, 1] == 0)
        )
        return relative_angles

    def angles_to_motor_positions(
        self, angles: torch.Tensor, device: torch.device | None = None
    ) -> torch.Tensor:
        """
        Calculate the motor positions for given joint angles.

        First the relative angular changes are calculated based on the given angles.
        Then the corresponding stroke lengths are determined using trigonometric
        relationships. These stroke lengths are converted into motor steps.

        Parameters
        ----------
        angles : torch.Tensor
            The joint angles.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The motor steps.
        """
        device = get_device(device=device)

        delta_angles = torch.where(
            self.active_actuator_parameters[:, 1] == 1,
            angles - self.active_actuator_parameters[:, 6],
            self.active_actuator_parameters[:, 6] - angles,
        )

        absolute_initial_angles = self._motor_positions_to_absolute_angles(
            motor_positions=torch.zeros_like(angles, device=device),
        )

        initial_angles = absolute_initial_angles - delta_angles

        calc_step_3 = torch.cos(initial_angles)
        calc_step_2 = (
            2.0
            * self.active_actuator_parameters[:, 4]
            * self.active_actuator_parameters[:, 5]
        )
        calc_step_1 = calc_step_3 * calc_step_2
        stroke_lengths = torch.sqrt(
            self.active_actuator_parameters[:, 4] ** 2
            + self.active_actuator_parameters[:, 5] ** 2
            - calc_step_1
        )
        motor_positions = (
            stroke_lengths - self.active_actuator_parameters[:, 3]
        ) * self.active_actuator_parameters[:, 2]
        return motor_positions