import torch

from artist.field.actuators import Actuators
from artist.util.environment_setup import get_device


class LinearActuators(Actuators):
    """
    Implement the behavior of linear actuators.

    Attributes
    ----------
    geometry_parameters : torch.Tensor
        Parameters concerning the actuator geometry.
        Tensor of shape [number_of_heliostats, 7, 2].
    initial_parameters : torch.Tensor
        Parameters concerning the initial actuator configuration.
        Tensor of shape [number_of_heliostats, 2, 2].
    active_geometry_parameters : torch.Tensor
        Active geometry parameters.
        Tensor of shape [number_of_active_heliostats, 7, 2].
    active_geometry_parameters : torch.Tensor
        Active initial parameters.
        Tensor of shape [number_of_active_heliostats, 2, 2].

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

        A linear actuator describes movement within a 2D plane. One linear actuator has nine parameters.
        Ordered by index, the first parameter describes the type of the actuator, i.e., linear, the second parameter
        describes the turning direction of the actuator. The third and fourth parameters are the minimum and
        maximum motor positions. The next five parameters are the increment, which stores
        the information about the stroke length change per motor step, the initial stroke length, and an offset
        that describes the difference between the linear actuator's pivoting point and the point around which the
        actuator is allowed to pivot. Next, the actuator's pivoting radius is described by the pivot radius and
        lastly, the initial angle indicates the angle that the actuator introduces to the manipulated coordinate
        system at the initial stroke length.

        Parameters
        ----------
        actuator_parameters : torch.Tensor
            The nine actuator parameters.
            Tensor of shape [number_of_heliostats, 9, 2].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        super().__init__(actuator_parameters=actuator_parameters, device=device)
        self.epsilon = 1e-6

        self.geometry_parameters = actuator_parameters[:, :7]
        self.initial_parameters = actuator_parameters[:, -2:]

        self.active_geometry_parameters = torch.empty_like(
            self.geometry_parameters, device=device
        )
        self.active_initial_parameters = torch.empty_like(
            self.initial_parameters, device=device
        )

    def _physics_informed_parameters(
        self, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Limit actuator parameters to their physically valid ranges.

        The first four parameters (types, turning directions, min and max increments) are not learnable and
        do not need to be physics informed. The parameters increment, initial stroke lengths, offsets and
        pivot radii are defined to be strictly positive.

        Parameters
        ----------
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The physics-informed geometry parameters.
            Tensor of shape [number_of_active_heliostats, 7, 2].
        torch.Tensor
            The physics-informed initial parameters.
            Tensor of shape [number_of_active_heliostats, 2, 2].
        """
        device = get_device(device=device)

        geometry_parameters = self.active_geometry_parameters
        initial_parameters = self.active_initial_parameters

        physics_informed_geometry_parameters = torch.empty_like(
            self.active_geometry_parameters, device=device
        )
        physics_informed_initial_parameters = torch.empty_like(
            self.active_initial_parameters, device=device
        )

        physics_informed_geometry_parameters[:, [0, 1, 2, 3]] = geometry_parameters[
            :, [0, 1, 2, 3]
        ]
        physics_informed_initial_parameters[:, 0] = initial_parameters[:, 0]

        # Strictly positive parameters.
        # Increment.
        physics_informed_geometry_parameters[:, 4] = (
            torch.nn.functional.softplus(geometry_parameters[:, 4], beta=100)
            + self.epsilon
        )
        # Offset.
        physics_informed_geometry_parameters[:, 5] = (
            torch.nn.functional.softplus(geometry_parameters[:, 5], beta=100)
            + self.epsilon
        )
        # Pivot radius.
        physics_informed_geometry_parameters[:, 6] = (
            torch.nn.functional.softplus(geometry_parameters[:, 6], beta=100)
            + self.epsilon
        )
        # Initial stroke length.
        physics_informed_initial_parameters[:, 1] = (
            torch.nn.functional.softplus(initial_parameters[:, 1], beta=100)
            + self.epsilon
        )

        return physics_informed_geometry_parameters, physics_informed_initial_parameters

    def _motor_positions_to_absolute_angles(
        self, motor_positions: torch.Tensor, device: torch.device | None = None
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
            Tensor of shape [number_of_active_heliostats, 2].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The calculated absolute angles.
            Tensor of shape [number_of_active_heliostats, 2].
        """
        device = get_device(device=device)

        geometry_parameters, initial_parameters = self._physics_informed_parameters(
            device=device
        )
        increment, offsets, pivot_radii, initial_stroke_lengths = (
            geometry_parameters[:, 4],
            geometry_parameters[:, 5],
            geometry_parameters[:, 6],
            initial_parameters[:, 1],
        )

        stroke_lengths = motor_positions / increment + initial_stroke_lengths

        min_stroke_lengths = (offsets - pivot_radii).abs() + self.epsilon
        max_stroke_lengths = offsets + pivot_radii - self.epsilon
        stroke_lengths = torch.clamp(
            stroke_lengths, min=min_stroke_lengths, max=max_stroke_lengths
        )

        numerator = offsets**2 + pivot_radii**2 - stroke_lengths**2
        denominator = 2.0 * offsets * pivot_radii
        division = numerator / denominator

        absolute_angles = torch.arccos(
            torch.clamp(division, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        )
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
            Tensor of shape [number_of_active_heliostats, 2].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The joint angles corresponding to the motor positions.
            Tensor of shape [number_of_active_heliostats, 2].
        """
        device = get_device(device=device)

        _, initial_parameters = self._physics_informed_parameters(device=device)
        initial_angles = initial_parameters[:, 0]

        absolute_angles = self._motor_positions_to_absolute_angles(
            motor_positions=motor_positions, device=device
        )
        absolute_initial_angles = self._motor_positions_to_absolute_angles(
            motor_positions=torch.zeros_like(motor_positions, device=device),
            device=device,
        )
        delta_angles = absolute_initial_angles - absolute_angles

        relative_angles = (
            initial_angles
            + delta_angles * (self.active_geometry_parameters[:, 1] == 1)
            - delta_angles * (self.active_geometry_parameters[:, 1] == 0)
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
            Tensor of shape [number_of_active_heliostats, 2].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The motor steps.
            Tensor of shape [number_of_active_heliostats, 2].
        """
        device = get_device(device=device)

        geometry_parameters, initial_parameters = self._physics_informed_parameters(
            device=device
        )
        (
            increment,
            offsets,
            pivot_radii,
            initial_delta_angles,
            initial_stroke_lengths,
        ) = (
            geometry_parameters[:, 4],
            geometry_parameters[:, 5],
            geometry_parameters[:, 6],
            initial_parameters[:, 0],
            initial_parameters[:, 1],
        )

        delta_angles = torch.where(
            self.active_geometry_parameters[:, 1] == 1,
            angles - initial_delta_angles,
            initial_delta_angles - angles,
        )

        absolute_initial_angles = self._motor_positions_to_absolute_angles(
            motor_positions=torch.zeros_like(angles, device=device), device=device
        )

        initial_angles = absolute_initial_angles - delta_angles

        cos_initial_angles = torch.clamp(
            torch.cos(initial_angles), -1.0 + 1e-6, 1.0 - 1e-6
        )

        stroke_lengths = torch.sqrt(
            offsets**2
            + pivot_radii**2
            - 2.0 * offsets * pivot_radii * cos_initial_angles
        )

        min_stroke_lengths = (offsets - pivot_radii).abs() + self.epsilon
        max_stroke_lengths = offsets + pivot_radii - self.epsilon
        stroke_lengths = torch.clamp(
            stroke_lengths, min=min_stroke_lengths, max=max_stroke_lengths
        )

        motor_positions = (stroke_lengths - initial_stroke_lengths) * increment
        return motor_positions
