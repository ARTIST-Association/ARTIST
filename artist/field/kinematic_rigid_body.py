import torch

from artist.field.kinematic import Kinematic
from artist.util import config_dictionary, type_mappings, utils
from artist.util.environment_setup import get_device


class RigidBody(Kinematic):
    """
    Implement a rigid body kinematic model.

    Attributes
    ----------
    number_of_heliostats : int
        The number of total heliostats using this rigid body kinematic.
    heliostat_positions : torch.Tensor
        The positions of all heliostats.
        Tensor of shape [number_of_heliostats, 4].
    initial_orientations : torch.Tensor
        The initial orientation offsets of all heliostats.
        Tensor of shape [number_of_heliostats, 4].
    deviation_parameters : torch.Tensor
        The kinematic deviation parameters of all heliostats.
        Tensor of shape [number_of_heliostats, 18].
    number_of_active_heliostats : int
        The number of active heliostats.
    active_heliostat_positions : torch.Tensor
        The positions of all active heliostats.
        Tensor of shape [number_of_active_heliostats, 4].
    active_initial_orientations : torch.Tensor
        The initial orientations of all active heliostats.
        Tensor of shape [number_of_active_heliostats, 4].
    active_deviation_parameters : torch.Tensor
        The deviation parameters of all active heliostats.
        Tensor of shape [number_of_active_heliostats, 18].
    active_motor_positions : torch.Tensor
        The motor positions of active heliostats.
        Tensor of shape [number_of_active_heliostats, 2].
    artist_standard_orientation : torch.Tensor
        The standard orientation of the kinematic.
        Tensor of shape [4].
    actuators : Actuators
        The actuators used in the kinematic.

    Methods
    -------
    incident_ray_directions_to_orientations()
        Compute orientation matrices given incident ray directions.
    motor_positions_to_orientations()
        Compute orientation matrices given the motor positions.

    See Also
    --------
    :class:`Kinematic` : Reference to the parent class.
    """

    def __init__(
        self,
        number_of_heliostats: int,
        heliostat_positions: torch.Tensor,
        initial_orientations: torch.Tensor,
        deviation_parameters: torch.Tensor,
        actuator_parameters: torch.Tensor,
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize a rigid body kinematic.

        The rigid body kinematic determines transformation matrices that are applied to the heliostat surfaces in order to
        align them. The heliostats then reflect the incoming light according to the provided aim points. The rigid body
        kinematic works for heliostats equipped with two actuators that turn the heliostat surfaces.
        Furthermore, initial orientation offsets and deviation parameters determine the specific behavior of the kinematic.

        The kinematic deviations for the rigid body kinematic comprise 18 parameters. The first six parameters refer to
        the first joint, the second six parameters to the second joint, and the final 6 to the concentrator. Within each
        group of six parameters, the first three parameters are the translations in the east, north, and up direction
        respectively, whilst the second three parameters are the tilts in the east, north, and up direction. For example,
        the first six parameters are:

        - ``first_joint_translation_e``
        - ``first_joint_translation_n``
        - ``first_joint_translation_u``
        - ``first_joint_tilt_e``
        - ``first_joint_tilt_n``
        - ``first_joint_tilt_u``

        and this is then repeated for the second joint and the concentrator.

        Parameters
        ----------
        number_of_heliostats : int
            The number of heliostats using this rigid body kinematic.
        heliostat_positions : torch.Tensor
            The positions of all heliostats.
            Tensor of shape [number_of_heliostats, 4].
        initial_orientations : torch.Tensor
            The initial orientation offsets of all heliostats.
            Tensor of shape [number_of_heliostats, 4].
        deviation_parameters : torch.Tensor
            The kinematic deviation parameters of all heliostats.
            Tensor of shape [number_of_heliostats, 18].
        actuator_parameters : torch.Tensor
            The actuator parameters.
            Tensor of shape [number_of_heliostats, n, 2], where n=7 for linear actuators or n=2 for ideal actuators.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        super().__init__()

        device = get_device(device=device)

        self.number_of_heliostats = number_of_heliostats
        self.heliostat_positions = heliostat_positions
        self.initial_orientations = initial_orientations
        self.motor_positions = torch.zeros((number_of_heliostats, 2), device=device)

        self.deviation_parameters = deviation_parameters

        self.number_of_active_heliostats = 0
        self.active_heliostat_positions = torch.empty_like(
            heliostat_positions, device=device
        )
        self.active_initial_orientations = torch.empty_like(
            initial_orientations, device=device
        )
        self.active_deviation_parameters = torch.empty_like(
            deviation_parameters, device=device
        )
        self.active_motor_positions = torch.empty_like(
            self.motor_positions, device=device
        )

        self.artist_standard_orientation = torch.tensor(
            [0.0, -1.0, 0.0, 0.0], device=device
        )

        self.actuators = type_mappings.actuator_type_mapping[
            actuator_parameters[0, 0, 0].item()
        ](actuator_parameters=actuator_parameters, device=device)

    def _compute_orientations_from_motor_positions(
        self,
        motor_positions: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Compute orientation matrices from given motor positions without initial orientation offsets.

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
            The orientation matrices.
            Tensor of shape [number_of_active_heliostats, 4, 4].
        """
        joint_angles = self.actuators.motor_positions_to_angles(
            motor_positions=motor_positions,
            device=device,
        )

        initial_orientations = torch.eye(4, device=device).unsqueeze(0)

        # Account for positions.
        initial_orientations = initial_orientations @ utils.translate_enu(
            e=self.active_heliostat_positions[:, 0],
            n=self.active_heliostat_positions[:, 1],
            u=self.active_heliostat_positions[:, 2],
            device=device,
        )

        joint_rotations = torch.zeros(
            (
                self.number_of_active_heliostats,
                config_dictionary.rigid_body_number_of_actuators,
                4,
                4,
            ),
            device=device,
        )

        joint_rotations[:, 0] = (
            utils.rotate_n(n=self.active_deviation_parameters[:, 4], device=device)
            @ utils.rotate_u(u=self.active_deviation_parameters[:, 5], device=device)
            @ utils.translate_enu(
                e=self.active_deviation_parameters[:, 0],
                n=self.active_deviation_parameters[:, 1],
                u=self.active_deviation_parameters[:, 2],
                device=device,
            )
            @ utils.rotate_e(e=joint_angles[:, 0], device=device)
        )
        joint_rotations[:, 1] = (
            utils.rotate_e(e=self.active_deviation_parameters[:, 9], device=device)
            @ utils.rotate_n(n=self.active_deviation_parameters[:, 10], device=device)
            @ utils.translate_enu(
                e=self.active_deviation_parameters[:, 6],
                n=self.active_deviation_parameters[:, 7],
                u=self.active_deviation_parameters[:, 8],
                device=device,
            )
            @ utils.rotate_u(u=joint_angles[:, 1], device=device)
        )

        orientations = (
            initial_orientations
            @ joint_rotations[:, 0]
            @ joint_rotations[:, 1]
            @ utils.translate_enu(
                e=self.active_deviation_parameters[:, 12],
                n=self.active_deviation_parameters[:, 13],
                u=self.active_deviation_parameters[:, 14],
                device=device,
            )
        )

        return orientations

    def _apply_initial_orientation_offsets(
        self, orientations: torch.Tensor, device: torch.device | None = None
    ) -> torch.Tensor:
        """
        Apply the initial orientation offsets to the given orientation matrices.

        Parameters
        ----------
        orientations : torch.Tensor
            The orientation matrices.
            Tensor of shape [number_of_active_heliostats, 4, 4].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The orientation matrices with the initial orientation offset.
            Tensor of shape [number_of_active_heliostats, 4, 4].
        """
        east_angles, north_angles, up_angles = utils.decompose_rotations(
            initial_vector=self.active_initial_orientations[:, :-1],
            target_vector=self.artist_standard_orientation[:-1],
            device=device,
        )

        orientations_with_initial_orientation_offsets = (
            orientations
            @ utils.rotate_e(e=east_angles, device=device)
            @ utils.rotate_n(n=north_angles, device=device)
            @ utils.rotate_u(u=up_angles, device=device)
        )

        return orientations_with_initial_orientation_offsets

    def incident_ray_directions_to_orientations(
        self,
        incident_ray_directions: torch.Tensor,
        aim_points: torch.Tensor,
        device: torch.device | None = None,
        max_num_iterations: int = 2,
        min_eps: float = 0.0001,
    ) -> torch.Tensor:
        """
        Compute orientation matrices given incident ray directions.

        Parameters
        ----------
        incident_ray_directions : torch.Tensor
            The directions of the incident rays as seen from the heliostats.
            Tensor of shape [number_of_active_heliostats, 4].
        aim_points : torch.Tensor
            The aim points for the active heliostats.
            Tensor of shape [number_of_active_heliostats, 4].
        max_num_iterations : int
            Maximum number of iterations (default is 2).
        min_eps : float
            Convergence criterion (default is 0.0001).
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The orientation matrices.
            Tensor of shape [number_of_active_heliostats, 4, 4].
        """
        device = get_device(device=device)

        motor_positions = torch.zeros(
            (
                self.number_of_active_heliostats,
                config_dictionary.rigid_body_number_of_actuators,
            ),
            device=device,
        )
        last_iteration_loss = None

        for _ in range(max_num_iterations):
            orientations = self._compute_orientations_from_motor_positions(
                motor_positions=motor_positions, device=device
            )

            concentrator_normals = orientations @ torch.tensor(
                [0, -1, 0, 0], dtype=torch.float32, device=device
            )
            concentrator_origins = orientations @ torch.tensor(
                [0, 0, 0, 1], dtype=torch.float32, device=device
            )

            # Compute desired normals.
            desired_reflection_directions = torch.nn.functional.normalize(
                aim_points - concentrator_origins, p=2, dim=1
            )
            desired_concentrator_normals = torch.nn.functional.normalize(
                -incident_ray_directions + desired_reflection_directions, p=2, dim=1
            )

            # Compute loss.
            loss = torch.abs(desired_concentrator_normals - concentrator_normals).mean(
                dim=0
            )

            # Stop if converged.
            if isinstance(last_iteration_loss, torch.Tensor):
                eps = torch.abs(last_iteration_loss - loss)
                if torch.any(eps <= min_eps):
                    break
            last_iteration_loss = loss

            # Analytical solution for joint angles.
            joint_angles_1 = -torch.arcsin(
                torch.clamp(
                    -desired_concentrator_normals[:, 0]
                    / torch.cos(self.active_deviation_parameters[:, 7]),
                    min=-1,
                    max=1,
                )
            )

            a = -torch.cos(self.active_deviation_parameters[:, 6]) * torch.cos(
                joint_angles_1
            ) + torch.sin(self.active_deviation_parameters[:, 6]) * torch.sin(
                self.active_deviation_parameters[:, 7]
            ) * torch.sin(joint_angles_1)
            b = -torch.sin(self.active_deviation_parameters[:, 6]) * torch.cos(
                joint_angles_1
            ) - torch.cos(self.active_deviation_parameters[:, 6]) * torch.sin(
                self.active_deviation_parameters[:, 7]
            ) * torch.sin(joint_angles_1)

            joint_angles_0 = (
                torch.arctan2(
                    a * -desired_concentrator_normals[:, 2]
                    - b * -desired_concentrator_normals[:, 1],
                    a * -desired_concentrator_normals[:, 1]
                    + b * -desired_concentrator_normals[:, 2],
                )
                - torch.pi
            )

            joint_angles = torch.stack(
                [
                    joint_angles_0,
                    joint_angles_1,
                ],
                dim=1,
            )

            motor_positions = self.actuators.angles_to_motor_positions(
                angles=joint_angles, device=device
            )

        orientations_with_initial_orientation_offsets = (
            self._apply_initial_orientation_offsets(
                orientations=orientations, device=device
            )
        )

        self.active_motor_positions = motor_positions

        return orientations_with_initial_orientation_offsets

    def motor_positions_to_orientations(
        self, motor_positions: torch.Tensor, device: torch.device | None = None
    ) -> torch.Tensor:
        """
        Compute orientation matrices given the motor positions.

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
            The orientation matrices.
            Tensor of shape [number_of_active_heliostats, 4, 4].
        """
        device = get_device(device=device)

        orientations = self._compute_orientations_from_motor_positions(
            motor_positions=motor_positions, device=device
        )

        orientations_with_initial_orientation_offsets = (
            self._apply_initial_orientation_offsets(
                orientations=orientations, device=device
            )
        )

        return orientations_with_initial_orientation_offsets
