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
    initial_orientations : torch.Tensor
        The initial orientation offsets of all heliostats.
    deviation_parameters : torch.Tensor
        The kinematic deviation parameters of all heliostats.
    number_of_active_heliostats : int
        The number of active heliostats.
    active_heliostat_positions : torch.Tensor
        The positions of all active heliostats.
    active_initial_orientations : torch.Tensor
        The initial orientations of all active heliostats
    active_deviation_parameters : torch.Tensor
        The deviation parameters of all active heliostats.
    artist_standard_orientation : torch.Tensor
        The standard orientation of the kinematic.
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

        Parameters
        ----------
        number_of_heliostats : int
            The number of heliostats using this rigid body kinematic.
        heliostat_positions : torch.Tensor
            The positions of the heliostats.
        initial_orientations : torch.Tensor
            The initial orientation offsets of the heliostats.
        deviation_parameters : torch.Tensor
            The deviation parameters for the kinematic.
        actuator_parameters : torch.Tensor
            The actuator parameters.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        super().__init__()

        device = get_device(device=device)

        self.number_of_heliostats = number_of_heliostats
        self.heliostat_positions = heliostat_positions
        self.initial_orientations = initial_orientations

        #self.deviation_parameters = torch.nn.Parameter(deviation_parameters)
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
        self.artist_standard_orientation = torch.tensor(
            [0.0, -1.0, 0.0, 0.0], device=device
        )

        self.actuators = type_mappings.actuator_type_mapping[
            actuator_parameters[0, 0, 0].item()
        ](actuator_parameters=actuator_parameters, device=device)

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
        aim_points : torch.Tensor
            The aim points for the active heliostats.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        max_num_iterations : int
            Maximum number of iterations (default is 2).
        min_eps : float
            Convergence criterion (default is 0.0001).

        Returns
        -------
        torch.Tensor
            The orientation matrices.
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
                utils.rotate_n(
                    n=self.active_deviation_parameters[:, 4],
                    device=device,
                )
                @ utils.rotate_u(
                    u=self.active_deviation_parameters[:, 5],
                    device=device,
                )
                @ utils.translate_enu(
                    e=self.active_deviation_parameters[:, 0],
                    n=self.active_deviation_parameters[:, 1],
                    u=self.active_deviation_parameters[:, 2],
                    device=device,
                )
                @ utils.rotate_e(e=joint_angles[:, 0], device=device)
            )
            joint_rotations[:, 1] = (
                utils.rotate_e(
                    e=self.active_deviation_parameters[:, 9],
                    device=device,
                )
                @ utils.rotate_n(
                    n=self.active_deviation_parameters[:, 10],
                    device=device,
                )
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

            concentrator_normals = orientations @ torch.tensor(
                [0, -1, 0, 0], dtype=torch.float32, device=device
            )
            concentrator_origins = orientations @ torch.tensor(
                [0, 0, 0, 1], dtype=torch.float32, device=device
            )

            # Compute desired normals.
            desired_reflection_directions = torch.nn.functional.normalize(
                aim_points - concentrator_origins,
                p=2,
                dim=1,
            )
            desired_concentrator_normals = torch.nn.functional.normalize(
                -incident_ray_directions + desired_reflection_directions, p=2, dim=1
            )

            # Compute epoch loss.
            loss = torch.abs(desired_concentrator_normals - concentrator_normals).mean(
                dim=0
            )

            # Stop if converged.
            if isinstance(last_iteration_loss, torch.Tensor):
                eps = torch.abs(last_iteration_loss - loss)
                if torch.any(eps <= min_eps):
                    break
            last_iteration_loss = loss

            # Analytical Solution
            joint_angles = torch.zeros(
                (
                    self.number_of_active_heliostats,
                    config_dictionary.rigid_body_number_of_actuators,
                ),
                device=device,
            )

            # Calculate joint 2 angles.
            joint_angles[:, 1] = -torch.arcsin(
                -desired_concentrator_normals[:, 0]
                / torch.cos(self.active_deviation_parameters[:, 7])
            )

            # Calculate joint 1 angles.
            a = -torch.cos(self.active_deviation_parameters[:, 6]) * torch.cos(
                joint_angles[:, 1].clone()
            ) + torch.sin(self.active_deviation_parameters[:, 6]) * torch.sin(
                self.active_deviation_parameters[:, 7]
            ) * torch.sin(joint_angles[:, 1].clone())
            b = -torch.sin(self.active_deviation_parameters[:, 6]) * torch.cos(
                joint_angles[:, 1].clone()
            ) - torch.cos(self.active_deviation_parameters[:, 6]) * torch.sin(
                self.active_deviation_parameters[:, 7]
            ) * torch.sin(joint_angles[:, 1].clone())

            joint_angles[:, 0] = (
                torch.arctan2(
                    a * -desired_concentrator_normals[:, 2]
                    - b * -desired_concentrator_normals[:, 1],
                    a * -desired_concentrator_normals[:, 1]
                    + b * -desired_concentrator_normals[:, 2],
                )
                - torch.pi
            )

            motor_positions = self.actuators.angles_to_motor_positions(
                angles=joint_angles,
                device=device,
            )

        east_angles, north_angles, up_angles = utils.decompose_rotations(
            initial_vector=self.active_initial_orientations[:, :-1],
            target_vector=self.artist_standard_orientation[:-1],
            device=device,
        )

        # Return orientation matrices multiplied by the initial orientation offsets.
        return (
            orientations
            @ utils.rotate_e(
                e=east_angles,
                device=device,
            )
            @ utils.rotate_n(
                n=north_angles,
                device=device,
            )
            @ utils.rotate_u(
                u=up_angles,
                device=device,
            )
        )

    def motor_positions_to_orientations(
        self, motor_positions: torch.Tensor, device: torch.device | None = None
    ) -> torch.Tensor:
        """
        Compute orientation matrices given the motor positions.

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
            The orientation matrices.
        """
        device = get_device(device=device)

        joint_angles = self.actuators.motor_positions_to_angles(
            motor_positions=motor_positions,
            device=device,
        )

        initial_orientations = torch.eye(4, device=device)

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
            @ utils.rotate_n(
                n=self.active_deviation_parameters[:, 10],
                device=device,
            )
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

        east_angles, north_angles, up_angles = utils.decompose_rotations(
            initial_vector=self.active_initial_orientations[:, :-1],
            target_vector=self.artist_standard_orientation[:-1],
            device=device,
        )

        # Return orientation matrices multiplied by the initial orientation offsets.
        return (
            orientations
            @ utils.rotate_e(
                e=east_angles,
                device=device,
            )
            @ utils.rotate_n(
                n=north_angles,
                device=device,
            )
            @ utils.rotate_u(
                u=up_angles,
                device=device,
            )
        )
