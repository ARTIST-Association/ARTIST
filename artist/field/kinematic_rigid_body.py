from typing import Union

from artist.field.actuator_linear import LinearActuators
import torch

from artist.field.kinematic import (
    Kinematic,
)
from artist.util import config_dictionary, utils


class RigidBody(Kinematic):
    """
    Implement a rigid body kinematic model.

    Attributes
    ----------
    number_of_heliostats : int
        The number of heliostats using a rigid body kinematic.
    heliostat_positions : torch.Tensor
        The positions of the heliostats.
    aim_points : torch.Tensor
        The aim points of the heliostats.
    initial_orientations : torch.Tensor
        The initial orientation offsets of the heliostats.
    deviation_parameters : torch.Tensor
        The deviation parameters for the kinematic.
    artist_standard_orientation : torch.Tensor
        The standard orientation of the kinematic.
    actuators : LinearActuators
        The linear actuators of the kinematic.

    Methods
    -------
    incident_ray_direction_to_orientation()
        Compute orientation matrices given an incident ray direction.
    align_surface_with_incident_ray_direction()
        Align given surface points and surface normals according to an incident ray direction.
    motor_positions_to_orientation()
        Compute the orientation matrix given the motor positions.
    align_surface_with_motor_positions()
        Align given surface points and surface normals according to motor positions.
    forward()
        Specify the forward pass.

    See Also
    --------
    :class:`Kinematic` : Reference to the parent class.
    """
    def __init__(
        self,
        number_of_heliostats: int,
        heliostat_positions: torch.Tensor,
        aim_points: torch.Tensor,
        actuator_parameters: torch.Tensor,
        initial_orientations: torch.Tensor,
        deviation_parameters: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """
        Initialize the rigid body kinematic.

        The rigid body kinematic determines a transformation matrix that is applied to the heliostat surfaces in order to
        align them. The heliostats then reflect the incoming light according to the provided aim points. The rigid body 
        kinematic works for heliostats equipped with two actuators that turn the heliostat surfaces.
        Furthermore, initial orientation offsets and deviation parameters determine the specific behavior of the kinematic.

        Parameters
        ----------
        number_of_heliostats : int
            The number of heliostats using a rigid body kinematic.
        heliostat_positions : torch.Tensor
            The positions of the heliostats.
        aim_points : torch.Tensor
            The aim points of the heliostats.
        actuator_parameters : torch.Tensor
            The actuator parameters.
        initial_orientations : torch.Tensor
            The initial orientation offsets of the heliostats.
        deviation_parameters : torch.Tensor
            The deviation parameters for the kinematic.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).
        """
        super().__init__()
        device = torch.device(device)

        self.number_of_heliostats = number_of_heliostats
        self.heliostat_positions = heliostat_positions
        self.aim_points = aim_points
        self.initial_orientations = initial_orientations
        self.deviation_parameters = deviation_parameters

        self.artist_standard_orientation = torch.tensor(
            [0.0, -1.0, 0.0, 0.0], device=device
        )

        # TODO nicht nur Linear erlauben?
        self.actuators = LinearActuators(
            clockwise_axis_movements=actuator_parameters[:, 1],
            increments=actuator_parameters[:, 2],
            initial_stroke_lengths=actuator_parameters[:, 3],
            offsets=actuator_parameters[:, 4],
            pivot_radii=actuator_parameters[:, 5],
            initial_angles=actuator_parameters[:, 6],
        )

    def incident_ray_direction_to_orientation(
        self,
        incident_ray_direction: torch.Tensor,
        max_num_iterations: int = 2,
        min_eps: float = 0.0001,
        device: Union[torch.device, str] = "cuda",
    ) -> torch.Tensor:
        """
        Compute orientation matrices given an incident ray direction.

        Parameters
        ----------
        incident_ray_direction : torch.Tensor
            The direction of the incident ray as seen from the heliostat.
        max_num_iterations : int
            Maximum number of iterations (default: 2).
        min_eps : float
            Convergence criterion (default: 0.0001).
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        torch.Tensor
            The orientation matrix.
        """
        device = torch.device(device)
        motor_positions = torch.zeros((self.number_of_heliostats, config_dictionary.rigid_body_number_of_actuators), device=device)
        last_iteration_loss = None
        for _ in range(max_num_iterations):
            joint_angles = self.actuators.motor_positions_to_angles(
                motor_positions=motor_positions, device=device
            )

            initial_orientations = torch.eye(4, device=device).unsqueeze(0)

            # Account for position.
            initial_orientations = initial_orientations @ utils.translate_enu(
                e=self.heliostat_positions[:, 0],
                n=self.heliostat_positions[:, 1],
                u=self.heliostat_positions[:, 2],
                device=device,
            )

            joint_rotations = torch.zeros((self.number_of_heliostats, config_dictionary.rigid_body_number_of_actuators, 4, 4), device=device)

            joint_rotations[:, 0] = (
                utils.rotate_n(
                    n=self.deviation_parameters[:, 4], device=device
                )
                @ utils.rotate_u(
                    u=self.deviation_parameters[:, 5], device=device
                )
                @ utils.translate_enu(
                    e=self.deviation_parameters[:, 0],
                    n=self.deviation_parameters[:, 1],
                    u=self.deviation_parameters[:, 2],
                    device=device,
                )
                @ utils.rotate_e(
                    e=joint_angles[:, 0], device=device)
            )
            joint_rotations[:, 1] = (
                utils.rotate_e(
                    e=self.deviation_parameters[:, 9], device=device
                )
                @ utils.rotate_n(
                    n=self.deviation_parameters[:, 10], device=device
                )
                @ utils.translate_enu(
                    e=self.deviation_parameters[:, 6],
                    n=self.deviation_parameters[:, 7],
                    u=self.deviation_parameters[:, 8],
                    device=device,
                )
                @ utils.rotate_u(
                    u=joint_angles[:, 1], device=device)
            )

            orientations = (
                initial_orientations
                @ joint_rotations[:, 0]
                @ joint_rotations[:, 1]
                @ utils.translate_enu(
                    e=self.deviation_parameters[:, 12],
                    n=self.deviation_parameters[:, 13],
                    u=self.deviation_parameters[:, 14],
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
            desired_reflect_vecs = torch.nn.functional.normalize(self.aim_points - concentrator_origins, p=2, dim=1)
            desired_concentrator_normals = torch.nn.functional.normalize(-incident_ray_direction + desired_reflect_vecs, p=2, dim=1)

            # Compute epoch loss.
            loss = torch.abs(desired_concentrator_normals - concentrator_normals).mean(dim=0)

            # Stop if converged.
            if isinstance(last_iteration_loss, torch.Tensor):
                eps = torch.abs(last_iteration_loss - loss)
                if torch.any(eps <= min_eps):
                    break
            last_iteration_loss = loss

            # Analytical Solution
            joint_angles = torch.zeros((self.number_of_heliostats, config_dictionary.rigid_body_number_of_actuators), device=device)

            # Calculate joint 2 angles.
            joint_angles[:, 1] = -torch.arcsin(
                -desired_concentrator_normals[:, 0]
                / torch.cos(self.deviation_parameters[:, 7])
            )

            # Calculate joint 1 angles.
            a = -torch.cos(
                self.deviation_parameters[:, 6]
            ) * torch.cos(joint_angles[:, 1]) + torch.sin(
                self.deviation_parameters[:, 6]
            ) * torch.sin(
                self.deviation_parameters[:, 7]
            ) * torch.sin(joint_angles[:, 1])
            b = -torch.sin(
                self.deviation_parameters[:, 6]
            ) * torch.cos(joint_angles[:, 1]) - torch.cos(
                self.deviation_parameters[:, 6]
            ) * torch.sin(
                self.deviation_parameters[:, 7]
            ) * torch.sin(joint_angles[:, 1])

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
                joint_angles, device
            )

        east_angles, north_angles, up_angles = utils.decompose_rotations(
            initial_vector=self.initial_orientations[:, :-1],
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

    def align_surfaces_with_incident_ray_direction(
        self,
        incident_ray_direction: torch.Tensor,
        surface_points: torch.Tensor,
        surface_normals: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Align given surface points and surface normals according to an incident ray direction.

        Parameters
        ----------
        incident_ray_direction : torch.Tensor
            The direction of the rays.
        surface_points : torch.Tensor
            Points on the surface of the heliostats that reflect the light.
        surface_normals : torch.Tensor
            Normals to the surface points.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        torch.Tensor
            The aligned surface points.
        torch.Tensor
            The aligned surface normals.
        """
        device = torch.device(device)

        orientation = self.incident_ray_direction_to_orientation(
            incident_ray_direction, device=device
        )

        aligned_surface_points = surface_points @ orientation.transpose(1, 2)
        aligned_surface_normals = surface_normals @ orientation.transpose(1, 2)

        return aligned_surface_points, aligned_surface_normals

    def motor_positions_to_orientation(
        self,
        motor_positions: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> torch.Tensor:
        """
        Compute the orientation matrix given the motor positions.

        Parameters
        ----------
        motor_positions : torch.Tensor
            The motor positions from the calibration.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        torch.Tensor
            The orientation matrix.
        """
        if len(self.actuators.actuator_list) != 2:
            raise ValueError(
                f"The rigid body kinematic requires exactly two actuators but {len(self.actuators.actuator_list)} were specified, please check the configuration!"
            )

        device = torch.device(device)

        joint_1_angle = self.actuators.actuator_list[0].motor_position_to_angle(
            motor_position=motor_positions[0], device=device
        )
        joint_2_angle = self.actuators.actuator_list[1].motor_position_to_angle(
            motor_position=motor_positions[1], device=device
        )

        initial_orientation = torch.eye(4, device=device)

        # Account for position.
        initial_orientation = initial_orientation @ utils.translate_enu(
            e=self.position[0],
            n=self.position[1],
            u=self.position[2],
            device=device,
        )

        joint_1_rotation = (
            utils.rotate_n(
                n=self.deviation_parameters.first_joint_tilt_n, device=device
            )
            @ utils.rotate_u(
                u=self.deviation_parameters.first_joint_tilt_u, device=device
            )
            @ utils.translate_enu(
                e=self.deviation_parameters.first_joint_translation_e,
                n=self.deviation_parameters.first_joint_translation_n,
                u=self.deviation_parameters.first_joint_translation_u,
                device=device,
            )
            @ utils.rotate_e(joint_1_angle, device=device)
        )
        joint_2_rotation = (
            utils.rotate_e(
                e=self.deviation_parameters.second_joint_tilt_e, device=device
            )
            @ utils.rotate_n(
                n=self.deviation_parameters.second_joint_tilt_n, device=device
            )
            @ utils.translate_enu(
                e=self.deviation_parameters.second_joint_translation_e,
                n=self.deviation_parameters.second_joint_translation_n,
                u=self.deviation_parameters.second_joint_translation_u,
                device=device,
            )
            @ utils.rotate_u(joint_2_angle, device=device)
        )

        orientation = (
            initial_orientation
            @ joint_1_rotation
            @ joint_2_rotation
            @ utils.translate_enu(
                e=self.deviation_parameters.concentrator_translation_e,
                n=self.deviation_parameters.concentrator_translation_n,
                u=self.deviation_parameters.concentrator_translation_u,
                device=device,
            )
        )

        east_angle, north_angle, up_angle = utils.decompose_rotation(
            initial_vector=self.initial_orientation_helisotat[:-1],
            target_vector=self.artist_standard_orientation[:-1],
            device=device,
        )

        # Return orientation matrix multiplied by the initial orientation offset.
        return (
            orientation
            @ utils.rotate_e(
                e=east_angle,
                device=device,
            )
            @ utils.rotate_n(
                n=north_angle,
                device=device,
            )
            @ utils.rotate_u(
                u=up_angle,
                device=device,
            )
        )

    def align_surface_with_motor_positions(
        self,
        motor_positions: torch.Tensor,
        surface_points: torch.Tensor,
        surface_normals: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Align given surface points and surface normals according to motor positions.

        Parameters
        ----------
        incident_ray_direction : torch.Tensor
            The direction of the rays.
        surface_points : torch.Tensor
            Points on the surface of the heliostat that reflect the light.
        surface_normals : torch.Tensor
            Normals to the surface points.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        torch.Tensor
            The aligned surface points.
        torch.Tensor
            The aligned surface normals.
        """
        device = torch.device(device)

        orientation = self.motor_positions_to_orientation(
            motor_positions, device=device
        )

        aligned_surface_points = surface_points @ orientation.T
        aligned_surface_normals = surface_normals @ orientation.T

        return aligned_surface_points, aligned_surface_normals

    def forward(self) -> None:
        """
        Specify the forward pass.

        Raises
        ------
        NotImplementedError
            Whenever called.
        """
        raise NotImplementedError("Not Implemented!")
