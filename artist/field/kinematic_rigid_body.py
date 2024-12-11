from typing import Union

import torch

from artist.field.actuator_array import ActuatorArray
from artist.field.kinematic import (
    Kinematic,
)
from artist.util import utils
from artist.util.configuration_classes import (
    ActuatorListConfig,
    KinematicDeviations,
)


class RigidBody(Kinematic):
    """
    Implement a rigid body kinematic model.

    Attributes
    ----------
    deviation_parameters : KinematicDeviations
        18 deviation parameters describing imperfections in the heliostat.
    initial_orientation : torch.Tensor
        The initial orientation-rotation angles of the heliostat.
    actuators : ActuatorArray
        The actuators required for the kinematic.
    artist_standard_orientation : torch.Tensor
        The standard orientation of the kinematic.

    Methods
    -------
    incident_ray_direction_to_orientation()
        Compute the orientation matrix given an incident ray direction.
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
        position: torch.Tensor,
        aim_point: torch.Tensor,
        actuator_config: ActuatorListConfig,
        initial_orientation: torch.Tensor,
        deviation_parameters: KinematicDeviations = KinematicDeviations(
            first_joint_translation_e=torch.tensor(0.0),
            first_joint_translation_n=torch.tensor(0.0),
            first_joint_translation_u=torch.tensor(0.0),
            first_joint_tilt_e=torch.tensor(0.0),
            first_joint_tilt_n=torch.tensor(0.0),
            first_joint_tilt_u=torch.tensor(0.0),
            second_joint_translation_e=torch.tensor(0.0),
            second_joint_translation_n=torch.tensor(0.0),
            second_joint_translation_u=torch.tensor(0.0),
            second_joint_tilt_e=torch.tensor(0.0),
            second_joint_tilt_n=torch.tensor(0.0),
            second_joint_tilt_u=torch.tensor(0.0),
            concentrator_translation_e=torch.tensor(0.0),
            concentrator_translation_n=torch.tensor(0.0),
            concentrator_translation_u=torch.tensor(0.0),
            concentrator_tilt_e=torch.tensor(0.0),
            concentrator_tilt_n=torch.tensor(0.0),
            concentrator_tilt_u=torch.tensor(0.0),
        ),
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """
        Initialize the rigid body kinematic.

        The rigid body kinematic determines a transformation matrix that is applied to the heliostat surface in order to
        align it. The heliostat then reflects the incoming light according to the provided aim point. The kinematic is
        equipped with an actuator array that encompasses one or more actuators that turn the heliostat surface.
        Furthermore, initial orientation offsets and deviation parameters, both for the kinematic, can be provided.

        Parameters
        ----------
        position : torch.Tensor
            The position of the heliostat.
        aim_point : torch.Tensor
            The aim point of the heliostat.
        actuator_config : ActuatorListConfig
            The actuator configuration parameters.
        initial_orientation_offsets : KinematicOffsets
            The initial orientation offsets of the kinematic (default: 0.0 for each possible offset).
        deviation_parameters : KinematicDeviations
            The deviation parameters for the kinematic (default: 0.0 for each deviation).
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).
        """
        super().__init__(position=position, aim_point=aim_point)
        device = torch.device(device)
        self.deviation_parameters = deviation_parameters
        for attr_name, attr_value in self.deviation_parameters.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                setattr(self.deviation_parameters, attr_name, attr_value.to(device))

        self.actuators = ActuatorArray(
            actuator_list_config=actuator_config, device=device
        )
        self.initial_orientation_helisotat = initial_orientation
        self.artist_standard_orientation = torch.tensor(
            [0.0, -1.0, 0.0, 0.0], device=device
        )

    def incident_ray_direction_to_orientation(
        self,
        incident_ray_direction: torch.Tensor,
        max_num_iterations: int = 2,
        min_eps: float = 0.0001,
        device: Union[torch.device, str] = "cuda",
    ) -> torch.Tensor:
        """
        Compute the orientation matrix given an incident ray direction.

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
        if len(self.actuators.actuator_list) != 2:
            raise ValueError(
                f"The rigid body kinematic requires exactly two actuators but {len(self.actuators.actuator_list)} were specified, please check the configuration!"
            )

        device = torch.device(device)
        motor_positions = torch.zeros(2, device=device)
        last_iteration_loss = None
        for _ in range(max_num_iterations):
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

            concentrator_normal = orientation @ torch.tensor(
                [0, -1, 0, 0], dtype=torch.float32, device=device
            )
            concentrator_origin = orientation @ torch.tensor(
                [0, 0, 0, 1], dtype=torch.float32, device=device
            )

            # Compute desired normal.
            desired_reflect_vec = self.aim_point - concentrator_origin
            desired_reflect_vec = desired_reflect_vec / desired_reflect_vec.norm()
            incident_ray_direction = (
                incident_ray_direction / incident_ray_direction.norm()
            )
            desired_concentrator_normal = incident_ray_direction + desired_reflect_vec
            desired_concentrator_normal = (
                desired_concentrator_normal / desired_concentrator_normal.norm()
            )

            # Compute epoch loss.
            loss = torch.abs(desired_concentrator_normal - concentrator_normal)

            # Stop if converged.
            if isinstance(last_iteration_loss, torch.Tensor):
                eps = torch.abs(last_iteration_loss - loss)
                if torch.any(eps <= min_eps):
                    break
            last_iteration_loss = loss.detach()

            # Analytical Solution

            # Calculate joint 2 angle.
            joint_2_angle = -torch.arcsin(
                -desired_concentrator_normal[0]
                / torch.cos(self.deviation_parameters.second_joint_translation_n)
            )

            # Calculate joint 1 angle.
            a = -torch.cos(
                self.deviation_parameters.second_joint_translation_e
            ) * torch.cos(joint_2_angle) + torch.sin(
                self.deviation_parameters.second_joint_translation_e
            ) * torch.sin(
                self.deviation_parameters.second_joint_translation_n
            ) * torch.sin(joint_2_angle)
            b = -torch.sin(
                self.deviation_parameters.second_joint_translation_e
            ) * torch.cos(joint_2_angle) - torch.cos(
                self.deviation_parameters.second_joint_translation_e
            ) * torch.sin(
                self.deviation_parameters.second_joint_translation_n
            ) * torch.sin(joint_2_angle)

            joint_1_angle = (
                torch.arctan2(
                    a * -desired_concentrator_normal[2]
                    - b * -desired_concentrator_normal[1],
                    a * -desired_concentrator_normal[1]
                    + b * -desired_concentrator_normal[2],
                )
                - torch.pi
            )

            motor_positions = torch.stack(
                (
                    self.actuators.actuator_list[0].angle_to_motor_position(
                        joint_1_angle, device
                    ),
                    self.actuators.actuator_list[1].angle_to_motor_position(
                        joint_2_angle, device
                    ),
                ),
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

    def align_surface_with_incident_ray_direction(
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

        orientation = self.incident_ray_direction_to_orientation(
            incident_ray_direction, device=device
        )

        aligned_surface_points = surface_points @ orientation.T
        aligned_surface_normals = surface_normals @ orientation.T

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
