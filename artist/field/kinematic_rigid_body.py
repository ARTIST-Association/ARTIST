from typing import Union

import torch

from artist.field.actuators_ideal import IdealActuators
from artist.field.actuators_linear import LinearActuators
from artist.field.kinematic import Kinematic
from artist.util import config_dictionary, utils


class RigidBody(Kinematic):
    """
    Implement a rigid body kinematic model.

    Attributes
    ----------
    number_of_heliostats : int
        The number of heliostats using this rigid body kinematic.
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
    actuators : Actuators
        The actuators used in the kinematic.

    Methods
    -------
    incident_ray_direction_to_orientations()
        Compute orientation matrices given an incident ray direction.
    align_surfaces_with_incident_ray_direction()
        Align given surface points and surface normals according to an incident ray direction.
    motor_positions_to_orientations()
        Compute orientation matrices given the motor positions..
    align_surfaces_with_motor_positions()
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
        Initialize a rigid body kinematic.

        The rigid body kinematic determines a transformation matrices that are applied to the heliostat surfaces in order to
        align them. The heliostats then reflect the incoming light according to the provided aim points. The rigid body
        kinematic works for heliostats equipped with two actuators that turn the heliostat surfaces.
        Furthermore, initial orientation offsets and deviation parameters determine the specific behavior of the kinematic.

        Parameters
        ----------
        number_of_heliostats : int
            The number of heliostats using this rigid body kinematic.
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

        if actuator_parameters.shape[1] == config_dictionary.number_of_linear_actuator_parameters:
            self.actuators = LinearActuators(
                clockwise_axis_movements=actuator_parameters[:, 0],
                increments=actuator_parameters[:, 1],
                initial_stroke_lengths=actuator_parameters[:, 2],
                offsets=actuator_parameters[:, 3],
                pivot_radii=actuator_parameters[:, 4],
                initial_angles=actuator_parameters[:, 5],
            )
        if actuator_parameters.shape[1] == config_dictionary.number_of_ideal_actuator_parameters:
            self.actuators = IdealActuators(
                clockwise_axis_movements=actuator_parameters[:, 0],
            )

    def incident_ray_direction_to_orientations(
        self,
        incident_ray_direction: torch.Tensor,
        active_heliostats_indices: list[int],
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
            Maximum number of iterations (default is 2).
        min_eps : float
            Convergence criterion (default is 0.0001).
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        torch.Tensor
            The orientation matrices.
        """
        device = torch.device(device)

        motor_positions = torch.zeros(
            (
                len(active_heliostats_indices),
                config_dictionary.rigid_body_number_of_actuators,
            ),
            device=device,
        )
        last_iteration_loss = None
        for _ in range(max_num_iterations):
            joint_angles = self.actuators.motor_positions_to_angles(
                active_heliostats_indices=active_heliostats_indices,
                motor_positions=motor_positions,
                device=device
            )

            initial_orientations = torch.eye(4, device=device).unsqueeze(0)

            # Account for positions.
            initial_orientations = initial_orientations @ utils.translate_enu(
                e=self.heliostat_positions[active_heliostats_indices, 0],
                n=self.heliostat_positions[active_heliostats_indices, 1],
                u=self.heliostat_positions[active_heliostats_indices, 2],
                device=device,
            )

            joint_rotations = torch.zeros(
                (
                    len(active_heliostats_indices),
                    config_dictionary.rigid_body_number_of_actuators,
                    4,
                    4,
                ),
                device=device,
            )

            joint_rotations[:, 0] = (
                utils.rotate_n(n=self.deviation_parameters[active_heliostats_indices, 4], device=device)
                @ utils.rotate_u(u=self.deviation_parameters[active_heliostats_indices, 5], device=device)
                @ utils.translate_enu(
                    e=self.deviation_parameters[active_heliostats_indices, 0],
                    n=self.deviation_parameters[active_heliostats_indices, 1],
                    u=self.deviation_parameters[active_heliostats_indices, 2],
                    device=device,
                )
                @ utils.rotate_e(e=joint_angles[:, 0], device=device)
            )
            joint_rotations[:, 1] = (
                utils.rotate_e(e=self.deviation_parameters[active_heliostats_indices, 9], device=device)
                @ utils.rotate_n(n=self.deviation_parameters[active_heliostats_indices, 10], device=device)
                @ utils.translate_enu(
                    e=self.deviation_parameters[active_heliostats_indices, 6],
                    n=self.deviation_parameters[active_heliostats_indices, 7],
                    u=self.deviation_parameters[active_heliostats_indices, 8],
                    device=device,
                )
                @ utils.rotate_u(u=joint_angles[:, 1], device=device)
            )

            orientations = (
                initial_orientations
                @ joint_rotations[:, 0]
                @ joint_rotations[:, 1]
                @ utils.translate_enu(
                    e=self.deviation_parameters[active_heliostats_indices, 12],
                    n=self.deviation_parameters[active_heliostats_indices, 13],
                    u=self.deviation_parameters[active_heliostats_indices, 14],
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
            desired_reflection_direction = torch.nn.functional.normalize(
                self.aim_points[active_heliostats_indices] - concentrator_origins, p=2, dim=1
            )
            desired_concentrator_normals = torch.nn.functional.normalize(
                -incident_ray_direction + desired_reflection_direction, p=2, dim=1
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
                    len(active_heliostats_indices),
                    config_dictionary.rigid_body_number_of_actuators,
                ),
                device=device,
            )

            # Calculate joint 2 angles.
            joint_angles[:, 1] = -torch.arcsin(
                -desired_concentrator_normals[:, 0]
                / torch.cos(self.deviation_parameters[active_heliostats_indices, 7])
            )

            # Calculate joint 1 angles.
            a = -torch.cos(self.deviation_parameters[active_heliostats_indices, 6]) * torch.cos(
                joint_angles[:, 1].clone()
            ) + torch.sin(self.deviation_parameters[active_heliostats_indices, 6]) * torch.sin(
                self.deviation_parameters[active_heliostats_indices, 7]
            ) * torch.sin(joint_angles[:, 1].clone())
            b = -torch.sin(self.deviation_parameters[active_heliostats_indices, 6]) * torch.cos(
                joint_angles[:, 1].clone()
            ) - torch.cos(self.deviation_parameters[active_heliostats_indices, 6]) * torch.sin(
                self.deviation_parameters[active_heliostats_indices, 7]
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
                active_heliostats_indices=active_heliostats_indices,
                angles=joint_angles,
                device=device
            )

        east_angles, north_angles, up_angles = utils.decompose_rotations(
            initial_vector=self.initial_orientations[active_heliostats_indices, :-1],
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
        active_heliostats_indices: list[int],
        surface_points: torch.Tensor,
        surface_normals: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Align given surface points and surface normals according to an incident ray direction.

        Parameters
        ----------
        incident_ray_direction : torch.Tensor
            The direction of the incident ray as seen from the heliostat.
        surface_points : torch.Tensor
            The points on the surface of the heliostats that reflect the light.
        surface_normals : torch.Tensor
            The normals to the surface points.
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

        orientations = self.incident_ray_direction_to_orientations(
            incident_ray_direction, 
            active_heliostats_indices=active_heliostats_indices,
            device=device
        )

        aligned_surface_points = surface_points @ orientations.transpose(1, 2)
        aligned_surface_normals = surface_normals @ orientations.transpose(1, 2)

        return aligned_surface_points, aligned_surface_normals

    def motor_positions_to_orientations(
        self,
        motor_positions: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> torch.Tensor:
        """
        Compute orientation matrices given the motor positions.

        Parameters
        ----------
        motor_positions : torch.Tensor
            The motor positions.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        torch.Tensor
            The orientation matrices.
        """
        device = torch.device(device)

        joint_angles = self.actuators.motor_positions_to_angles(
            motor_positions=motor_positions, device=device
        )

        initial_orientations = torch.eye(4, device=device)

        # Account for positions.
        initial_orientations = initial_orientations @ utils.translate_enu(
            e=self.heliostat_positions[:, 0],
            n=self.heliostat_positions[:, 1],
            u=self.heliostat_positions[:, 2],
            device=device,
        )

        joint_rotations = torch.zeros(
            (
                self.number_of_heliostats,
                config_dictionary.rigid_body_number_of_actuators,
                4,
                4,
            ),
            device=device,
        )

        joint_rotations[:, 0] = (
            utils.rotate_n(n=self.deviation_parameters[:, 4], device=device)
            @ utils.rotate_u(u=self.deviation_parameters[:, 5], device=device)
            @ utils.translate_enu(
                e=self.deviation_parameters[:, 0],
                n=self.deviation_parameters[:, 1],
                u=self.deviation_parameters[:, 2],
                device=device,
            )
            @ utils.rotate_e(e=joint_angles[:, 0], device=device)
        )
        joint_rotations[:, 1] = (
            utils.rotate_e(e=self.deviation_parameters[:, 9], device=device)
            @ utils.rotate_n(n=self.deviation_parameters[:, 10], device=device)
            @ utils.translate_enu(
                e=self.deviation_parameters[:, 6],
                n=self.deviation_parameters[:, 7],
                u=self.deviation_parameters[:, 8],
                device=device,
            )
            @ utils.rotate_u(u=joint_angles[:, 1], device=device)
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

    def align_surfaces_with_motor_positions(
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
        motor_positions : torch.Tensor
            The motor positions.
        surface_points : torch.Tensor
            The points on the surface of the heliostats that reflect the light.
        surface_normals : torch.Tensor
            The normals to the surface points.
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

        orientations = self.motor_positions_to_orientations(
            motor_positions, device=device
        )

        aligned_surface_points = surface_points @ orientations.transpose(1, 2)
        aligned_surface_normals = surface_normals @ orientations.transpose(1, 2)

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
