import logging
import torch

from artist.field.actuators import Actuators
from artist.field.kinematics import Kinematics
from artist.geometry import coordinates, rotations, transforms
from artist.util import constants, indices, type_registry
from artist.util.env import get_device

log = logging.getLogger(__name__)
"""A logger for the rigid body kinematics."""

class RigidBody(Kinematics):
    """
    Implement a rigid body kinematics model.

    Attributes
    ----------
    number_of_heliostats : int
        The number of total heliostats using this rigid body kinematics.
    heliostat_positions : torch.Tensor
        The positions of all heliostats.
        Shape is ``[number_of_heliostats, 4]``.
    initial_orientations : torch.Tensor
        The initial orientation offsets of all heliostats.
        Shape is ``[number_of_heliostats, 4]``.
    translation_deviation_parameters : torch.Tensor
        Kinematics translation deviation parameter.
        Shape is ``[number_of_heliostats, 9]``.
    rotation_deviation_parameters : torch.Tensor
        Kinematics rotation deviation parameter.
        Shape is ``[number_of_heliostats, 4]``.
    number_of_active_heliostats : int
        The number of active heliostats.
    active_heliostat_positions : torch.Tensor
        The positions of all active heliostats.
        Shape is ``[number_of_active_heliostats, 4]``.
    active_initial_orientations : torch.Tensor
        The initial orientations of all active heliostats.
        Shape is ``[number_of_active_heliostats, 4]``.
    translation_deviation_parameters : torch.Tensor
        Kinematics translation deviation parameter of all active heliostats.
        Shape is ``[number_of_active_heliostats, 9]``.
    rotation_deviation_parameters : torch.Tensor
        Kinematics rotation deviation parameter of all active heliostats.
        Shape is ``[number_of_active_heliostats, 4]``.
    active_motor_positions : torch.Tensor
        The motor positions of active heliostats.
        Shape is ``[number_of_active_heliostats, 2]``.
    actuators : Actuators
        The actuators used in the kinematics.
    kinematics_standard_orientation : torch.Tensor
        Standard orientation of the kinematics system: south (0, -1, 0, 0) in homogeneous ENU.
        Shape is ``[4]``
    initial_orientation_offsets : torch.Tensor
        Rotation matrix to account for the initial orientation offset between surface mesh and kinematics system. 
        Shape is ``[1, 4, 4]``.

    Methods
    -------
    incident_ray_directions_to_orientations()
        Compute orientation matrices given incident ray directions.
    motor_positions_to_orientations()
        Compute orientation matrices given the motor positions.

    See Also
    --------
    :class:`Kinematics` : Reference to the parent class.
    """

    def __init__(
        self,
        number_of_heliostats: int,
        heliostat_positions: torch.Tensor,
        initial_orientations: torch.Tensor,
        translation_deviation_parameters: torch.Tensor,
        rotation_deviation_parameters: torch.Tensor,
        actuator_parameters_non_optimizable: torch.Tensor,
        actuator_parameters_optimizable: torch.Tensor = torch.tensor([]),
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize a rigid body kinematics.

        The rigid body kinematics determines transformation matrices that are applied to the heliostat surfaces in order to
        align them. The heliostats then reflect the incoming light according to the provided aim points. The rigid body
        kinematics works for heliostats equipped with two actuators that turn the heliostat surfaces.
        Furthermore, initial orientation offsets and deviation parameters determine the specific behavior of the kinematics.

        The kinematics deviations are split into translation and rotation parameters. There are three translation parameters
        for each joint and for the concentrator. One translation deviation in the east, north and up direction respectively.
        For joint one and two there are also rotation deviations. For joint one in the north and up direction and for joint
        two in the east and north direction.

        Parameters
        ----------
        number_of_heliostats : int
            The number of heliostats using this rigid body kinematics.
        heliostat_positions : torch.Tensor
            The positions of all heliostats.
            Shape is ``[number_of_heliostats, 4]``.
        initial_orientations : torch.Tensor
            The initial orientation offsets of all heliostats.
            Shape is ``[number_of_heliostats, 4]``.
        translation_deviation_parameters : torch.Tensor
            Kinematics translation deviation parameter.
            Shape is ``[number_of_heliostats, 9]``.
        rotation_deviation_parameters : torch.Tensor
            Kinematics rotation deviation parameter.
            Shape is ``[number_of_heliostats, 4]``.
        actuator_parameters_non_optimizable : torch.Tensor
            The non-optimizable actuator parameters.
            Shape is ``[number_of_heliostats, 7, 2]`` for linear actuators
            or ``[number_of_heliostats, 4, 2]`` for ideal actuators.
        actuator_parameters_optimizable : torch.Tensor
            The optimizable actuator parameters.
            Shape is ``[number_of_heliostats, 2, 2]`` for linear actuators
            or ``[]`` for ideal actuators (default is ``torch.tensor([])``).
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
        self.motor_positions = torch.zeros(
            (
                number_of_heliostats,
                indices.rigid_body_motor_position_dimension,
            ),
            device=device,
        )

        self.translation_deviation_parameters = translation_deviation_parameters
        self.rotation_deviation_parameters = rotation_deviation_parameters

        self.number_of_active_heliostats = 0
        self.active_heliostat_positions = torch.empty_like(
            heliostat_positions, device=device
        )
        self.active_initial_orientations = torch.empty_like(
            initial_orientations, device=device
        )
        self.active_translation_deviation_parameters = torch.empty_like(
            translation_deviation_parameters, device=device
        )
        self.active_rotation_deviation_parameters = torch.empty_like(
            rotation_deviation_parameters, device=device
        )
        self.active_motor_positions = torch.empty_like(
            self.motor_positions, device=device
        )

        self.actuators: Actuators = type_registry.actuator_type_mapping[
            actuator_parameters_non_optimizable[
                0, indices.actuator_type, indices.actuator_one_index
            ].item()
        ](
            non_optimizable_parameters=actuator_parameters_non_optimizable,
            optimizable_parameters=actuator_parameters_optimizable.to(device),
            device=device,
        )

        # The surface points and normals are always sampled from a model (converted NURBS from deflectometry or ideal NURBS) that lays
        # flat on the ground, i.e., the surface normals are pointing upwards [0.0, 0.0, 1.0]. Since the kinematics in ARTIST expects the 
        # points and normals to be initially oriented to the south, an extra rotation needs to be applied.
        self.kinematics_standard_orientation = torch.tensor(
            [0.0, -1.0, 0.0, 0.0], device=device
        )
        sampled_surface_orientation = torch.tensor(
            [0.0, 0.0, 1.0, 0.0], device=device
        )
        east_angles, north_angles, up_angles = rotations.decompose_rotations(
            initial_vector=sampled_surface_orientation[None, :],
            target_vector=self.kinematics_standard_orientation,
        )
        self.initial_orientation_offsets = (
            transforms.rotate_e(e=east_angles, device=device)
            @ transforms.rotate_n(n=north_angles, device=device)
            @ transforms.rotate_u(u=up_angles, device=device)
        )


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
            Shape is ``[number_of_active_heliostats, 2]``.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The orientation matrices.
            Shape is ``[number_of_active_heliostats, 4, 4]``.
        """
        device = get_device(device=device)

        joint_angles = self.actuators.motor_positions_to_angles(
            motor_positions=motor_positions,
            device=device,
        )

        initial_orientations = torch.eye(4, device=device).unsqueeze(0)

        # Account for positions.
        initial_orientations = initial_orientations @ transforms.translate_enu(
            e=self.active_heliostat_positions[:, indices.e],
            n=self.active_heliostat_positions[:, indices.n],
            u=self.active_heliostat_positions[:, indices.u],
            device=device,
        )

        joint_rotations = torch.zeros(
            (
                self.number_of_active_heliostats,
                constants.rigid_body_number_of_actuators,
                4,
                4,
            ),
            device=device,
        )

        joint_rotations[:, indices.first_joint_index] = (
            transforms.rotate_n(
                n=self.active_rotation_deviation_parameters[
                    :, indices.first_joint_tilt_n
                ],
                device=device,
            )
            @ transforms.rotate_u(
                u=self.active_rotation_deviation_parameters[
                    :, indices.first_joint_tilt_u
                ],
                device=device,
            )
            @ transforms.translate_enu(
                e=self.active_translation_deviation_parameters[
                    :, indices.first_joint_translation_e
                ],
                n=self.active_translation_deviation_parameters[
                    :, indices.first_joint_translation_n
                ],
                u=self.active_translation_deviation_parameters[
                    :, indices.first_joint_translation_u
                ],
                device=device,
            )
            @ transforms.rotate_e(
                e=joint_angles[:, indices.joint_angles_e], device=device
            )
        )
        joint_rotations[:, indices.second_joint_index] = (
            transforms.rotate_e(
                e=self.active_rotation_deviation_parameters[
                    :, indices.second_joint_tilt_e
                ],
                device=device,
            )
            @ transforms.rotate_n(
                n=self.active_rotation_deviation_parameters[
                    :, indices.second_joint_tilt_n
                ],
                device=device,
            )
            @ transforms.translate_enu(
                e=self.active_translation_deviation_parameters[
                    :, indices.second_joint_translation_e
                ],
                n=self.active_translation_deviation_parameters[
                    :, indices.second_joint_translation_n
                ],
                u=self.active_translation_deviation_parameters[
                    :, indices.second_joint_translation_u
                ],
                device=device,
            )
            @ transforms.rotate_u(
                u=joint_angles[:, indices.joint_angles_u], device=device
            )
        )

        orientations = (
            initial_orientations
            @ joint_rotations[:, indices.first_joint_index]
            @ joint_rotations[:, indices.second_joint_index]
            @ transforms.translate_enu(
                e=self.active_translation_deviation_parameters[
                    :, indices.concentrator_translation_e
                ],
                n=self.active_translation_deviation_parameters[
                    :, indices.concentrator_translation_n
                ],
                u=self.active_translation_deviation_parameters[
                    :, indices.concentrator_translation_u
                ],
                device=device,
            )
        )

        return orientations

    def _compute_motor_positions_from_normal(
        self,
        normals: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Compute the motor positions from a normal vector.
        
        This is the inverse kinematics. First the joint angles are computed from the desired normal vector.
        Then the motor positions are computed from the joint angles.
        The inverse kinematics produces two solutions, the valid solution is chosen according to resulting
        motor positions which must lie within the minimum and maximum allowed motor positions defined within 
        the actuator parameters. 

        Parameters
        ----------
        normals : torch.Tensor
            Concentrator normals.
            Shape is ``[number_of_active_heliostats, 4]``.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The joint angles.
            Shape is ``[number_of_active_heliostats, 2]``.
        """
        device = get_device(device=device)

        first_rotation_axis_deviations = (
            transforms.rotate_n(
                n=self.active_rotation_deviation_parameters[
                    :, indices.first_joint_tilt_n
                ],
                device=device,
            )
            @ transforms.rotate_u(
                u=self.active_rotation_deviation_parameters[
                    :, indices.first_joint_tilt_u
                ],
                device=device,
            )
        )

        second_rotation_axis_deviations = (
            transforms.rotate_e(
                e=self.active_rotation_deviation_parameters[
                    :, indices.second_joint_tilt_e
                ],
                device=device,
            )
            @ transforms.rotate_n(
                n=self.active_rotation_deviation_parameters[
                    :, indices.second_joint_tilt_n
                ],
                device=device,
            )
        )

        normal_after_first_deviation = first_rotation_axis_deviations.transpose(-1, -2) @ normals[:, :, None]

        # Determine second joint angles.
        second_axis_deviation_00 = second_rotation_axis_deviations[:, 0, 0]
        second_axis_deviation_01 = second_rotation_axis_deviations[:, 0, 1]

        denominator = torch.sqrt(second_axis_deviation_00 ** 2 + second_axis_deviation_01 ** 2)[:, None]
        phi = torch.atan2(-second_axis_deviation_01, second_axis_deviation_00)[:, None]

        ratio = torch.clamp((normal_after_first_deviation[:, 0] / (denominator + 1e-12)), -1.0 + 1e-7, 1.0 - 1e-7)
        second_joint_angle_1 = torch.arcsin(ratio) - phi
        second_joint_angle_2 = torch.pi - torch.arcsin(ratio) - phi

        second_joint_angle_1 = torch.atan2(torch.sin(second_joint_angle_1), torch.cos(second_joint_angle_1))
        second_joint_angle_2 = torch.atan2(torch.sin(second_joint_angle_2), torch.cos(second_joint_angle_2))

        # Determine first joint angles. There are two valid solutions.
        v_1 = second_rotation_axis_deviations @ transforms.rotate_u(second_joint_angle_1[:, 0], device=device) @ self.kinematics_standard_orientation
        first_joint_angle_1 = torch.atan2(
            v_1[:, 1] * normal_after_first_deviation[:, 2, 0] - v_1[:, 2] * normal_after_first_deviation[:, 1, 0],
            v_1[:, 1] * normal_after_first_deviation[:, 1, 0] + v_1[:, 2] * normal_after_first_deviation[:, 2, 0],
        )
        v_2 = second_rotation_axis_deviations @ transforms.rotate_u(second_joint_angle_2[:, 0], device=device) @ self.kinematics_standard_orientation
        first_joint_angle_2 = torch.atan2(
            v_2[:, 1] * normal_after_first_deviation[:, 2, 0] - v_2[:, 2] * normal_after_first_deviation[:, 1, 0],
            v_2[:, 1] * normal_after_first_deviation[:, 1, 0] + v_2[:, 2] * normal_after_first_deviation[:, 2, 0],
        )
        
        first_joint_angle_1 = torch.atan2(torch.sin(first_joint_angle_1), torch.cos(first_joint_angle_1))
        first_joint_angle_2 = torch.atan2(torch.sin(first_joint_angle_2), torch.cos(first_joint_angle_2))

        # Determine possible motor positions.
        motor_positions_1 = self.actuators.angles_to_motor_positions(
            angles=torch.stack([first_joint_angle_1, second_joint_angle_1[:, 0]], dim=-1),
            device=device
        )
        motor_positions_2 = self.actuators.angles_to_motor_positions(
            angles=torch.stack([first_joint_angle_2, second_joint_angle_2[:, 0]], dim=-1),
            device=device
        )

        # Determine valid solution. Prefer motor_positions_1 when valid, otherwise use motor_positions_2.
        min_pos = self.actuators.active_non_optimizable_parameters[
            :, indices.actuator_min_motor_position
        ]
        max_pos = self.actuators.active_non_optimizable_parameters[
            :, indices.actuator_max_motor_position
        ]
        valid_1 = (motor_positions_1 >= min_pos) & (motor_positions_1 <= max_pos)
        valid_2 = (motor_positions_2 >= min_pos) & (motor_positions_2 <= max_pos)

        solution_1_valid = valid_1.all(dim=1)
        solution_2_valid = valid_2.all(dim=1)

        # Detect active heliostats where neither solution works.
        invalid_rows = torch.nonzero(~(solution_1_valid | solution_2_valid))[:, 0]

        if invalid_rows.numel() > 0:
            log.warning(
                f"No valid motor position combination for active heliostat number(s): {invalid_rows.tolist()}."
            )
        
        motor_positions = torch.where(
            solution_1_valid[:, None], motor_positions_1, motor_positions_2,
        )

        return motor_positions


    def motor_positions_to_orientations(
        self, motor_positions: torch.Tensor, device: torch.device | None = None
    ) -> torch.Tensor:
        """
        Compute orientation matrices given the motor positions.

        Parameters
        ----------
        motor_positions : torch.Tensor
            The motor positions.
            Shape is ``[number_of_active_heliostats, 2]``.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The orientation matrices.
            Shape is ``[number_of_active_heliostats, 4, 4]``.
        """
        device = get_device(device=device)

        orientations = self._compute_orientations_from_motor_positions(
            motor_positions=motor_positions, device=device
        )

        return orientations @ self.initial_orientation_offsets


    def incident_ray_directions_to_orientations(
        self,
        incident_ray_directions: torch.Tensor,
        aim_points: torch.Tensor,
        max_num_iterations: int = 4,
        min_eps: float = 0.0001,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Compute orientation matrices given incident ray directions.

        Parameters
        ----------
        incident_ray_directions : torch.Tensor
            The directions of the incident rays as seen from the heliostats.
            Shape is ``[number_of_active_heliostats, 4]``.
        aim_points : torch.Tensor
            The aim points for the active heliostats.
            Shape is ``[number_of_active_heliostats, 4]``.
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
            Shape is ``[number_of_active_heliostats, 4, 4]``.
        """
        device = get_device(device=device)

        motor_positions = torch.zeros(
            (
                self.number_of_active_heliostats,
                constants.rigid_body_number_of_actuators,
            ),
            device=device,
        )
        last_iteration_loss = None

        for _ in range(max_num_iterations):
            orientations = self._compute_orientations_from_motor_positions(
                motor_positions=motor_positions, device=device,
            )

            concentrator_normals = orientations @ torch.tensor(
                [0.0, -1.0, 0.0, 0.0], device=device,
            )

            concentrator_origins = orientations @ torch.tensor(
                [0.0, 0.0, 0.0, 1.0], device=device,
            )

            desired_reflection_directions = torch.nn.functional.normalize(
                aim_points[:, :3] - concentrator_origins[:, :3],
                p=2,
                dim=1,
                eps=1e-8,
            )

            desired_concentrator_normals = torch.nn.functional.normalize(
                -incident_ray_directions[:, :3] + desired_reflection_directions,
                p=2,
                dim=1,
                eps=1e-8,
            )
            
            desired_concentrator_normals = coordinates.convert_3d_directions_to_4d_format(desired_concentrator_normals, device=device)

            loss = torch.abs(desired_concentrator_normals - concentrator_normals).mean(
                dim=-1
            )

            # Stop if converged.
            print(loss, loss.mean())
            if isinstance(last_iteration_loss, torch.Tensor):
                eps = torch.abs(last_iteration_loss - loss)
                if torch.any(eps <= min_eps):
                    break
            last_iteration_loss = loss
            
            motor_positions = self._compute_motor_positions_from_normal(
                normals=desired_concentrator_normals,
                device=device
            )

        self.active_motor_positions = motor_positions
        
        return orientations @ self.initial_orientation_offsets
