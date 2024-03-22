from typing import Any, Dict

import torch

from artist.physics_objects.kinematic import (
    AKinematicModule,
)
from artist.util import config_dictionary, utils


class RigidBodyModule(AKinematicModule):
    """
    This class implements a rigid body kinematic model.

    Attributes
    ----------
    position : torch.Tensor
        The position of the heliostat in the field.
    aim_point : torch.Tensor
        The aim point.
    actuator_1 : Union[LinearActuator, Ideal_actuator, ...]
        Actuator number one of the heliostat.
    actuator_2 : Union[LinearActuator, Ideal_actuator, ...]
        Actuator number two of the heliostat.
    deviation_parameters : Dict[str, torch.Tensor]
        18 deviation parameters describing imperfections in the heliostat.
    initial_orientation_offset : float
        The initial orientation-rotation angle of the heliostat.

    Methods
    -------
    align()
        Compute the rotation matrix to align the concentrator along a desired orientation.

    See Also
    --------
    :class:`AKinematicModule` : Reference to the parent class.
    """

    def __init__(
        self,
        position: torch.Tensor,
        aim_point: torch.Tensor,
        actuator_type: Any,
        deviation_parameters: Dict[str, torch.Tensor] = {
            config_dictionary.first_joint_translation_e: torch.tensor(0.0),
            config_dictionary.first_joint_translation_n: torch.tensor(0.0),
            config_dictionary.first_joint_translation_u: torch.tensor(0.0),
            config_dictionary.first_joint_tilt_e: torch.tensor(0.0),
            config_dictionary.first_joint_tilt_n: torch.tensor(0.0),
            config_dictionary.first_joint_tilt_u: torch.tensor(0.0),
            config_dictionary.second_joint_translation_e: torch.tensor(0.0),
            config_dictionary.second_joint_translation_n: torch.tensor(0.0),
            config_dictionary.second_joint_translation_u: torch.tensor(0.0),
            config_dictionary.second_joint_tilt_e: torch.tensor(0.0),
            config_dictionary.second_joint_tilt_n: torch.tensor(0.0),
            config_dictionary.second_joint_tilt_u: torch.tensor(0.0),
            config_dictionary.concentrator_translation_e: torch.tensor(0.0),
            config_dictionary.concentrator_translation_n: torch.tensor(0.0),
            config_dictionary.concentrator_translation_u: torch.tensor(0.0),
            config_dictionary.concentrator_tilt_e: torch.tensor(0.0),
            config_dictionary.concentrator_tilt_n: torch.tensor(0.0),
            config_dictionary.concentrator_tilt_u: torch.tensor(0.0),
        },
        initial_orientation_offset: float = 0.0,
    ) -> None:
        """
        Initialize the rigid body kinematic.

        Parameters
        ----------
        position : torch.Tensor
            Position of the heliostat in the field.
        aim_point : torch.Tensor
            The aim point.
        actuator_type : str
            The type of the actuators of the heliostat.
        deviation_parameters : Dict[str, torch.Tensor]
            The 18 deviation parameters of the kinematic module.
        initial_orientation_offset : float
            The initial orientation-rotation angle of the heliostat.
        """
        super().__init__(position=position)
        self.position = position
        self.aim_point = aim_point

        self.actuator_1 = actuator_type(joint_number=1, clockwise=False)
        self.actuator_2 = actuator_type(joint_number=2, clockwise=True)

        self.deviation_parameters = deviation_parameters
        self.initial_orientation_offset = initial_orientation_offset

    def align(
        self,
        incident_ray_direction: torch.Tensor,
        max_num_iterations: int = 2,
        min_eps: float = 0.0001,
    ) -> torch.Tensor:
        """
        Compute the rotation matrix to align the concentrator along a desired orientation.

        Parameters
        ----------
        incident_ray_direction : torch.Tensor
            The direction of the incident ray as seen from the heliostat.
        max_num_iterations : int
            Maximum number of iterations (default 2).
        min_eps : float
            Convergence criterion (default 0.0001).

        Returns
        -------
        torch.Tensor
            The orientation matrix.
        """
        actuator_steps = torch.zeros((1, 2), requires_grad=True)
        orientation = None
        last_iteration_loss = None
        for _ in range(max_num_iterations):
            joint_1_angles = self.actuator_1(actuator_pos=actuator_steps[:, 0])
            joint_2_angles = self.actuator_2(actuator_pos=actuator_steps[:, 1])

            initial_orientations = (
                torch.eye(4).unsqueeze(0).repeat(len(joint_1_angles), 1, 1)
            )

            # Account for position
            initial_orientations = initial_orientations @ utils.translate_enu(
                e=self.position[0], n=self.position[1], u=self.position[2]
            )

            joint_1_rotations = (
                utils.rotate_n(
                    n=self.deviation_parameters[config_dictionary.first_joint_tilt_n]
                )
                @ utils.rotate_u(
                    u=self.deviation_parameters[config_dictionary.first_joint_tilt_u]
                )
                @ utils.translate_enu(
                    e=self.deviation_parameters[
                        config_dictionary.first_joint_translation_e
                    ],
                    n=self.deviation_parameters[
                        config_dictionary.first_joint_translation_n
                    ],
                    u=self.deviation_parameters[
                        config_dictionary.first_joint_translation_u
                    ],
                )
                @ utils.rotate_e(joint_1_angles)
            )
            joint_2_rotations = (
                utils.rotate_e(
                    e=self.deviation_parameters[config_dictionary.second_joint_tilt_e]
                )
                @ utils.rotate_n(
                    n=self.deviation_parameters[config_dictionary.second_joint_tilt_n]
                )
                @ utils.translate_enu(
                    e=self.deviation_parameters[
                        config_dictionary.second_joint_translation_e
                    ],
                    n=self.deviation_parameters[
                        config_dictionary.second_joint_translation_n
                    ],
                    u=self.deviation_parameters[
                        config_dictionary.second_joint_translation_u
                    ],
                )
                @ utils.rotate_u(joint_2_angles)
            )

            orientation = (
                initial_orientations
                @ joint_1_rotations
                @ joint_2_rotations
                @ utils.translate_enu(
                    e=self.deviation_parameters[
                        config_dictionary.concentrator_translation_e
                    ],
                    n=self.deviation_parameters[
                        config_dictionary.concentrator_translation_n
                    ],
                    u=self.deviation_parameters[
                        config_dictionary.concentrator_translation_u
                    ],
                )
            )

            concentrator_normals = orientation @ torch.tensor(
                [0, -1, 0, 0], dtype=torch.float32
            )
            concentrator_origins = orientation @ torch.tensor(
                [0, 0, 0, 1], dtype=torch.float32
            )

            # Compute desired normal.
            desired_reflect_vec = self.aim_point - concentrator_origins
            desired_reflect_vec /= desired_reflect_vec.norm()
            incident_ray_direction /= incident_ray_direction.norm()
            desired_concentrator_normal = incident_ray_direction + desired_reflect_vec
            desired_concentrator_normal /= desired_concentrator_normal.norm()

            # Compute epoch loss.
            loss = torch.abs(desired_concentrator_normal - concentrator_normals)

            # Stop if converged.
            if isinstance(last_iteration_loss, torch.Tensor):
                eps = torch.abs(last_iteration_loss - loss)
                if torch.any(eps <= min_eps):
                    break
            last_iteration_loss = loss.detach()

            # Analytical Solution

            # Calculate joint 2 angle.
            joint_2_angles = -torch.arcsin(
                -desired_concentrator_normal[:, 0]
                / torch.cos(
                    self.deviation_parameters[
                        config_dictionary.second_joint_translation_n
                    ]
                )
            )

            # Calculate joint 1 angle.
            a = -torch.cos(
                self.deviation_parameters[config_dictionary.second_joint_translation_e]
            ) * torch.cos(joint_2_angles) + torch.sin(
                self.deviation_parameters[config_dictionary.second_joint_translation_e]
            ) * torch.sin(
                self.deviation_parameters[config_dictionary.second_joint_translation_n]
            ) * torch.sin(joint_2_angles)
            b = -torch.sin(
                self.deviation_parameters[config_dictionary.second_joint_translation_e]
            ) * torch.cos(joint_2_angles) - torch.cos(
                self.deviation_parameters[config_dictionary.second_joint_translation_e]
            ) * torch.sin(
                self.deviation_parameters[config_dictionary.second_joint_translation_n]
            ) * torch.sin(joint_2_angles)

            joint_1_angles = (
                torch.arctan2(
                    a * -desired_concentrator_normal[:, 2]
                    - b * -desired_concentrator_normal[:, 1],
                    a * -desired_concentrator_normal[:, 1]
                    + b * -desired_concentrator_normal[:, 2],
                )
                - torch.pi
            )

            actuator_steps = torch.stack(
                (
                    self.actuator_1.angles_to_motor_steps(joint_1_angles),
                    self.actuator_2.angles_to_motor_steps(joint_2_angles),
                ),
                dim=-1,
            )

        # Return orientation matrix multiplied by the initial orientation offset.
        return orientation @ utils.rotate_e(
            e=torch.tensor([self.initial_orientation_offset])
        )
