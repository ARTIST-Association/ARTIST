import math
from typing import List, Dict, Any

import h5py
import torch
import numpy as np
from yacs.config import CfgNode
from scipy.spatial.transform import Rotation

from artist.io.datapoint import HeliostatDataPoint
from artist.physics_objects.heliostats.alignment.kinematic.actuators.actuator import (
    AActuatorModule,
)
from artist.physics_objects.heliostats.alignment.kinematic.actuators.ideal_actuator import (
    IdealActuator,
)

from artist.physics_objects.heliostats.alignment.kinematic.kinematic import (
    AKinematicModule,
)
from artist.physics_objects.heliostats.alignment.kinematic.parameter import AParameter
from artist.util import utils, artist_type_mapping_dict
from artist.util import config_dictionary


class RigidBodyModule(AKinematicModule):
    """
    This class implements a rigid body kinematic model.

    Attributes
    ----------
    position : torch.Tensor
        TODO

    Methods
    -------
    compute_orientation_from_aimpoint()
        Computes the orientation matrix given the desired aim point.

    See Also
    --------
    :class: AKinematicModule : Reference to the parent class
    """

    def __init__(
        self,
        position: torch.Tensor,
        aim_point: torch.Tensor,
        actuator_type: str,
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
        Initialize the neural network rigid body fusion as a kinematic module.

        Parameters
        ----------
        position : torch.Tensor
            Position of the heliostat for which the kinematic model is valid.
        **deviations
            Additional keyword arguments.
        """
        super().__init__(position=position)
        self.position = position
        self.aim_point = aim_point

        # TODO: Figure out how to handle true and false
        self.actuator_1 = artist_type_mapping_dict.actuator_type_mapping.get(
            actuator_type
        )(joint_number=1, clockwise=False)
        self.actuator_2 = artist_type_mapping_dict.actuator_type_mapping.get(
            actuator_type
        )(joint_number=2, clockwise=True)

        self.deviation_parameters = deviation_parameters
        self.initial_orientation_offset = initial_orientation_offset

    def align(
        self,
        incident_ray_direction: torch.tensor,
        max_num_epochs: int = 2,
        min_eps: float = 0.0001,
    ) -> torch.Tensor:
        """
        Compute the orientation-matrix from an aimpoint defined in a datapoint.

        Parameters
        ----------
        data_point : HeliostatDataPoint
            Datapoint containing the desired aimpoint.
        max_num_epochs : int
            Maximum number of iterations (default 20)
        min_eps : float
            Convergence criterion (default 0.0001)

        Returns
        -------
        torch.Tensor
            The orientation matrix.
        """
        actuator_steps = torch.zeros((1, 2), requires_grad=True)
        orientation = None
        last_epoch_loss = None
        for epoch in range(max_num_epochs):
            # orientation = self.compute_orientation_from_steps(
            #     actuator_1_steps=actuator_steps[:, 0],
            #     actuator_2_steps=actuator_steps[:, 1],
            # )
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
            if isinstance(last_epoch_loss, torch.Tensor):
                eps = torch.abs(last_epoch_loss - loss)
                if torch.any(eps <= min_eps):
                    break
            last_epoch_loss = loss.detach()

            # Analytical Solution

            # Calculate joint 2 angle
            joint_2_angles = -torch.arcsin(
                -desired_concentrator_normal[:, 0]
                / torch.cos(
                    self.deviation_parameters[
                        config_dictionary.second_joint_translation_n
                    ]
                )
            )

            # Calculate joint 1 angle
            a = -torch.cos(
                self.deviation_parameters[config_dictionary.second_joint_translation_e]
            ) * torch.cos(joint_2_angles) + torch.sin(
                self.deviation_parameters[config_dictionary.second_joint_translation_e]
            ) * torch.sin(
                self.deviation_parameters[config_dictionary.second_joint_translation_n]
            ) * torch.sin(
                joint_2_angles
            )
            b = -torch.sin(
                self.deviation_parameters[config_dictionary.second_joint_translation_e]
            ) * torch.cos(joint_2_angles) - torch.cos(
                self.deviation_parameters[config_dictionary.second_joint_translation_e]
            ) * torch.sin(
                self.deviation_parameters[config_dictionary.second_joint_translation_n]
            ) * torch.sin(
                joint_2_angles
            )

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

        return orientation @ utils.rotate_e(
            e=torch.tensor([self.initial_orientation_offset])
        )

    # def build_first_rotation_matrix(self, angles: torch.Tensor) -> torch.Tensor:
    #     """
    #     Build the first rotation matrices.
    #
    #     The first joint rotation is around the x-axis (east-axis).
    #
    #     Parameters
    #     ----------
    #     angles : torch.Tensor
    #         Angles specifying the rotation.
    #
    #     Returns
    #     -------
    #     torch.Tensor
    #         The rotation-matrices for the specified angles.
    #     """
    #     # Compute rotation matrix elements
    #     zeros = torch.zeros_like(angles)
    #     ones = torch.ones_like(angles)
    #     cos_theta = torch.cos(angles)
    #     sin_theta = torch.sin(angles)
    #
    #     # Initialize rotation_matrices with tilt deviations.
    #     rot_matrix = torch.stack(
    #         [
    #             torch.stack(
    #                 [
    #                     ones,
    #                     zeros,
    #                     zeros,
    #                     ones
    #                     * self.deviation_parameters[
    #                         config_dictionary.first_joint_translation_e
    #                     ],
    #                 ],
    #                 dim=1,
    #             ),
    #             torch.stack(
    #                 [
    #                     zeros,
    #                     cos_theta,
    #                     -sin_theta,
    #                     ones
    #                     * self.deviation_parameters[
    #                         config_dictionary.first_joint_translation_n
    #                     ],
    #                 ],
    #                 dim=1,
    #             ),
    #             torch.stack(
    #                 [
    #                     zeros,
    #                     sin_theta,
    #                     cos_theta,
    #                     ones
    #                     * self.deviation_parameters[
    #                         config_dictionary.first_joint_translation_u
    #                     ],
    #                 ],
    #                 dim=1,
    #             ),
    #             torch.stack([zeros, zeros, zeros, ones], dim=1),
    #         ],
    #         dim=1,
    #     )
    #     # TODO: include east tilt matrix
    #     north_tilt_matrix = self.build_north_rotation_4x4(
    #         angle=self.deviation_parameters[config_dictionary.first_joint_tilt_n]
    #     )
    #     up_tilt_matrix = self.build_up_rotation_4x4(
    #         angle=self.deviation_parameters[config_dictionary.first_joint_tilt_u]
    #     )
    #     rotation_matrices = north_tilt_matrix @ up_tilt_matrix @ rot_matrix
    #     return rotation_matrices

    # def build_second_rotation_matrix(self, angles: torch.Tensor) -> torch.Tensor:
    #     """
    #     Build the second rotation matrices.
    #
    #     The second joint rotation is around the z-axis (up-axis).
    #
    #     Parameters
    #     ----------
    #     angles : torch.Tensor
    #         Angle specifying the rotation.
    #
    #     Returns
    #     -------
    #     torch.Tensor
    #         The rotation matrices for the specified angles.
    #     """
    #     # Compute rotation matrix elements.
    #     zeros = torch.zeros_like(angles)
    #     ones = torch.ones_like(angles)
    #     cos_theta = torch.cos(angles)
    #     sin_theta = torch.sin(angles)
    #
    #     # Initialize rotation matrices with tilt deviations.
    #     rot_matrix = torch.stack(
    #         [
    #             torch.stack(
    #                 [
    #                     cos_theta,
    #                     -sin_theta,
    #                     zeros,
    #                     ones
    #                     * self.deviation_parameters[
    #                         config_dictionary.second_joint_translation_e
    #                     ],
    #                 ],
    #                 dim=1,
    #             ),
    #             torch.stack(
    #                 [
    #                     sin_theta,
    #                     cos_theta,
    #                     zeros,
    #                     ones
    #                     * self.deviation_parameters[
    #                         config_dictionary.second_joint_translation_n
    #                     ],
    #                 ],
    #                 dim=1,
    #             ),
    #             torch.stack(
    #                 [
    #                     zeros,
    #                     zeros,
    #                     ones,
    #                     ones
    #                     * self.deviation_parameters[
    #                         config_dictionary.second_joint_translation_u
    #                     ],
    #                 ],
    #                 dim=1,
    #             ),
    #             torch.stack([zeros, zeros, zeros, ones], dim=1),
    #         ],
    #         dim=1,
    #     )
    #     # TODO: include up rotation matrix
    #     east_tilt_matrix = self.build_east_rotation_4x4(
    #         angle=self.deviation_parameters[config_dictionary.second_joint_tilt_e]
    #     )
    #     north_tilt_matrix = self.build_north_rotation_4x4(
    #         angle=self.deviation_parameters[config_dictionary.second_joint_tilt_n]
    #     )
    #     rotation_matrices = east_tilt_matrix @ north_tilt_matrix @ rot_matrix
    #     return rotation_matrices

    # def build_concentrator_matrix(self) -> torch.Tensor:
    #     """
    #     Build the concentrator rotation matrix.
    #
    #     Returns
    #     -------
    #     torch.Tensor
    #         The rotation matrix.
    #     """
    #     # TODO: include all rotation matrices
    #     rotation_matrix = torch.eye(4)
    #     rotation_matrix[0, -1] += self.deviation_parameters[
    #         config_dictionary.concentrator_translation_e
    #     ]
    #     rotation_matrix[1, -1] += self.deviation_parameters[
    #         config_dictionary.concentrator_translation_n
    #     ]
    #     rotation_matrix[2, -1] += self.deviation_parameters[
    #         config_dictionary.concentrator_translation_u
    #     ]
    #     return rotation_matrix

    # def compute_orientation_from_steps(
    #     self, actuator_1_steps: torch.Tensor, actuator_2_steps: torch.Tensor
    # ) -> torch.Tensor:
    #     """
    #     Compute the orientation matrix from given actuator steps.
    #
    #     Parameters
    #     ----------
    #     actuator_1_steps : torch.Tensor
    #         Steps of actuator 1.
    #     actuator_2_steps : torch.Tensor
    #         Steps of actuator 2.
    #
    #     Returns
    #     -------
    #     torch.Tensor
    #         The orientation matrix.
    #     """
    #     first_joint_rot_angles = self.actuator_1(actuator_pos=actuator_1_steps)
    #     second_joint_rot_angles = self.actuator_2(actuator_pos=actuator_2_steps)
    #     return self.compute_orientation_from_angles(
    #         first_joint_rot_angles, second_joint_rot_angles
    #     )

    # def compute_orientation_from_angles(
    #     self,
    #     joint_1_angles: torch.Tensor,
    #     joint_2_angles: torch.Tensor,
    # ) -> torch.Tensor:
    #     """
    #     Compute the orientation matrix from given joint angles.
    #
    #     Parameters
    #     ----------
    #     joint_1_angles : torch.Tensor
    #         Angles of the first joint.
    #     joint_2_angles : torch.Tensor
    #         Angles of the second joint.
    #
    #     Returns
    #     -------
    #     torch.Tensor
    #         The orientation matrix.
    #     """
    #     # Heliostats are initially orientated at (0,0,0) pointing to the south
    #     initial_orientations = (
    #         torch.eye(4).unsqueeze(0).repeat(len(joint_1_angles), 1, 1)
    #     )
    #
    #     # Account for position
    #     initial_orientations = initial_orientations @ utils.translate_enu(
    #         e=self.position[0], n=self.position[1], u=self.position[2]
    #     )
    #
    #     joint_1_rotations = (
    #         utils.rotate_n(
    #             n=self.deviation_parameters[config_dictionary.first_joint_tilt_n]
    #         )
    #         @ utils.rotate_u(
    #             u=self.deviation_parameters[config_dictionary.first_joint_tilt_u]
    #         )
    #         @ utils.translate_enu(
    #             e=self.deviation_parameters[
    #                 config_dictionary.first_joint_translation_e
    #             ],
    #             n=self.deviation_parameters[
    #                 config_dictionary.first_joint_translation_n
    #             ],
    #             u=self.deviation_parameters[
    #                 config_dictionary.first_joint_translation_u
    #             ],
    #         )
    #         @ utils.rotate_e(joint_1_angles)
    #     )
    #     joint_2_rotations = (
    #         utils.rotate_e(
    #             e=self.deviation_parameters[config_dictionary.second_joint_tilt_e]
    #         )
    #         @ utils.rotate_n(
    #             n=self.deviation_parameters[config_dictionary.second_joint_tilt_n]
    #         )
    #         @ utils.translate_enu(
    #             e=self.deviation_parameters[
    #                 config_dictionary.second_joint_translation_e
    #             ],
    #             n=self.deviation_parameters[
    #                 config_dictionary.second_joint_translation_n
    #             ],
    #             u=self.deviation_parameters[
    #                 config_dictionary.second_joint_translation_u
    #             ],
    #         )
    #         @ utils.rotate_u(joint_2_angles)
    #     )
    #
    #     return (
    #         initial_orientations
    #         @ joint_1_rotations
    #         @ joint_2_rotations
    #         @ utils.translate_enu(
    #             e=self.deviation_parameters[
    #                 config_dictionary.concentrator_translation_e
    #             ],
    #             n=self.deviation_parameters[
    #                 config_dictionary.concentrator_translation_n
    #             ],
    #             u=self.deviation_parameters[
    #                 config_dictionary.concentrator_translation_u
    #             ],
    #         )
    #     )

    # def transform_normal_to_first_coord_sys(
    #     self, concentrator_normal: torch.Tensor
    # ) -> torch.Tensor:
    #     """
    #     Transform the concentrator normal from the global coordinate system to the CS of the first joint.
    #
    #     Parameters
    #     ----------
    #     concentrator_normal : torch.Tensor
    #         Normal vector of the heliostat.
    #
    #     Returns
    #     -------
    #     torch.Tensor
    #         The transformed concentrator normal.
    #     """
    #     # normal4x1 = -torch.cat(
    #     #     (concentrator_normal, torch.zeros(concentrator_normal.shape[0], 1)), dim=1
    #     # )
    #     normal4x1 = -concentrator_normal
    #     first_rot_matrices = self.build_first_rotation_matrix(
    #         torch.zeros(len(concentrator_normal))
    #     )
    #
    #     initial_orientations = (
    #         torch.eye(4).unsqueeze(0).repeat(len(concentrator_normal), 1, 1)
    #     )
    #
    #     # Account for position
    #     initial_orientations = initial_orientations @ utils.translate_enu(
    #         e=self.position[0], n=self.position[1], u=self.position[2]
    #     )
    #
    #     first_orientations = torch.matmul(initial_orientations, first_rot_matrices)
    #     transposed_first_orientations = torch.transpose(first_orientations, 1, 2)
    #
    #     normal_first_orientation = torch.matmul(
    #         transposed_first_orientations, normal4x1.unsqueeze(-1)
    #     ).squeeze(-1)
    #     return normal_first_orientation[:, :3]

    # def compute_steps_from_normal(
    #     self, concentrator_normal: torch.Tensor
    # ) -> torch.Tensor:
    #     """
    #     Compute the steps for actuator 1 and 2 from the concentrator normal.
    #
    #     Parameters
    #     ----------
    #     concentrator_normal : torch.Tensor
    #         Normal vector of the heliostat.
    #
    #     Returns
    #     -------
    #     torch.Tensor
    #         The calculated necessary actuator steps to reach a given normal vector.
    #     """
    #     joint_angles = self.compute_angles_from_normal(
    #         normal_first_orientation=-concentrator_normal
    #     )
    #
    #     actuator_steps_1 = self.actuator_1.angles_to_motor_steps(joint_angles[:, 0])
    #     actuator_steps_2 = self.actuator_2.angles_to_motor_steps(joint_angles[:, 1])
    #
    #     return torch.stack((actuator_steps_1, actuator_steps_2), dim=-1)

    # def compute_angles_from_normal(
    #     self, normal_first_orientation: torch.Tensor
    # ) -> torch.Tensor:
    #     """
    #     Compute the two joint angles from a normal vector.
    #
    #     Parameters
    #     ----------
    #     normal_first_orientation : torch.Tensor
    #         Normal transformed into the coordinate system of the first joint.
    #
    #     Returns
    #     -------
    #     torch.Tensor
    #         The calculated necessary actuator angles to reach a given normal vector.
    #     """
    #     e = 0
    #     n = 1
    #     u = 2
    #
    #     sin_2e = torch.sin(
    #         self.deviation_parameters[config_dictionary.second_joint_translation_e]
    #     )
    #     cos_2e = torch.cos(
    #         self.deviation_parameters[config_dictionary.second_joint_translation_e]
    #     )
    #
    #     sin_2n = torch.sin(
    #         self.deviation_parameters[config_dictionary.second_joint_translation_n]
    #     )
    #     cos_2n = torch.cos(
    #         self.deviation_parameters[config_dictionary.second_joint_translation_n]
    #     )
    #
    #     calc_step_1 = normal_first_orientation[:, e] / cos_2n
    #     joint_2_angles = -torch.arcsin(calc_step_1)
    #
    #     sin_2u = torch.sin(joint_2_angles)
    #     cos_2u = torch.cos(joint_2_angles)
    #
    #     # Joint angle 1
    #     a = -cos_2e * cos_2u + sin_2e * sin_2n * sin_2u
    #     b = -sin_2e * cos_2u - cos_2e * sin_2n * sin_2u
    #
    #     numerator = (
    #         a * normal_first_orientation[:, u] - b * normal_first_orientation[:, n]
    #     )
    #     denominator = (
    #         a * normal_first_orientation[:, n] + b * normal_first_orientation[:, u]
    #     )
    #
    #     joint_1_angles = torch.arctan2(numerator, denominator) - torch.pi
    #
    #     return torch.stack((joint_1_angles, joint_2_angles), dim=-1)

    # @staticmethod
    # def build_east_rotation_4x4(
    #     angle: torch.Tensor,
    #     dtype: torch.dtype = torch.get_default_dtype(),
    #     device: torch.device = torch.device("cpu"),
    # ) -> torch.Tensor:
    #     """
    #     Build 4x4 rotation matrix for east direction.
    #
    #     Parameters
    #     ----------
    #     angle : torch.Tensor
    #         Angle specifying the rotation.
    #     dtype : torch.dtype
    #         Type and size of the data.
    #     device : torch.device
    #         The device type responsible to load tensors into memory.
    #
    #     Returns
    #     -------
    #     torch.Tensor
    #         The rotation matrix
    #     """
    #     s = torch.sin(angle)
    #     c = torch.cos(angle)
    #     o = torch.tensor(1, dtype=dtype, device=device, requires_grad=False)
    #     z = torch.tensor(0, dtype=dtype, device=device, requires_grad=False)
    #
    #     r_e = torch.stack([o, z, z, z])
    #     r_n = torch.stack([z, c, -s, z])
    #     r_u = torch.stack([z, s, c, z])
    #     r_pos = torch.stack([z, z, z, o])
    #
    #     return torch.vstack((r_e, r_n, r_u, r_pos))
    #
    # @staticmethod
    # def build_north_rotation_4x4(
    #     angle: torch.Tensor,
    #     dtype: torch.dtype = torch.get_default_dtype(),
    #     device: torch.device = torch.device("cpu"),
    # ) -> torch.Tensor:
    #     """
    #     Build 4x4 rotation matrix for north direction.
    #
    #     Parameters
    #     ----------
    #     angle : torch.Tensor
    #         Angle specifying the rotation.
    #     dtype : torch.dtype
    #         Type and size of the data.
    #     device : torch.device
    #         The device type responsible to load tensors into memory.
    #
    #     Returns
    #     -------
    #     torch.Tensor
    #         The rotation matrix.
    #     """
    #     s = torch.sin(angle)
    #     c = torch.cos(angle)
    #     o = torch.tensor(1, dtype=dtype, device=device, requires_grad=False)
    #     z = torch.tensor(0, dtype=dtype, device=device, requires_grad=False)
    #
    #     r_e = torch.stack([c, z, s, z])
    #     r_n = torch.stack([z, o, z, z])
    #     r_u = torch.stack([-s, z, c, z])
    #     r_pos = torch.stack([z, z, z, o])
    #
    #     return torch.stack((r_e, r_n, r_u, r_pos)).squeeze()

    # @staticmethod
    # def build_up_rotation_4x4(
    #     angle: torch.Tensor,
    #     dtype: torch.dtype = torch.get_default_dtype(),
    #     device: torch.device = torch.device("cpu"),
    # ) -> torch.Tensor:
    #     """
    #     Build 4x4 rotation matrix for up direction.
    #
    #     Parameters
    #     ----------
    #     angle : torch.Tensor
    #         Angle specifying the rotation.
    #     dtype : torch.dtype
    #         Type and size of the data.
    #     device : torch.device
    #         The device type responsible to load tensors into memory.
    #
    #     Returns
    #     -------
    #     torch.Tensor
    #         The rotation matrix.
    #     """
    #     s = torch.sin(angle)
    #     c = torch.cos(angle)
    #     o = torch.tensor(1, dtype=dtype, device=device, requires_grad=False)
    #     z = torch.tensor(0, dtype=dtype, device=device, requires_grad=False)
    #
    #     r_e = torch.stack([c, -s, z, z])
    #     r_n = torch.stack([s, c, z, z])
    #     r_u = torch.stack([z, z, o, z])
    #     r_pos = torch.stack([z, z, z, o])
    #
    #     return torch.stack((r_e, r_n, r_u, r_pos)).squeeze()
