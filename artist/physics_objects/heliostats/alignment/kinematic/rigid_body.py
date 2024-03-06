from typing import List

import h5py
import torch
from yacs.config import CfgNode

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
from artist.util import utils


class RigidBodyModule(AKinematicModule):
    """
    This class implements a rigid body kinematic model.

    See Also
    --------
    :class: AKinematicModule : Reference to the parent class
    """

    # class DevTranslationParameter(AParameter):
    #     def __init__(
    #         self,
    #         name: str,
    #         value: float = 0.0,
    #         tolerance: float = 0.1,
    #         requires_grad: bool = True,
    #         distort: bool = False,
    #     ):  # -> +/- 0.1 => 1
    #         super().__init__(value, tolerance, distort, requires_grad)
    #         self.name = name
    #
    # class DevRotationParameter(AParameter):
    #     def __init__(
    #         self,
    #         name: str,
    #         value: float = 0.0,
    #         tolerance: float = 0.01,
    #         requires_grad: bool = True,
    #         distort: bool = False,
    #     ):
    #         super().__init__(value, tolerance, distort, requires_grad)
    #         self.name = name
    #
    # class DevPercentageParameter(AParameter):
    #     def __init__(
    #         self,
    #         name: str,
    #         value: float = 0.0,
    #         tolerance: float = 0.01,
    #         requires_grad: bool = True,
    #         distort: bool = False,
    #     ) -> None:
    #         super().__init__(value, tolerance, distort, requires_grad)
    #         self.name = name
    #
    # DEV_PARAMETERS = {
    #     "dev_first_translation_e": DevTranslationParameter(
    #         "dev_first_translation_e"
    #     ),  # -> +/- 0.1 => 1
    #     "dev_first_translation_n": DevTranslationParameter(
    #         "dev_first_translation_n"
    #     ),  # -> +/- 0.1 => 1
    #     "dev_first_translation_u": DevTranslationParameter(
    #         "dev_first_translation_u"
    #     ),  # -> +/- 0.1 => 1
    #     "dev_second_translation_e": DevTranslationParameter(
    #         "dev_second_translation_e"
    #     ),  # -> +/- 0.1 => 1
    #     "dev_second_translation_n": DevTranslationParameter(
    #         "dev_second_translation_n"
    #     ),  # -> +/- 0.1 => 1
    #     "dev_second_translation_u": DevTranslationParameter(
    #         "dev_second_translation_u"
    #     ),  # -> +/- 0.1 => 1
    #     "dev_conc_translation_e": DevTranslationParameter(
    #         "dev_conc_translation_e"
    #     ),  # -> +/- 0.1 => 1
    #     "dev_conc_translation_n": DevTranslationParameter(
    #         "dev_conc_translation_n"
    #     ),  # -> +/- 0.1 => 1
    #     "dev_conc_translation_u": DevTranslationParameter(
    #         "dev_conc_translation_u"
    #     ),  # -> +/- 0.1 => 1
    #     "dev_north_tilt_1": DevRotationParameter("dev_north_tilt_1"),
    #     "dev_up_tilt_1": DevRotationParameter("dev_up_tilt_1"),
    #     "dev_east_tilt_2": DevRotationParameter("dev_east_tilt_2"),
    #     "dev_north_tilt_2": DevRotationParameter("dev_north_tilt_2"),
    # }
    # actuator_1_params = {
    #     "increment": torch.tensor(154166.666),
    #     "initial_stroke_length": torch.tensor(0.075),
    #     "actuator_offset": torch.tensor(0.34061),
    #     "joint_radius": torch.tensor(0.3204),
    #     "phi_0": torch.tensor(-1.570796),
    # }
    # actuator_2_params = {
    #     "phi_0": torch.tensor(0.959931),
    #     "increment": torch.tensor(154166.666),
    #     "initial_stroke_length": torch.tensor(0.075),
    #     "actuator_offset": torch.tensor(0.3479),
    #     "joint_radius": torch.tensor(0.309),
    # }
    # actuator_1_params = {}
    # actuator_2_params = {}

    def __init__(
        self, heliostat_name: str, config_file: h5py.File, **deviations
    ) -> None:
        """
        Initialize the neural network rigid body fusion as a kinematic module.

        Parameters
        ----------
        position : torch.Tensor
            Position of the heliostat for which the kinematic model is valid.
        """
        super().__init__(
            position=torch.tensor(
                config_file["heliostats"]["heliostats_list"][heliostat_name][
                    "position"
                ][()]
            )
        )

        self.aim_point = torch.tensor(
            config_file["heliostats"]["heliostats_list"][heliostat_name]["aim_point"][
                ()
            ], dtype=torch.float
        )

        actuator_type = config_file["heliostats"]["heliostats_list"][heliostat_name][
            "parameters"
        ]["actuator_type"][()].decode("utf-8")

        if actuator_type == "ideal":
            # TODO: Check if the clockwise convention always applies
            self.actuator_1 = IdealActuator(joint_number=1, clockwise=False)
            self.actuator_2 = IdealActuator(joint_number=2, clockwise=True)
        else:
            raise NotImplementedError("ARTIST currently only supports ideal actuators.")

        # self.deviations = deviations
        # self.parameter_deviations = {
        #     param: deviations.get(param_name)
        #     for param_name, param in self.DEV_PARAMETERS.items()
        # }
        # for param in self.parameter_deviations:
        #     self._register_parameter(param)
        #
        # self.add_module(
        #     self.config.ACTUATOR_TYPE_1,
        #     AActuatorModule(
        #         joint_number=1, clockwise=False, params=self.actuator_1_params
        #     ),
        # )
        # self.add_module(
        #     self.config.ACTUATOR_TYPE_2,
        #     AActuatorModule(
        #         joint_number=2, clockwise=True, params=self.actuator_2_params
        #     ),
        # )

    # parameters_dict = {
    #     "first_translation_e": torch.tensor([0.0]),
    #     "first_translation_n": torch.tensor([0.0]),
    #     "first_translation_u": torch.tensor([0.0]),
    #     "second_translation_e": torch.tensor([0.0]),
    #     "second_translation_n": torch.tensor([0.0]),
    #     "second_translation_u": torch.tensor([0.0]),
    #     "conc_translation_e": torch.tensor([0.0]),
    #     "conc_translation_n": torch.tensor([0.0]),
    #     "conc_translation_u": torch.tensor([0.0]),
    #     "north_tilt_1": torch.tensor([0.0]),
    #     "up_tilt_1": torch.tensor([0.0]),
    #     "east_tilt_2": torch.tensor([0.0]),
    #     "north_tilt_2": torch.tensor([0.0]),
    # }

    def compute_orientation_from_aimpoint(
        self,
        incident_ray_direction: torch.Tensor,
        num_iterations: int = 2,
        min_eps: float = 0.0001,
    ) -> torch.Tensor:
        """
        Compute the orientation-matrix from an aimpoint defined in a datapoint.

        Parameters
        ----------
        incident_ray_direction : torch.Tensor
            The direction of rays.
        num_iterations : int
            Maximum number of iterations (default 2)
        min_eps : float
            Convergence criterion (default 0.0001)

        Returns
        -------
        torch.Tensor
            The orientation matrix.
        """
        actuator_steps = torch.zeros(2, 1, requires_grad=True)
        last_iteration_loss = torch.inf
        for _ in range(num_iterations):
            orientation = self.compute_orientation_from_steps(
                actuator_1_steps=actuator_steps, actuator_2_steps=actuator_steps
            )

            orientation[0][:, 1] = orientation[0][:, 2]
            orientation[0][:3, 2] = torch.linalg.cross(
                orientation[0][:3, 0], orientation[0][:3, 1]
            )

            concentrator_normals = (
                orientation @ torch.tensor([0, 0, 1, 0], dtype=torch.float32)
            )[:1, :3]
            concentrator_origins = (
                orientation @ torch.tensor([0, 0, 0, 1], dtype=torch.float32)
            )[:1, :3]

            # Compute desired normal.
            desired_reflect_vec = self.aim_point - concentrator_origins.T.contiguous()
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
            actuator_steps = self.compute_steps_from_normal(desired_concentrator_normal)

        return orientation

    def compute_orientation_from_steps(
        self, actuator_1_steps: torch.Tensor, actuator_2_steps: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the orientation matrix from given actuator steps.

        Parameters
        ----------
        actuator_1_steps : torch.Tensor
            Steps of actuator 1.
        actuator_2_steps : torch.Tensor
            Steps of actuator 2.

        Returns
        -------
        torch.Tensor
            The orientation matrix.
        """
        first_joint_rot_angles = self.actuator_1(actuator_pos=actuator_1_steps)
        second_joint_rot_angles = self.actuator_2(actuator_pos=actuator_2_steps)
        return self.compute_orientation_from_angles(
            first_joint_rot_angles, second_joint_rot_angles
        )

    def _translation_with_deviation(self, parameter_name: str) -> torch.Tensor:
        """
        Perform a translation with a deviation.

        Parameters
        ----------
        parameter_name : str
            Name of the parameter.

        Returns
        -------
        torch.Tensor
            The deviation of the parameter added to the value of the parameter.
        """
        return self.parameters_dict[parameter_name] + self._get_parameter(
            "dev_" + parameter_name
        )

    def _rotation_with_deviation(self, parameter_name: str) -> torch.Tensor:
        """
        Perform a rotation with a deviation.

        Parameters
        ----------
        parameter_name : str
            Name of the parameter.

        Returns
        -------
        torch.Tensor
            The deviation of the parameter added to the value of the parameter.
        """
        return self.parameters_dict[parameter_name] + self._get_parameter(
            "dev_" + parameter_name
        )

    def _first_translation_e(self) -> torch.Tensor:
        return self._translation_with_deviation("first_translation_e")

    def _first_translation_n(self) -> torch.Tensor:
        return self._translation_with_deviation("first_translation_n")

    def _first_translation_u(self) -> torch.Tensor:
        return self._translation_with_deviation("first_translation_u")

    def _second_translation_e(self) -> torch.Tensor:
        return self._translation_with_deviation("second_translation_e")

    def _second_translation_n(self) -> torch.Tensor:
        return self._translation_with_deviation("second_translation_n")

    def _second_translation_u(self) -> torch.Tensor:
        return self._translation_with_deviation("second_translation_u")

    def _conc_translation_e(self) -> torch.Tensor:
        return self._translation_with_deviation("conc_translation_e")

    def _conc_translation_n(self) -> torch.Tensor:
        return self._translation_with_deviation("conc_translation_n")

    def _conc_translation_u(self) -> torch.Tensor:
        return self._translation_with_deviation("conc_translation_u")

    def _north_tilt_1(self) -> torch.Tensor:
        return self._rotation_with_deviation("north_tilt_1")

    def _up_tilt_1(self) -> torch.Tensor:
        return self._rotation_with_deviation("up_tilt_1")

    def _east_tilt_2(self) -> torch.Tensor:
        return self._rotation_with_deviation("east_tilt_2")

    def _north_tilt_2(self) -> torch.Tensor:
        return self._rotation_with_deviation("north_tilt_2")

    def _register_parameter(self, parameter: AParameter) -> None:
        """
        Add parameter to the module, register it.

        Parameters
        ----------
        parameter : AParameter
            name of the parameter to be registered.
        """
        self.register_parameter(
            parameter.name,
            torch.nn.Parameter(
                parameter.initial_value,
                parameter.requires_grad,
            ),
        )

    def _get_parameter(self, name: str) -> AParameter:
        """
        Return the specified parameter.

        Parameters
        ----------
        name : str
            Name of the parameter to be returned

        Returns
        -------
        AParameter
            The parameter referenced by name.
        """
        return self.get_parameter(name)

    def build_rotation_matrix_first_axis_east(
        self, angles: torch.Tensor
    ) -> torch.Tensor:
        """
        Build the first rotation matrices.

        The first joint rotation is around the x-axis (east-axis).

        Parameters
        ----------
        angles : torch.Tensor
            Angles specifying the rotation.

        Returns
        -------
        torch.Tensor
            The rotation-matrices for the specified angles.
        """
        # Compute rotation matrix elements
        zeros = torch.zeros_like(angles)
        ones = torch.ones_like(angles)
        cos_theta = torch.cos(angles)
        sin_theta = torch.sin(angles)

        # Initialize rotation_matrices with tilt deviations.

        rot_matrix = torch.stack(
            [
                torch.stack(
                    [ones, zeros, zeros, ones * self._first_translation_e()[0]], dim=1
                ),
                torch.stack(
                    [
                        zeros,
                        cos_theta,
                        -sin_theta,
                        ones * self._first_translation_n()[0],
                    ],
                    dim=1,
                ),
                torch.stack(
                    [
                        zeros,
                        sin_theta,
                        cos_theta,
                        ones * self._first_translation_u()[0],
                    ],
                    dim=1,
                ),
                torch.stack([zeros, zeros, zeros, ones], dim=1),
            ],
            dim=1,
        )

        north_tilt_matrix = self.build_north_rotation_4x4(angle=self._north_tilt_1()[0])
        up_tilt_matrix = self.build_up_rotation_4x4(angle=self._up_tilt_1()[0])
        rotation_matrices = north_tilt_matrix @ up_tilt_matrix @ rot_matrix
        return rotation_matrices

    def build_second_rotation_matrix(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Build the second rotation matrices.

        The second joint rotation is around the z-axis (up-axis).

        Parameters
        ----------
        angles : torch.Tensor
            Angle specifying the rotation.

        Returns
        -------
        torch.Tensor
            The rotation matrices for the specified angles.
        """
        # Compute rotation matrix elements.
        zeros = torch.zeros_like(angles)
        ones = torch.ones_like(angles)
        cos_theta = torch.cos(angles)
        sin_theta = torch.sin(angles)

        # Initialize rotation matrices with tilt deviations.
        rot_matrix = torch.stack(
            [
                torch.stack(
                    [
                        cos_theta,
                        -sin_theta,
                        zeros,
                        ones * self._second_translation_e()[0],
                    ],
                    dim=1,
                ),
                torch.stack(
                    [
                        sin_theta,
                        cos_theta,
                        zeros,
                        ones * self._second_translation_n()[0],
                    ],
                    dim=1,
                ),
                torch.stack(
                    [zeros, zeros, ones, ones * self._second_translation_u()[0]], dim=1
                ),
                torch.stack([zeros, zeros, zeros, ones], dim=1),
            ],
            dim=1,
        )

        east_tilt_matrix = self.build_east_rotation_4x4(angle=self._east_tilt_2()[0])
        north_tilt_matrix = self.build_north_rotation_4x4(angle=self._north_tilt_2()[0])
        rotation_matrices = east_tilt_matrix @ north_tilt_matrix @ rot_matrix
        return rotation_matrices

    def build_concentrator_matrix(self) -> torch.Tensor:
        """
        Build the concentrator rotation matrix.

        Returns
        -------
        rotation_matrix
            The rotation matrix.
        """
        rotation_matrix = torch.eye(4)
        rotation_matrix[0, -1] += self._conc_translation_e()[0]
        rotation_matrix[1, -1] += self._conc_translation_n()[0]
        rotation_matrix[2, -1] += self._conc_translation_u()[0]
        return rotation_matrix

    def compute_orientation_from_angles(
        self, joint_1_angles: torch.Tensor, joint_2_angles: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the orientation matrix from given joint angles.

        Parameters
        ----------
        joint_1_angles : torch.Tensor
            Angles of the first joint.
        joint_2_angles : torch.Tensor
            Angles of the second joint.

        Returns
        -------
        torch.Tensor
            The orientation matrix.
        """
        general_rot_matrices = utils.general_affine_matrix(
            rx=joint_1_angles,
            rz=joint_2_angles,
            ry=torch.zeros(joint_1_angles.shape),
            tx=torch.zeros(joint_1_angles.shape),
            ty=torch.zeros(joint_1_angles.shape),
            tz=torch.zeros(joint_1_angles.shape),
            sx=torch.ones(joint_1_angles.shape),
            sy=torch.ones(joint_1_angles.shape),
            sz=torch.ones(joint_1_angles.shape),
        )

        # first_rot_matrices = self.build_rotation_matrix_first_axis_east(joint_1_angles)
        # second_rot_matrices = self.build_second_rotation_matrix(joint_2_angles)
        # conc_trans_matrix = self.build_concentrator_matrix()

        initial_orientations = (
            torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(len(joint_1_angles), 1, 1)
        )
        initial_orientations[:, 0, 3] += self.position[0]
        initial_orientations[:, 1, 3] += self.position[1]
        initial_orientations[:, 2, 3] += self.position[2]

        # first_orientations = initial_orientations @ first_rot_matrices
        # second_orientations = first_orientations @ second_rot_matrices

        return initial_orientations @ general_rot_matrices

    def transform_normal_to_first_coord_sys(
        self, concentrator_normal: torch.Tensor
    ) -> torch.Tensor:
        """
        Transform the concentrator normal from the global coordinate system to the CS of the first joint.

        Parameters
        ----------
        concentrator_normal : torch.Tensor
            Normal vector of the heliostat.

        Returns
        -------
        torch.Tensor
            The transformed concentrator normal.
        """
        normal4x1 = -torch.cat(
            (concentrator_normal, torch.zeros(concentrator_normal.shape[0], 1)), dim=1
        )
        first_rot_matrices = self.build_rotation_matrix_first_axis_east(
            torch.zeros(len(concentrator_normal))
        )

        initial_orientations = (
            torch.eye(4).unsqueeze(0).repeat(len(concentrator_normal), 1, 1)
        )
        initial_orientations[:, 0, 3] += self.position[0]
        initial_orientations[:, 1, 3] += self.position[1]
        initial_orientations[:, 2, 3] += self.position[2]

        first_orientations = torch.matmul(initial_orientations, first_rot_matrices)
        transposed_first_orientations = torch.transpose(first_orientations, 1, 2)

        normal_first_orientation = torch.matmul(
            transposed_first_orientations, normal4x1.unsqueeze(-1)
        ).squeeze(-1)
        return normal_first_orientation[:, :3]

    def compute_steps_from_normal(
        self, concentrator_normal: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the steps for actuator 1 and 2 from the concentrator normal

        Parameters
        ----------
        concentrator_normal : torch.Tensor
            Normal vector of the heliostat.

        Returns
        -------
        torch.Tensor
            The calculated necessary actuator steps to reach a given normal vector.
        """
        # normal_first_orientation = self.transform_normal_to_first_coord_sys(
        #     concentrator_normal=concentrator_normal
        # )

        joint_angles = self.compute_angles_from_normal(
            normal_first_orientation=concentrator_normal
        )

        # linear_actuator_1 = getattr(self, "LinearActuator1")
        # linear_actuator_2 = getattr(self, "LinearActuator2")

        actuator_steps_1 = self.actuator_1.angles_to_motor_steps(joint_angles[0, :])
        actuator_steps_2 = self.actuator_2.angles_to_motor_steps(joint_angles[1, :])
        return torch.vstack((actuator_steps_1, actuator_steps_2))

    def compute_angles_from_normal(
        self, normal_first_orientation: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the two joint angles from a normal vector.

        Parameters
        ----------
        normal_first_orientation : torch.Tensor
            Normal transformed into the coordinate system of the first joint.

        Returns
        -------
        torch.Tensor
            The calculated necessary actuator angles to reach a given normal vector.
        """
        e = 0
        n = 1
        u = 2

        sin_2e = torch.sin(torch.tensor([0.0]))
        cos_2e = torch.cos(torch.tensor([0.0]))

        sin_2n = torch.sin(torch.tensor([0.0]))
        cos_2n = torch.cos(torch.tensor([0.0]))

        calc_step_1 = normal_first_orientation[e, :] / cos_2n
        joint_2_angles = -torch.arcsin(calc_step_1)

        sin_2u = torch.sin(joint_2_angles)
        cos_2u = torch.cos(joint_2_angles)

        # Joint angle 1
        a = -cos_2e * cos_2u + sin_2e * sin_2n * sin_2u
        b = -sin_2e * cos_2u - cos_2e * sin_2n * sin_2u

        numerator = (
            a * normal_first_orientation[u, :] - b * normal_first_orientation[n, :]
        )
        denominator = (
            a * normal_first_orientation[n, :] + b * normal_first_orientation[u, :]
        )

        joint_1_angles = torch.arctan2(numerator, denominator) - torch.pi

        return torch.vstack((joint_1_angles, joint_2_angles))

    @staticmethod
    def build_east_rotation_4x4(
        angle: torch.Tensor,
        dtype: torch.dtype = torch.get_default_dtype(),
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Build 4x4 rotation matrix for east direction.

        Parameters
        ----------
        angle : torch.Tensor
            Angle specifying the rotation.
        dtype : torch.dtype
            Type and size of the data.
        device : torch.device
            The device type responsible to load tensors into memory.

        Returns
        -------
        torch.Tensor
            The rotation matrix
        """
        s = torch.sin(angle)
        c = torch.cos(angle)
        o = torch.tensor(1, dtype=dtype, device=device, requires_grad=False)
        z = torch.tensor(0, dtype=dtype, device=device, requires_grad=False)

        r_e = torch.stack([o, z, z, z])
        r_n = torch.stack([z, c, -s, z])
        r_u = torch.stack([z, s, c, z])
        r_pos = torch.stack([z, z, z, o])

        return torch.vstack((r_e, r_n, r_u, r_pos))

    @staticmethod
    def build_north_rotation_4x4(
        angle: torch.Tensor,
        dtype: torch.dtype = torch.get_default_dtype(),
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Build 4x4 rotation matrix for north direction.

        Parameters
        ----------
        angle : torch.Tensor
            Angle specifying the rotation.
        dtype : torch.dtype
            Type and size of the data.
        device : torch.device
            The device type responsible to load tensors into memory.

        Returns
        -------
        torch.Tensor
            The rotation matrix
        """
        s = torch.sin(angle)
        c = torch.cos(angle)
        o = torch.tensor(1, dtype=dtype, device=device, requires_grad=False)
        z = torch.tensor(0, dtype=dtype, device=device, requires_grad=False)

        r_e = torch.stack([c, z, s, z])
        r_n = torch.stack([z, o, z, z])
        r_u = torch.stack([-s, z, c, z])
        r_pos = torch.stack([z, z, z, o])

        return torch.vstack((r_e, r_n, r_u, r_pos))

    @staticmethod
    def build_up_rotation_4x4(
        angle: torch.Tensor,
        dtype: torch.dtype = torch.get_default_dtype(),
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Build 4x4 rotation matrix for up direction.

        Parameters
        ----------
        angle : torch.Tensor
            Angle specifying the rotation.
        dtype : torch.dtype
            Type and size of the data.
        device : torch.device
            The device type responsible to load tensors into memory.

        Returns
        -------
        torch.Tensor
            The rotation matrix
        """
        s = torch.sin(angle)
        c = torch.cos(angle)
        o = torch.tensor(1, dtype=dtype, device=device, requires_grad=False)
        z = torch.tensor(0, dtype=dtype, device=device, requires_grad=False)

        r_e = torch.stack([c, -s, z, z])
        r_n = torch.stack([s, c, z, z])
        r_u = torch.stack([z, z, o, z])
        r_pos = torch.stack([z, z, z, o])

        return torch.vstack((r_e, r_n, r_u, r_pos))
