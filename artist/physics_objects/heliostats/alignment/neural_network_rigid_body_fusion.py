import torch
from artist.physics_objects.heliostats.alignment.actuator import ActuatorModule
from artist.io.datapoint import HeliostatDataPoint
from artist.physics_objects.heliostats.alignment.kinematic import AKinematicModule
from artist.physics_objects.parameter import AParameter


class NeuralNetworkRigidBodyFusion(AKinematicModule):
    class DevTranslationParameter(AParameter):
        def __init__(
            self,
            name,
            value=0.0,
            tolerance=0.1,
            requires_grad=True,
            distort: bool = False,
        ):  # -> +/- 0.1 => 1
            super().__init__(value, tolerance, distort, requires_grad)
            self.NAME = name

    class DevRotationParameter(AParameter):
        def __init__(
            self,
            name,
            value=0.0,
            tolerance=0.01,
            requires_grad=True,
            distort: bool = False,
        ):
            super().__init__(value, tolerance, distort, requires_grad)
            self.NAME = name

    DEV_PARAMETERS = {
        "dev_first_translation_e": DevTranslationParameter(
            "dev_first_translation_e"
        ),  # -> +/- 0.1 => 1
        "dev_first_translation_n": DevTranslationParameter(
            "dev_first_translation_n"
        ),  # -> +/- 0.1 => 1
        "dev_first_translation_u": DevTranslationParameter(
            "dev_first_translation_u"
        ),  # -> +/- 0.1 => 1
        "dev_second_translation_e": DevTranslationParameter(
            "dev_second_translation_e"
        ),  # -> +/- 0.1 => 1
        "dev_second_translation_n": DevTranslationParameter(
            "dev_second_translation_n"
        ),  # -> +/- 0.1 => 1
        "dev_second_translation_u": DevTranslationParameter(
            "dev_second_translation_u"
        ),  # -> +/- 0.1 => 1
        "dev_conc_translation_e": DevTranslationParameter(
            "dev_conc_translation_e"
        ),  # -> +/- 0.1 => 1
        "dev_conc_translation_n": DevTranslationParameter(
            "dev_conc_translation_n"
        ),  # -> +/- 0.1 => 1
        "dev_conc_translation_u": DevTranslationParameter(
            "dev_conc_translation_u"
        ),  # -> +/- 0.1 => 1
        "dev_north_tilt_1": DevRotationParameter("dev_north_tilt_1"),
        "dev_up_tilt_1": DevRotationParameter("dev_up_tilt_1"),
        "dev_east_tilt_2": DevRotationParameter("dev_east_tilt_2"),
        "dev_north_tilt_2": DevRotationParameter("dev_north_tilt_2"),
    }

    actuator_1_params = {
        "increment": torch.tensor(154166.666),
        "initial_stroke_length": torch.tensor(0.075),
        "actuator_offset": torch.tensor(0.34061),
        "joint_radius": torch.tensor(0.3204),
        "phi_0": torch.tensor(-1.570796),
    }
    actuator_2_params = {
        "phi_0": torch.tensor(0.959931),
        "increment": torch.tensor(154166.666),
        "initial_stroke_length": torch.tensor(0.075),
        "actuator_offset": torch.tensor(0.3479),
        "joint_radius": torch.tensor(0.309),
    }

    def __init__(self, position: torch.Tensor, **deviations):
        super().__init__(position=position)
        self.position = position
        self.deviations = deviations
        self.parameter_deviations = {
            param: deviations.get(param_name)
            for param_name, param in self.DEV_PARAMETERS.items()
        }
        for param in self.parameter_deviations:
            # register and normalize deviations
            self._register_parameter(param)
        
        self.add_module(
            "LinearActuator1",
            ActuatorModule(
                joint_number=1, clockwise=False, params=self.actuator_1_params
            ),
        )
        self.add_module(
            "LinearActuator2",
            ActuatorModule(
                joint_number=2, clockwise=True, params=self.actuator_2_params
            ),
        )

    parameters_dict = {
        "first_translation_e": torch.tensor(0.0),
        "first_translation_n": torch.tensor(0.0),
        "first_translation_u": torch.tensor(0.0),
        "second_translation_e": torch.tensor(0.0),
        "second_translation_n": torch.tensor(0.0),
        "second_translation_u": torch.tensor(0.0),
        "conc_translation_e": torch.tensor(0.0),
        "conc_translation_n": torch.tensor(0.0),
        "conc_translation_u": torch.tensor(0.0),
        "north_tilt_1": torch.tensor(0.0),
        "up_tilt_1": torch.tensor(0.0),
        "east_tilt_2": torch.tensor(0.0),
        "north_tilt_2": torch.tensor(0.0),
    }

    def _translation_with_deviation(self, parameter_name):
        return self.parameters_dict[parameter_name] + self._get_parameter(
            "dev_" + parameter_name
        )
    
    def _rotation_with_deviation(self, parameter_name):
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
    
    def _register_parameter(self, parameter: AParameter):
        self.register_parameter(
            parameter.NAME,
            torch.nn.Parameter(
                parameter.initial_value,
                parameter.requires_grad,
            ),
        )

    def _get_parameter(self, name: str):
        return  self.get_parameter(name)
    
    def firstRotationMatrix(self, angles: torch.Tensor) -> torch.Tensor:
        """Build the first rotation matrix. The first joint rotation is around the x-axis (east-axis)."""
        # Compute rotation matrix elements
        zeros = torch.zeros_like(angles)
        ones = torch.ones_like(angles)
        cos_theta = torch.cos(angles)
        sin_theta = torch.sin(angles)

        # initialize rotation_matrices with tilt deviations
        rot_matrix = torch.stack(
            [
                torch.stack(
                    [ones, zeros, zeros, ones * self._first_translation_e()], dim=1
                ),
                torch.stack(
                    [zeros, cos_theta, -sin_theta, ones * self._first_translation_n()],
                    dim=1,
                ),
                torch.stack(
                    [zeros, sin_theta, cos_theta, ones * self._first_translation_u()],
                    dim=1,
                ),
                torch.stack([zeros, zeros, zeros, ones], dim=1),
            ],
            dim=1,
        )

        north_tilt_matrix = self.northRotation4x4(angle=self._north_tilt_1())
        up_tilt_matrix = self.upRotation4x4(angle=self._up_tilt_1())
        rotation_matrices = north_tilt_matrix @ up_tilt_matrix @ rot_matrix
        return rotation_matrices
    
    def secondRotationMatrix(self, angles: torch.Tensor) -> torch.Tensor:
        """Build the second rotation matrix. The second joint rotation is around the z-axis (up-axis)."""
        # Compute rotation matrix elements
        zeros = torch.zeros_like(angles)
        ones = torch.ones_like(angles)
        cos_theta = torch.cos(angles)
        sin_theta = torch.sin(angles)

        # initialize rotation_matrices with tilt deviations
        rot_matrix = torch.stack(
            [
                torch.stack(
                    [cos_theta, -sin_theta, zeros, ones * self._second_translation_e()],
                    dim=1,
                ),
                torch.stack(
                    [sin_theta, cos_theta, zeros, ones * self._second_translation_n()],
                    dim=1,
                ),
                torch.stack(
                    [zeros, zeros, ones, ones * self._second_translation_u()], dim=1
                ),
                torch.stack([zeros, zeros, zeros, ones], dim=1),
            ],
            dim=1,
        )

        east_tilt_matrix = self.eastRotation4x4(angle=self._east_tilt_2())
        north_tilt_matrix = self.northRotation4x4(angle=self._north_tilt_2())
        rotation_matrices = east_tilt_matrix @ north_tilt_matrix @ rot_matrix
        return rotation_matrices
    
    def concentratorMatrix(self) -> torch.Tensor:
        """Build the concentrator rotation matrix."""
        rotation_matrix = torch.eye(4)
        rotation_matrix[0, -1] += self._conc_translation_e()
        rotation_matrix[1, -1] += self._conc_translation_n()
        rotation_matrix[2, -1] += self._conc_translation_u()
        return rotation_matrix
    
    def computeOrientationFromSteps(self, actuator_1_steps : torch.Tensor, actuator_2_steps : torch.Tensor):
        """
        Compute the orientation matrix from given actuator steps.

        Keyword arguments:
        actuator_1_steps -- Steps of actuator 1
        actuator_2_steps -- Steps of actuator 2
        """
        linear_actuator_1 = getattr(self, "LinearActuator1")
        linear_actuator_2 = getattr(self, "LinearActuator2")
        first_joint_rot_angles = linear_actuator_1(
            actuator_pos=actuator_1_steps
        )
        second_joint_rot_angles = linear_actuator_2(
            actuator_pos=actuator_2_steps
        )
        return self.computeOrientationFromAngles(first_joint_rot_angles, second_joint_rot_angles)

    def computeOrientationFromAngles(self, joint_1_angles : torch.Tensor, joint_2_angles : torch.Tensor):
        """
        Compute the orientation matrix from given joint angles.

        Keyword arguments:
        joint_1_angles -- angles of the first joint
        joint_2_angles -- angles of the second joint
        """
        first_rot_matrices = self.firstRotationMatrix(joint_1_angles)
        second_rot_matrices = self.secondRotationMatrix(joint_2_angles)
        conc_trans_matrix = self.concentratorMatrix()

        initial_orientations = (
            torch.eye(4).unsqueeze(0).repeat(len(joint_1_angles), 1, 1)
        )
        initial_orientations[:, 0, 3] += self.position[0]
        initial_orientations[:, 1, 3] += self.position[1]
        initial_orientations[:, 2, 3] += self.position[2]

        first_orienations = initial_orientations @ first_rot_matrices
        second_orientations = first_orienations @ second_rot_matrices
        conc_orientations = second_orientations @ conc_trans_matrix

        return conc_orientations
    
    def transNormalToFirstCoSys(self, concentrator_normal : torch.Tensor):
        """
        Transform the concentrator normal from the global coordinate System to the CS of the first joint.

        Keyword arguments:
        concentrator_normal -- normal vector of the heliostat
        """
        normal4x1 = -torch.cat((concentrator_normal, torch.zeros(concentrator_normal.shape[0],1)), dim=1)
        first_rot_matrices = self.firstRotationMatrix(torch.zeros(len(concentrator_normal)))

        initial_orientations = (
            torch.eye(4).unsqueeze(0).repeat(len(concentrator_normal), 1, 1)
        )
        initial_orientations[:, 0, 3] += self.position[0]
        initial_orientations[:, 1, 3] += self.position[1]
        initial_orientations[:, 2, 3] += self.position[2]

        first_orientations = torch.matmul(initial_orientations, first_rot_matrices)
        transposed_first_orientations = torch.transpose(first_orientations, 1, 2)

        normal_first_orientation = torch.matmul(transposed_first_orientations, normal4x1.unsqueeze(-1)).squeeze(-1)
        return normal_first_orientation[:,:3]
    
    def computeStepsFromNormal(self, concentrator_normal : torch.Tensor):
        """
        Compute the steps for actuator 1 and 2 from the concentrator normal

        Keyword arguments:
        concentrator_normal -- normal vector of the heliostat
        """
        normal_first_orientation = self.transNormalToFirstCoSys(concentrator_normal=concentrator_normal)

        joint_angles = self.computeAnglesFromNormal(normal_first_orientation=normal_first_orientation)

        linear_actuator_1 = getattr(self, "LinearActuator1")
        linear_actuator_2 = getattr(self, "LinearActuator2")

        actuator_steps_1 = linear_actuator_1._anglesToSteps(joint_angles[:,0])
        actuator_steps_2 = linear_actuator_2._anglesToSteps(joint_angles[:,1])
        actuator_steps = torch.stack((actuator_steps_1, actuator_steps_2), dim=-1)

        return actuator_steps
    
    def computeAnglesFromNormal(self, normal_first_orientation : torch.Tensor):
        """
        Compute the two joint angles from a normal vector.

        Keyword arguments:
        normal_first_orientation -- normal transformed into the coordinate system of the first joint
        """
        E = 0
        N = 1
        U = 2

        sin_2e = torch.sin(self._second_translation_e())
        cos_2e = torch.cos(self._second_translation_e())

        sin_2n = torch.sin(self._second_translation_n())
        cos_2n = torch.cos(self._second_translation_n())

        calc_step_1 = normal_first_orientation[:,E] / cos_2n
        joint_2_angles = -torch.arcsin(calc_step_1)

        sin_2u = torch.sin(joint_2_angles)
        cos_2u = torch.cos(joint_2_angles)

        # joint angle 1
        a = - cos_2e * cos_2u + sin_2e * sin_2n * sin_2u
        b = - sin_2e * cos_2u - cos_2e * sin_2n * sin_2u

        numerator = a * normal_first_orientation[:,U] - b * normal_first_orientation[:,N]
        denominator = a * normal_first_orientation[:,N] + b * normal_first_orientation[:,U]

        joint_1_angles = torch.arctan2(numerator, denominator) - torch.pi

        return torch.stack((joint_1_angles, joint_2_angles), dim=-1)
    
    def computeOrientationFromAimpoint(
            self, data_point: HeliostatDataPoint,
            max_num_epochs : int = 20,
            min_eps : float = 0.0001
    ):
        """Compute the orientation-matrix from an aimpoint defined in a datapoint.

        Keyword arguments:
        data_point -- datapoint containing the desired aimpoint
        max_num_epochs -- maximum number of iterations (default 20)
        min_eps -- epsilon, convergence criteria (default 0.0001)
        """
        actuator_steps = torch.zeros((2,2), requires_grad=True)

        last_epoch_loss = None
        for epoch in range(max_num_epochs):
            orientation = self.computeOrientationFromSteps(actuator_1_steps=actuator_steps, actuator_2_steps=actuator_steps)
            concentrator_normals = (orientation @ torch.tensor([0, -1, 0, 0], dtype=torch.float32))[:1,:3]
            concentrator_origins = (orientation @ torch.tensor([0, 0, 0, 1], dtype=torch.float32))[:1,:3]
        
            orientation[0][:,1] = -orientation[0][:,1]
            orientation[0][:3,2] = torch.cross(orientation[0][:3,0], orientation[0][:3,1])

            # compute desired normal
            desired_reflect_vec = data_point.desired_aimpoint - concentrator_origins
            desired_reflect_vec /= desired_reflect_vec.norm()
            data_point.light_directions /= data_point.light_directions.norm()
            desired_concentrator_normal = data_point.light_directions + desired_reflect_vec
            desired_concentrator_normal /= desired_concentrator_normal.norm()
            desired_concentrator_normal -= concentrator_origins
            desired_concentrator_normal /= desired_concentrator_normal.norm()

            # compute epoch loss
            loss = torch.abs(desired_concentrator_normal - concentrator_normals)
            
            # stop if converged
            if isinstance(last_epoch_loss, torch.Tensor):
                eps = torch.abs(last_epoch_loss - loss)
                if torch.any(eps <= min_eps):
                    break
            last_epoch_loss = loss.detach()
            actuator_steps = self.computeStepsFromNormal(desired_concentrator_normal)

        return orientation

    def eastRotation4x4(
        self,
        angle: torch.Tensor,
        dtype: torch.dtype = torch.get_default_dtype(),
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Build 4x4 rotation matrix for east direction."""
        s = torch.sin(angle)
        c = torch.cos(angle)
        o = torch.tensor(1, dtype=dtype, device=device, requires_grad=False)
        z = torch.tensor(0, dtype=dtype, device=device, requires_grad=False)

        rE = torch.stack([o, z, z, z])
        rN = torch.stack([z, c, -s, z])
        rU = torch.stack([z, s, c, z])
        rPOS = torch.stack([z, z, z, o])

        mat = torch.vstack((rE, rN, rU, rPOS))
        return mat

    def northRotation4x4(
        self,
        angle: torch.Tensor,
        dtype: torch.dtype = torch.get_default_dtype(),
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Build 4x4 rotation matrix for north direction."""
        s = torch.sin(angle)
        c = torch.cos(angle)
        o = torch.tensor(1, dtype=dtype, device=device, requires_grad=False)
        z = torch.tensor(0, dtype=dtype, device=device, requires_grad=False)

        rE = torch.stack([c, z, s, z])
        rN = torch.stack([z, o, z, z])
        rU = torch.stack([-s, z, c, z])
        rPOS = torch.stack([z, z, z, o])

        mat = torch.vstack((rE, rN, rU, rPOS))
        return mat

    def upRotation4x4(
        self,
        angle: torch.Tensor,
        dtype: torch.dtype = torch.get_default_dtype(),
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Build 4x4 rotation matrix for up direction."""
        s = torch.sin(angle)
        c = torch.cos(angle)
        o = torch.tensor(1, dtype=dtype, device=device, requires_grad=False)
        z = torch.tensor(0, dtype=dtype, device=device, requires_grad=False)

        rE = torch.stack([c, -s, z, z])
        rN = torch.stack([s, c, z, z])
        rU = torch.stack([z, z, o, z])
        rPOS = torch.stack([z, z, z, o])

        mat = torch.vstack((rE, rN, rU, rPOS))
        return mat
