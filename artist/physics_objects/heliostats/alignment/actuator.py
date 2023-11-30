import torch

from artist.physics_objects.module import AModule
from artist.physics_objects.heliostats.normalization import ParameterNormalizer
from artist.physics_objects.parameter import AParameter


class ActuatorModule(AModule):
    """
    This class implements the behavior of a linear actuator that moves within a 2D plane as a ``PyTorch`` module.

    The actuator is parametrized by 5 parameters:
    - increment: stroke length change per motor step
    - initial_stroke_length: stroke length for a motor step of 0
    - actuator_offset: offset between the linear actuator's pivoting point and the point around which the actuator is
    allowed to pivot
    - joint_radius: the actuator's pivoting radius
    - phi_0 : the angle the actuator introduces to the manipulated coordinate system at the initial stroke length

    The actuator is modelled by the law of cosine. Two directions are implemented:
    - motor steps -> angle
    - angle -> motor steps
    """

    class DevRotationParameter(AParameter):
        """
        Attributes
        ----------

        Methods
        -------
        """

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

    class DevPercentageParameter(AParameter):
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
        "dev_increment": DevPercentageParameter("dev_increment"),  # * 99%/101% => 1
        # 'dev_initial_stroke_length': DevPercentageParameter('dev_initial_stroke_length'),# * 99%/101% => 1
        # 'dev_actuator_offset': DevPercentageParameter('dev_actuator_offset'),# * 99%/101% => 1
        # 'dev_joint_radius': DevPercentageParameter('dev_joint_radius'),# * 99%/101% => 1
        "dev_phi_0": DevRotationParameter("dev_phi_0"),  # +/- 10mRad
    }

    def _percentage_with_deviation(self, parameter_name):
        return self.PARAMS[parameter_name] * (
            1 + self._get_parameter("dev_" + parameter_name)
        )

    def _rotation_with_deviation(self, parameter_name):
        return self.PARAMS[parameter_name] + self._get_parameter(
            "dev_" + parameter_name
        )

    def _normed_percentage_with_deviation(self, parameter_name):
        return self.PARAMS[parameter_name].norm() * (
            1 + self._get_parameter("dev_" + parameter_name)
        )

    def _increment(self) -> torch.Tensor:
        return self._percentage_with_deviation("increment")

    def _initial_stroke_length(self) -> torch.Tensor:
        # return self._percentage_with_deviation('initial_stroke_length')
        return self.PARAMS["initial_stroke_length"]

    def _actuator_offset(self) -> torch.Tensor:
        #    return self._percentage_with_deviation('actuator_offset')
        return self.PARAMS["actuator_offset"]

    def _joint_radius(self) -> torch.Tensor:
        # return self._percentage_with_deviation('joint_radius')
        return self.PARAMS["joint_radius"]

    def _phi_0(self) -> torch.Tensor:
        return self._rotation_with_deviation("phi_0")

    def _register_parameter(self, parameter: AParameter):
        if not hasattr(self, "parameter_normalizer"):
            self.parameter_normalizer = ParameterNormalizer()
        self.parameter_normalizer.register_parameter(parameter)

        self.register_parameter(
            parameter.name,
            torch.nn.Parameter(
                self.parameter_normalizer.get_normalized_parameter(
                    parameter.name, parameter.initial_value
                ),
                parameter.requires_grad,
            ),
        )

    def _get_parameter(self, name: str):
        return self.parameter_normalizer.get_denormalized_parameter(
            name, self.get_parameter(name)
        )

    def __init__(self, joint_number: int, clockwise: bool, params: dict, **deviations):
        super().__init__()
        self.JOINT_NUMBER = joint_number
        self.CLOCKWISE = clockwise
        self.PARAMS = params
        self.deviations = deviations

        self.parameter_deviations = {
            param: deviations.get(param_name)
            for param_name, param in self.DEV_PARAMETERS.items()
        }
        for param in self.parameter_deviations:
            # register and normalize deviations
            self._register_parameter(param)

    # TODO remove self.JOINT_NUMBER

    def _steps_to_phi(self, actuator_pos: torch.Tensor):
        # Access actuator_pos via joint number: items in actuator_pos list have to be ordered by number
        stroke_length = (
            actuator_pos[:, self.JOINT_NUMBER - 1] / self._increment()
            + self._initial_stroke_length()
        )
        calc_step_1 = (
            self._actuator_offset() ** 2
            + self._joint_radius() ** 2
            - stroke_length**2
        )
        calc_step_2 = 2.0 * self._actuator_offset() * self._joint_radius()
        calc_step_3 = calc_step_1 / calc_step_2
        angle = torch.arccos(calc_step_3)
        return angle

    def _steps_to_angles(self, actuator_pos: torch.Tensor):
        phi = self._steps_to_phi(actuator_pos=actuator_pos)
        phi_0 = self._steps_to_phi(actuator_pos=torch.zeros(actuator_pos.shape))
        delta_phi = phi_0 - phi

        angles = (
            self._phi_0() + delta_phi if self.CLOCKWISE else self._phi_0() - delta_phi
        )
        return angles

    def _angles_to_steps(self, angles: torch.Tensor):
        delta_phi = angles - self._phi_0() if self.CLOCKWISE else self._phi_0() - angles

        phi_0 = self._steps_to_phi(actuator_pos=torch.zeros(angles.shape[0], 2))
        phi = phi_0 - delta_phi

        calc_step_3 = torch.cos(phi)
        calc_step_2 = 2.0 * self._actuator_offset() * self._joint_radius()
        calc_step_1 = calc_step_3 * calc_step_2
        stroke_length = torch.sqrt(
            self._actuator_offset() ** 2 + self._joint_radius() ** 2 - calc_step_1
        )
        actuator_steps = (
            stroke_length - self._initial_stroke_length()
        ) * self._increment()
        return actuator_steps

    def forward(self, actuator_pos: torch.Tensor) -> torch.Tensor:
        """

        :param actuator_pos:
        :return:
        """
        # Access actuator_pos via joint number: items in actuator_pos list have to be ordered by number
        return self._steps_to_angles(actuator_pos=actuator_pos)
