import torch

from artist.physics_objects.module import AModule
from artist.physics_objects.heliostats.normalization import ParameterNormalizer
from artist.physics_objects.parameter import AParameter


class LinearActuatorModule(AModule):
    ## This class implements the behavior of a linear actuator that moves within a 2D plane as a pytorch module.
    #
    # The actuator is parametrized by 5 parameters:
    # - increment: stroke length change per motor step
    # - initial_stroke_length: stroke length for a motor step of 0
    # - actuator_offset: offset between the linear actuator's pivoting point and the point around which the actuator is allowed to pivot
    # - joint_radius: the actuator's pivoting radius
    # - phi_0 : the angle the actuator introduces to the manipulated coordinate system at the initial stroke length
    #
    # The actuator is modelled by the law of cosine. Two directions are implemented:
    # - motor steps -> angle
    # - angle -> motor steps

    #######################
    #   Parametrization   #
    #######################

    class DevRotationParameter(AParameter):
        """
        Sub-Component of the Heliokon Kinematic Module class that is used for handling rotation deviations.
        """

        def __init__(
            self,
            name,
            value=0.0,
            tolerance=0.01,
            requires_grad=True,
            distort: bool = False,
        ):
            """
            Standard constructor method.

            Parameters
            ----------
            name : str
                    The parameter name to be used within the module's state dict
            value : float
                    The default parameter value that will be returned for a parameter shift of 0.
            tolerance: float
                    The deviation in positive or negative direction in rad from the default value for a parameter shift of +/- 1.
            requires_grad: bool
                    Flag for PyTorch to mark the parameter as a gradient source.
            distort: bool
                    Flag for distorting the parameter value (adding an artificial error), used for simulating experiments.
            """
            super().__init__(value, tolerance, distort, requires_grad)
            self.NAME = name

    class DevPercentageParameter(AParameter):
        """
        Sub-Component of the Heliokon Kinematic Module class that is used for handling percentage deviations.
        """

        def __init__(
            self,
            name,
            value=0.0,
            tolerance=0.01,
            requires_grad=True,
            distort: bool = False,
        ):
            """
            Standard constructor method.

            Parameters
            ----------
            name : str
                    The parameter name to be used within the module's state dict
            value : float
                    The default parameter value that will be returned for a parameter shift of 0.
            tolerance: float
                    The deviation in positive or negative direction in % from the default value for a parameter shift of +/- 1.
            requires_grad: bool
                    Flag for PyTorch to mark the parameter as a gradient source.
            distort: bool
                    Flag for distorting the parameter value (adding an artificial error), used for simulating experiments.
            """
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
        """
        Auxiliary method for obtaining a percentage deviation by a parameter name.

        Parameters
        ----------
        parameter_name : str
                        The parameter's name representation.

        Returns
        -------
        torch.Tensor
                    percentage deviation value in %
        """
        return self.PARAMS[parameter_name] * (
            1 + self._get_parameter("dev_" + parameter_name)
        )

    def _rotation_with_deviation(self, parameter_name):
        """
        Auxiliary method for obtaining a rotation deviation by a parameter name.

        Parameters
        ----------
        parameter_name : str
                        The parameter's name representation.

        Returns
        -------
        torch.Tensor
                    rotation deviation value in rad
        """
        return self.PARAMS[parameter_name] + self._get_parameter(
            "dev_" + parameter_name
        )

    def _normed_percentage_with_deviation(self, parameter_name):
        """
        Auxiliary method for obtaining a normed percentage deviation by a parameter name.

        Parameters
        ----------
        parameter_name : str
                        The parameter's name representation.

        Returns
        -------
        torch.Tensor
                    normed percentage deviation value
        """
        return self.PARAMS[parameter_name].norm() * (
            1 + self._get_parameter("dev_" + parameter_name)
        )

    def _increment(self) -> torch.Tensor:
        """
        Getter for the actuator's increment.

        Returns
        -------
        torch.Tensor
                    Length increment in meters of the linear actuator per step.
        """
        return self._percentage_with_deviation("increment")

    def _initial_stroke_length(self) -> torch.Tensor:
        """
        Getter for the actuator's initial stroke length.

        Returns
        -------
        torch.Tensor
                    Stroke length in meters for an actuator step of 0 steps.
        """
        # return self._percentage_with_deviation('initial_stroke_length')
        return self.PARAMS["initial_stroke_length"]

    def _actuator_offset(self) -> torch.Tensor:
        """
        Getter for the actuator's offset between pivot point and actuating point.

        Returns
        -------
        torch.Tensor
                    Offset length in meters.
        """
        #    return self._percentage_with_deviation('actuator_offset')
        return self.PARAMS["actuator_offset"]

    def _joint_radius(self) -> torch.Tensor:
        """
        Getter for the actuator's radius between pivoting point and actuator attack point.

        Returns
        -------
        torch.Tensor
                    Radius length in meters.
        """
        # return self._percentage_with_deviation('joint_radius')
        return self.PARAMS["joint_radius"]

    def _phi_0(self) -> torch.Tensor:
        """
        Getter for the actuator's rotation angle for actuator steps of value 0.

        Returns
        -------
        torch.Tensor
                    Rotation angle in radians.
        """
        return self._rotation_with_deviation("phi_0")

    def _register_parameter(self, parameter: AParameter):
        """
        Auxiliary method for registrating parameters.

        Parameters
        ----------
        parameter : AParameter
                    parameter to be registered with the module.
        """
        if not hasattr(self, "parameter_normalizer"):
            self.parameter_normalizer = ParameterNormalizer()
        self.parameter_normalizer.register_parameter(parameter)

        self.register_parameter(
            parameter.NAME,
            torch.nn.Parameter(
                self.parameter_normalizer.get_normalized_parameter(
                    parameter.NAME, parameter.initial_value
                ),
                parameter.requires_grad,
            ),
        )

    def _get_parameter(self, name: str):
        """
        Auxiliary method for getting a module's parameter by name.

        Parameters
        ----------
        name : str
                the name of the parameter to get

        Returns
        -------
        torch.Tensor
                value of the denormalized parameter
        """
        return self.parameter_normalizer.get_denormalized_parameter(
            name, self.get_parameter(name)
        )

    #######################
    #     Constructor     #
    #######################

    def __init__(self, joint_number: int, clockwise: bool, params: dict, **deviations):
        """
        Standard constructor.

        Parameters
        ----------
        joint_number : int
                        Decides which input angle to get. (Should be deprecated!)
        clockwise : bool
                        Decides the actuated joints rotation direction. True = clockwise / left handed direction around the actuated axis.
        params : dict
                        Parameter values to be set.
        """
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

    #######################
    #       Methods       #
    #######################

    # TODO remove self.JOINT_NUMBER
    def _stepsToPhi(self, actuator_pos: torch.Tensor):
        """
        Computes a rotation offsets relative to the initial rotation from actuator steps.

        Parameters
        ----------
        actuator_pos : torch.Tensor
                        n x 2 tensor of actuator steps

        Returns
        -------
        torch.Tensor
                        n x 1 tensor of rotation offsets in radians.
        """
        # access actuator_pos via joint number: items in actuator_pos list have to be ordered by number
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

    def _stepsToAngles(self, actuator_pos: torch.Tensor):
        """
        Computes actuator rotations form steps.

        Parameters
        ----------
        actuator_pos : torch.Tensor
                        n x 2 tensor of actuator steps.

        Returns
        -------
        torch.Tensor
                        n x 1 tensor of rotations in radians.

        """
        phi = self._stepsToPhi(actuator_pos=actuator_pos)
        phi_0 = self._stepsToPhi(actuator_pos=torch.zeros(actuator_pos.shape))
        delta_phi = phi_0 - phi

        angles = (
            self._phi_0() + delta_phi if self.CLOCKWISE else self._phi_0() - delta_phi
        )
        return angles

    def _anglesToSteps(self, angles: torch.Tensor):
        """
        Computes actuator steps from rotation angles.

        Parameters
        ----------
        angles : torch.Tensor
                    n x 1 tensor of rotation angles in radians.

        Returns
        -------
        torch.Tensor
                    n x 1 tensor of actuator steps
        """
        delta_phi = angles - self._phi_0() if self.CLOCKWISE else self._phi_0() - angles

        phi_0 = self._stepsToPhi(actuator_pos=torch.zeros(angles.shape[0], 2))
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

    def forward(self, actuator_pos: torch.Tensor):
        """
        Computes actuator steps from rotation angles. Used for model training.

        Parameters
        ----------
        angles : torch.Tensor
                    n x 1 tensor of rotation angles in radians.

        Returns
        -------
        torch.Tensor
                    n x 1 tensor of actuator steps
        """

        # access actuator_pos via joint number: items in actuator_pos list have to be ordered by number
        return self._stepsToAngles(actuator_pos=actuator_pos)
