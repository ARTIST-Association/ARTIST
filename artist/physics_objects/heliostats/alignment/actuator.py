import torch

from artist.physics_objects.heliostats.normalization import ParameterNormalizer
from artist.physics_objects.module import AModule
from artist.physics_objects.parameter import AParameter


class ActuatorModule(AModule):
    """
    A linear actuator moving within a 2D plane implemented as a ``PyTorch`` module.

    The actuator is parametrized by five parameters:
    - increment: stroke length change per motor step
    - initial_stroke_length: stroke length for a motor step of 0
    - actuator_offset: offset between the linear actuator's pivoting point and the point around which the actuator is
    allowed to pivot
    - joint_radius: the actuator's pivoting radius
    - phi_0 : the angle the actuator introduces to the manipulated coordinate system at the initial stroke length

    The actuator is modelled by the law of cosine. Two directions are implemented:
    - motor steps -> angle
    - angle -> motor steps

    Attributes
    ----------
    DEV_PARAMETERS : dict[str, Union[DevRotationParameter, DevPercentageParameter]]
        [INSERT EXPLANATION HERE!]
    clockwise : bool
        [INSERT EXPLANATION HERE!]
    deviations : dict[str, Any]
        [INSERT EXPLANATION HERE!]
    joint_number : int
        [INSERT EXPLANATION HERE!]
    parameter_deviations : dict
        [INSERT EXPLANATION HERE!]
    parameter_normalizer : ParameterNormalizer
        [INSERT EXPLANATION HERE!]
    params : dict
        [INSERT EXPLANATION HERE!]

    Methods
    -------
    forward()
        The forward kinematic.

    See Also
    --------
    :class: AModule : Reference to the parent class.
    """

    class DevRotationParameter(AParameter):
        """
        Deviation rotation parameter.

        [INSERT DESCRIPTION HERE!]

        Attributes
        ----------
        name : str
            [INSERT DESCRIPTION HERE!]
        has_tolerance : bool
            [INSERT DESCRIPTION HERE!]
        initial_value : torch.Tensor
            [INSERT DESCRIPTION HERE!]
        max : Union[torch.Tensor, float]
            [INSERT DESCRIPTION HERE!]
        min : Union[torch.Tensor, float]
            [INSERT DESCRIPTION HERE!]
        requires_grad : bool
            True if gradient calculation is required, else False.
        tolerance : float
            [INSERT DESCRIPTION HERE!]

        Methods
        -------
        distort()
            [INSERT DESCRIPTION HERE!]

        See Also
        --------
        :class: AParameter : Reference to the parent class.

        """

        def __init__(
            self,
            name: str,
            value: float = 0.0,
            tolerance: float = 0.01,
            requires_grad: bool = True,
            distort: bool = False,
        ) -> None:
            """
            [INSERT DESCRIPTION HERE!].

            Parameters
            ----------
            name : str
                [INSERT DESCRIPTION HERE!]
            value : float
                [INSERT DESCRIPTION HERE!]
            tolerance : float
                [INSERT DESCRIPTION HERE!]
            requires_grad : bool
                True if gradient calculation is required, else False.
            distort : bool
                [INSERT DESCRIPTION HERE!]
            """
            super().__init__(value, tolerance, distort, requires_grad)
            self.name = name

    class DevPercentageParameter(AParameter):
        """
        Deviation percentage parameter.

        [INSERT DESCRIPTION HERE!]

        Attributes
        ----------
        name : str
            [INSERT DESCRIPTION HERE!]
        has_tolerance : bool
            [INSERT DESCRIPTION HERE!]
        initial_value : torch.Tensor
            [INSERT DESCRIPTION HERE!]
        max : Union[torch.Tensor, float]
            [INSERT DESCRIPTION HERE!]
        min : Union[torch.Tensor, float]
            [INSERT DESCRIPTION HERE!]
        requires_grad : bool
            True if gradient calculation is required, else False.
        tolerance : float
            [INSERT DESCRIPTION HERE!]

        Methods
        -------
        distort()
            [INSERT DESCRIPTION HERE!]

        See Also
        --------
        :class: AParameter : Reference to the parent class.
        """

        def __init__(
            self,
            name: str,
            value: float = 0.0,
            tolerance: float = 0.01,
            requires_grad: bool = True,
            distort: bool = False,
        ) -> None:
            """
            [INSERT DESCRIPTION HERE!].

            Parameters
            ----------
            name : str
                [INSERT DESCRIPTION HERE!]
            value : float
                [INSERT DESCRIPTION HERE!]
            tolerance : float
                [INSERT DESCRIPTION HERE!]
            requires_grad : bool
                [INSERT DESCRIPTION HERE!]
            distort : bool
                [INSERT DESCRIPTION HERE!]
            """
            super().__init__(value, tolerance, distort, requires_grad)
            self.name = name

    DEV_PARAMETERS = {
        "dev_increment": DevPercentageParameter("dev_increment"),  # * 99%/101% => 1
        "dev_phi_0": DevRotationParameter("dev_phi_0"),  # +/- 10mRad
    }

    def _percentage_with_deviation(self, parameter_name: str) -> torch.Tensor:
        """
        [INSERT DESCRIPTION HERE!].

        Parameters
        ----------
        parameter_name : str
            [INSERT DESCRIPTION HERE!]

        Returns
        -------
        torch.Tensor
            [INSERT DESCRIPTION HERE!]
        """
        return self.params[parameter_name] * (
            1 + self._get_parameter("dev_" + parameter_name)
        )

    def _rotation_with_deviation(self, parameter_name: str) -> torch.Tensor:
        """
        [INSERT DESCRIPTION HERE!].

        Parameters
        ----------
        parameter_name : str
            [INSERT DESCRIPTION HERE!]

        Returns
        -------
        torch.Tensor
            [INSERT DESCRIPTION HERE!]
        """
        return self.params[parameter_name] + self._get_parameter(
            "dev_" + parameter_name
        )

    def _normed_percentage_with_deviation(self, parameter_name: str) -> torch.Tensor:
        """
        [INSERT DESCRIPTION HERE!].

        Parameters
        ----------
        parameter_name : str
            [INSERT DESCRIPTION HERE!]

        Returns
        -------
        torch.Tensor
            [INSERT DESCRIPTION HERE!]
        """
        return self.params[parameter_name].norm() * (
            1 + self._get_parameter("dev_" + parameter_name)
        )

    def _increment(self) -> torch.Tensor:
        """
        [INSERT DESCRIPTION HERE!].

        Returns
        -------
        torch.Tensor
            [INSERT DESCRIPTION HERE!]
        """
        return self._percentage_with_deviation("increment")

    def _initial_stroke_length(self) -> torch.Tensor:
        """
        [INSERT DESCRIPTION HERE!].

        Returns
        -------
        torch.Tensor
            [INSERT DESCRIPTION HERE!]
        """
        return self.params["initial_stroke_length"]

    def _actuator_offset(self) -> torch.Tensor:
        """
        [INSERT DESCRIPTION HERE!].

        Returns
        -------
        torch.Tensor
            [INSERT DESCRIPTION HERE!]
        """
        return self.params["actuator_offset"]

    def _joint_radius(self) -> torch.Tensor:
        """
        [INSERT DESCRIPTION HERE!].

        Returns
        -------
        torch.Tensor
            [INSERT DESCRIPTION HERE!]
        """
        return self.params["joint_radius"]

    def _phi_0(self) -> torch.Tensor:
        """
        [INSERT DESCRIPTION HERE!].

        Returns
        -------
        torch.Tensor
            [INSERT DESCRIPTION HERE!]
        """
        return self._rotation_with_deviation("phi_0")

    def _register_parameter(self, parameter: AParameter) -> None:
        """
        [INSERT DESCRIPTION HERE!].

        Parameters
        ----------
        parameter : AParameter
            [INSERT DESCRIPTION HERE!]
        """
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

    def _get_parameter(self, name: str) -> torch.Tensor:
        """
        [INSERT DESCRIPTION HERE!].

        Parameters
        ----------
        name : str
            [INSERT DESCRIPTION HERE!]

        Returns
        -------
        torch.Tensor
            [INSERT DESCRIPTION HERE!]
        """
        return self.parameter_normalizer.get_denormalized_parameter(
            name, self.get_parameter(name)
        )

    def __init__(
        self, joint_number: int, clockwise: bool, params: dict, **deviations
    ) -> None:
        """
        [INSERT DESCRIPTION HERE!].

        Parameters
        ----------
        joint_number : int
            [INSERT DESCRIPTION HERE!]
        clockwise : bool
            [INSERT DESCRIPTION HERE!]
        params : dict
            [INSERT DESCRIPTION HERE!]
        """
        super().__init__()
        self.joint_number = joint_number
        self.clockwise = clockwise
        self.params = params
        self.deviations = deviations

        self.parameter_deviations = {
            param: deviations.get(param_name)
            for param_name, param in self.DEV_PARAMETERS.items()
        }
        for param in self.parameter_deviations:
            # register and normalize deviations
            self._register_parameter(param)

    # TODO remove self.joint_number

    def _steps_to_phi(self, actuator_pos: torch.Tensor) -> torch.Tensor:
        """
        [INSERT DESCRIPTION HERE!].

        Parameters
        ----------
        actuator_pos : torch.Tensor
            [INSERT DESCRIPTION HERE!]

        Returns
        -------
        torch.Tensor
            [INSERT DESCRIPTION HERE!]
        """
        # Access actuator_pos via joint number: items in actuator_pos list have to be ordered by number
        stroke_length = (
            actuator_pos[:, self.joint_number - 1] / self._increment()
            + self._initial_stroke_length()
        )
        calc_step_1 = (
            self._actuator_offset() ** 2 + self._joint_radius() ** 2 - stroke_length**2
        )
        calc_step_2 = 2.0 * self._actuator_offset() * self._joint_radius()
        calc_step_3 = calc_step_1 / calc_step_2
        angle = torch.arccos(calc_step_3)
        return angle

    def _steps_to_angles(self, actuator_pos: torch.Tensor) -> torch.Tensor:
        """
        Translate the actuator steps to angles.

        Parameters
        ----------
        actuator_pos : torch.Tensor
            The current position of the actuator.

        Returns
        -------
        torch.Tensor
            The angles corresponding to the actuator steps.
        """
        phi = self._steps_to_phi(actuator_pos=actuator_pos)
        phi_0 = self._steps_to_phi(actuator_pos=torch.zeros(actuator_pos.shape))
        delta_phi = phi_0 - phi

        angles = (
            self._phi_0() + delta_phi if self.clockwise else self._phi_0() - delta_phi
        )
        return angles

    def _angles_to_steps(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Translate the angles to actuator steps.

        Parameters
        ----------
        angles : torch.Tensor
            The angles that are to be converted.

        Returns
        -------
        torch.Tensor
            The actuator steps corresponding to the given angles.
        """
        delta_phi = angles - self._phi_0() if self.clockwise else self._phi_0() - angles

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
        Perform the forward kinematic.

        Parameters
        ----------
        actuator_pos : torch.Tensor
            The position of the actuator.

        Returns
        -------
        torch.Tensor
            The required angles.
        """
        # Access actuator_pos via joint number: items in actuator_pos list have to be ordered by number
        return self._steps_to_angles(actuator_pos=actuator_pos)
