# system dependencies
import torch
import typing
import os
import sys

# local dependencies
module_dir = os.path.abspath(os.path.join(__file__, os.pardir))
sys.path.append(module_dir)
# lib_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
# sys.path.append(lib_dir)
from ParametrizedModel import AbstractParametrizedModel


class AbstractActuator(AbstractParametrizedModel):
    class Keys(typing.NamedTuple):
        min_actuator_steps : str = 'min_actuator_steps'
        max_actuator_steps : str = 'max_actuator_steps'
        rotating_clockwise : str = 'rotating_clockwise'
        increment : str = 'increment'
        initial_angle : str = 'initial_angle'
    keys = Keys()

    class DistKeys(typing.NamedTuple):
        increment : str = 'increment'
        initial_angle : str = 'initial_angle'
    dist_keys = DistKeys()

    def __init__(self,

                 # parametrization
                 parameter_dict: typing.Dict[str, torch.Tensor] = {},
                 disturbance_dict: typing.Dict[str, torch.Tensor] = {},

                 # keys
                 min_actuator_steps_key : typing.Optional[str] = None,
                 max_actuator_steps_key : typing.Optional[str] = None,
                 rotating_clockwise_key : typing.Optional[str] = None,
                 increment_key: typing.Optional[str] = None,
                 increment_dist_key: typing.Optional[str] = None,
                 initial_angle_key: typing.Optional[str] = None,
                 initial_angle_dist_key: typing.Optional[str] = None,
                 
                 # disturbance factors
                 dist_factor_perc: typing.Optional[torch.Tensor] = None,
                 dist_factor_rot: typing.Optional[torch.Tensor] = None,

                 # pytorch config
                 device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.get_default_dtype(),
                ):
    
        super().__init__(
                        parameter_dict=parameter_dict,
                        disturbance_dict=disturbance_dict,
                        dist_factor_perc=dist_factor_perc,
                        dist_factor_rot=dist_factor_rot,
                        device=device,
                        dtype=dtype,
                        )

        self.keys = AbstractActuator.Keys(
            min_actuator_steps = min_actuator_steps_key if min_actuator_steps_key else AbstractActuator.keys.min_actuator_steps,
            max_actuator_steps = max_actuator_steps_key if max_actuator_steps_key else AbstractActuator.keys.max_actuator_steps,
            rotating_clockwise = rotating_clockwise_key if rotating_clockwise_key else AbstractActuator.keys.rotating_clockwise,
            increment = increment_key if increment_key else AbstractActuator.keys.increment,
            initial_angle = initial_angle_key if initial_angle_key else AbstractActuator.keys.initial_angle,
        )

        self.dist_keys = AbstractActuator.DistKeys(
            increment = increment_dist_key if increment_dist_key else AbstractActuator.dist_keys.increment,
            initial_angle = initial_angle_dist_key if initial_angle_dist_key else AbstractActuator.dist_keys.initial_angle,
        )

        # abstract class guard
        if type(self).__name__ == AbstractActuator.__name__:
            raise Exception("Don't implement an abstract class!")

    def angleFromSteps(self, actuator_steps: torch.Tensor) -> torch.Tensor:
        # abstract class guard
        raise Exception("Abstract method must be overridden!")

    def stepsFromAngle(self, angle: torch.Tensor) -> torch.Tensor:
        # abstract class guard
        raise Exception("Abstract method must be overridden!")

    ####################
    #-   Parameters   -#
    ####################
    def _increment(self):
        increment = self._percParam(parameter_key=self.keys.increment, disturbance_key=self.dist_keys.increment)
        return increment

    def _initial_angle(self):
        initial_angle = self._rotParam(parameter_key=self.keys.initial_angle, disturbance_key=self.dist_keys.initial_angle)
        return initial_angle

    def _min_actuator_steps(self) -> torch.Tensor:
        min_steps = self._tensorParam(parameter_key=self.keys.min_actuator_steps)
        return min_steps

    def _max_actuator_steps(self) -> torch.Tensor:
        max_steps = self._tensorParam(parameter_key=self.keys.max_actuator_steps)
        return max_steps

    def _rotates_clockwise(self) -> bool:
        rotates_clockwise = self._boolParam(parameter_key=self.keys.rotating_clockwise)
        return rotates_clockwise

class RotatoryActuator(AbstractActuator):

    def __init__(self,
                 max_actuator_steps: torch.Tensor,

                 # keys
                 min_actuator_steps_key : typing.Optional[str] = None,
                 max_actuator_steps_key : typing.Optional[str] = None,
                 rotating_clockwise_key : typing.Optional[str] = None,
                 increment_key: typing.Optional[str] = None,
                 increment_dist_key: typing.Optional[str] = None,
                 initial_angle_key: typing.Optional[str] = None,
                 initial_angle_dist_key: typing.Optional[str] = None,

                 # parametrization
                 parameter_dict: typing.Dict[str, torch.Tensor] = {},
                 disturbance_dict: typing.Dict[str, torch.Tensor] = {},

                 # disturbance factors
                 dist_factor_perc: typing.Optional[torch.Tensor] = None,
                 dist_factor_rot: typing.Optional[torch.Tensor] = None,

                 # pytorch config
                 device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.get_default_dtype(),
                ):
        super().__init__(
                        min_actuator_steps_key=min_actuator_steps_key,
                        max_actuator_steps_key=max_actuator_steps_key,
                        rotating_clockwise_key=rotating_clockwise_key,
                        increment_key=increment_key,
                        increment_dist_key=increment_dist_key,
                        initial_angle_key=initial_angle_key,
                        initial_angle_dist_key=initial_angle_dist_key,
                        max_actuator_steps=max_actuator_steps,
                        parameter_dict=parameter_dict,
                        disturbance_dict=disturbance_dict,
                        dist_factor_perc=dist_factor_perc,
                        dist_factor_rot=dist_factor_rot,
                        device=device,
                        dtype=dtype,
                        )
    ################
    #-   Angles   -#
    ################
    # override
    def angleFromSteps(self, actuator_steps: torch.Tensor) -> torch.Tensor:
        angle = self._initial_angle() + actuator_steps * self._increment()
        return angle

    # override
    def stepsFromAngle(self, angle: torch.Tensor) -> torch.Tensor:
        steps = (angle - self._initial_angle()) / self._increment() 
        return steps   

class LinearActuator(AbstractActuator):
    class Keys(typing.NamedTuple):
        min_actuator_steps : str    = 'min_actuator_steps'
        max_actuator_steps : str    = 'max_actuator_steps'
        rotating_clockwise : str    = 'rotating_clockwise'
        increment : str             = 'increment'
        initial_angle : str         = 'initial_angle'
        actuator_offset : str       = 'actuator_offset'
        joint_radius : str          = 'joint_radius'
        initial_stroke : str        = 'initial_stroke'
    keys = Keys()

    class DistKeys(typing.NamedTuple):
        increment : str             = 'increment'
        initial_angle : str         = 'initial_angle'
        actuator_offset : str       = 'actuator_offset'
        joint_radius : str          = 'joint_radius'
        initial_stroke : str        = 'initial_stroke'
    dist_keys = DistKeys()

    def __init__(self,
                 min_actuator_steps_key : typing.Optional[str] = None,
                 max_actuator_steps_key : typing.Optional[str] = None,
                 rotating_clockwise_key : typing.Optional[str] = None,
                 increment_key : typing.Optional[str] = None,
                 increment_dist_key : typing.Optional[str] = None,
                 initial_angle_key : typing.Optional[str] = None,
                 initial_angle_dist_key : typing.Optional[str] = None,
                 actuator_offset_key : typing.Optional[str] = None,
                 actuator_offset_dist_key : typing.Optional[str] = None,
                 joint_radius_key : typing.Optional[str] = None,
                 joint_radius_dist_key : typing.Optional[str] = None,
                 initial_stroke_key : typing.Optional[str] = None,
                 initial_stroke_dist_key : typing.Optional[str] = None,

                 # parametrization
                 parameter_dict: typing.Dict[str, torch.Tensor] = {},
                 disturbance_dict: typing.Dict[str, torch.Tensor] = {},

                 # disturbance factors
                 dist_factor_perc: typing.Optional[torch.Tensor] = None,
                 dist_factor_rot: typing.Optional[torch.Tensor] = None,

                 # pytorch config
                 device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.get_default_dtype(),
                 ):

        super().__init__(
                        min_actuator_steps_key=min_actuator_steps_key,
                        max_actuator_steps_key=max_actuator_steps_key,
                        rotating_clockwise_key=rotating_clockwise_key,
                        increment_key=increment_key,
                        increment_dist_key=increment_dist_key,
                        initial_angle_key=initial_angle_key,
                        initial_angle_dist_key=initial_angle_dist_key,
                        parameter_dict=parameter_dict,
                        disturbance_dict=disturbance_dict,
                        dist_factor_perc=dist_factor_perc,
                        dist_factor_rot=dist_factor_rot,
                        device=device,
                        dtype=dtype,
                        )

        # optionals
        self.keys = LinearActuator.Keys(
            min_actuator_steps = min_actuator_steps_key if min_actuator_steps_key else LinearActuator.keys.min_actuator_steps,
            max_actuator_steps = max_actuator_steps_key if max_actuator_steps_key else LinearActuator.keys.max_actuator_steps,
            rotating_clockwise = rotating_clockwise_key if rotating_clockwise_key else LinearActuator.keys.rotating_clockwise,
            increment = increment_key if increment_key else LinearActuator.keys.increment,
            initial_angle = initial_angle_key if initial_angle_key else LinearActuator.keys.initial_angle,
            actuator_offset = actuator_offset_key if actuator_offset_key else LinearActuator.keys.actuator_offset,
            joint_radius = joint_radius_key if joint_radius_key else LinearActuator.keys.joint_radius,
            initial_stroke = initial_stroke_key if initial_stroke_key else LinearActuator.keys.initial_stroke,
        )

        self.dist_keys = LinearActuator.DistKeys(
            increment = increment_dist_key if increment_dist_key else LinearActuator.dist_keys.increment,
            initial_angle = initial_angle_dist_key if initial_angle_dist_key else LinearActuator.dist_keys.initial_angle,
            actuator_offset = actuator_offset_dist_key if actuator_offset_dist_key else LinearActuator.dist_keys.actuator_offset,
            joint_radius = joint_radius_dist_key if joint_radius_dist_key else LinearActuator.dist_keys.joint_radius,
            initial_stroke = initial_stroke_dist_key if initial_stroke_dist_key else LinearActuator.dist_keys.initial_stroke,
        )

    ########################
    #-   Static Methods   -#
    ########################

    def parametersFromPositions(
                                actuator_origin: torch.Tensor,
                                initial_working_point: torch.Tensor,
                                joint_origin: torch.Tensor,
                                ) -> typing.List :

        actuator_offset = (actuator_origin - joint_origin).norm()
        joint_radius = (initial_working_point - joint_origin).norm()

        return actuator_offset, joint_radius

    ####################
    #-   Parameters   -#
    ####################
    def _actuator_offset(self):
        actuator_offset = self._percParam(parameter_key=self.keys.actuator_offset, disturbance_key=self.dist_keys.actuator_offset)
        return actuator_offset

    def _joint_radius(self):
        joint_radius = self._percParam(parameter_key=self.keys.joint_radius, disturbance_key=self.dist_keys.joint_radius)
        return joint_radius

    def _initial_stroke_length(self):
        initial_stroke_length = self._percParam(parameter_key=self.keys.initial_stroke, disturbance_key=self.dist_keys.initial_stroke)
        return initial_stroke_length

    def _rotating_dir(self):
        rotating_dir = -1 if self._rotates_clockwise() else 1
        return rotating_dir

    ################
    #-   Stroke   -#
    ################
    def _strokeLengthFromSteps(self, actuator_steps: torch.Tensor) -> torch.Tensor :
        stroke_length = actuator_steps / self._increment()
        assert(stroke_length != torch.tensor(torch.inf)), 'increment: ' + str(self._increment())
        return stroke_length

    def _stepsFromStrokeLength(self, stroke_length: torch.Tensor) -> torch.Tensor :
        steps = stroke_length * self._increment()
        return steps

    ################
    #-   Angles   -#
    ################
    def _phiFromStrokeLength(self, stroke_length: torch.Tensor) -> torch.Tensor:
        x = self._initial_stroke_length() + stroke_length
        calc_step_1 = self._actuator_offset()**2 + self._joint_radius()**2 - x**2
        calc_step_2 = 2.0 * self._actuator_offset() * self._joint_radius()
        calc_step_3 = calc_step_1 / calc_step_2
        phi = torch.arccos(calc_step_3)
        return phi

    def _angleFromStrokeLength(self, stroke_length: torch.Tensor) -> torch.Tensor:
        phi_0 = self._phiFromStrokeLength(stroke_length=torch.tensor(0.0, dtype=self._dtype, device=self._device))
        phi = self._phiFromStrokeLength(stroke_length=stroke_length)
        angle = phi_0 - phi
        return angle

    def _strokeLengthFromAngle(self, angle: torch.Tensor) -> torch.Tensor:
        phi_0 = self._phiFromStrokeLength(stroke_length=torch.tensor(0.0, dtype=self._dtype, device=self._device))
        phi = phi_0 - angle
        calc_step_2 = 2.0 * self._actuator_offset() * self._joint_radius()
        calc_step_1 = torch.cos(phi) * calc_step_2
        x2 = self._actuator_offset()**2 + self._joint_radius()**2 - calc_step_1
        stroke_length = torch.sqrt(x2) -  self._initial_stroke_length()
        return stroke_length

    # override
    def angleFromSteps(self, actuator_steps: torch.Tensor) -> torch.Tensor:
        stroke_length = self._strokeLengthFromSteps(actuator_steps=actuator_steps)
        angle = self._angleFromStrokeLength(stroke_length=stroke_length) + self._initial_angle()
        angle = angle * self._rotating_dir()
        assert (not torch.isnan(angle)), 'stroke_length: ' + str(stroke_length)
        return angle

    # override
    def stepsFromAngle(self, angle: torch.Tensor) -> torch.Tensor:
        angle = angle * self._rotating_dir() - self._initial_angle()
        stroke_length = self._strokeLengthFromAngle(angle=angle)
        steps = self._stepsFromStrokeLength(stroke_length=stroke_length)
        
        assert (not torch.isnan(steps)), 'stroke_length: ' + str(stroke_length)
        return steps


#################
#-   Testing   -#
#################
if __name__ == '__main__':
    print('success')