import torch

from artist.field.actuator_ideal import IdealActuator
from artist.field.actuator_linear import LinearActuator
from artist.util import config_dictionary
from artist.util.configuration_classes import ActuatorListConfig

actuator_type_mapping = {
    config_dictionary.ideal_actuator_key: IdealActuator,
    config_dictionary.linear_actuator_key: LinearActuator,
}


class ActuatorArray(torch.nn.Module):
    """
    This class wraps a list of actuators as a torch.nn.Module to allow gradient calculation.

    Attributes
    ----------
    actuator_list : List[Actuator]

    """

    def __init__(self, actuator_list_config: ActuatorListConfig):
        """
        Initialize the heliostat field.

        Parameters
        ----------
        actuator_list_config : ActuatorListConfig
            The configuration parameters for the actuators
        """
        super(ActuatorArray, self).__init__()
        actuator_array = []
        for i, actuator_config in enumerate(actuator_list_config.actuator_list):
            try:
                actuator_object = actuator_type_mapping[actuator_config.actuator_type]
                if actuator_config.actuator_parameters is not None:
                    actuator_array.append(
                        actuator_object(
                            joint_number=i + 1,
                            clockwise=actuator_config.actuator_clockwise,
                            increment=actuator_config.actuator_parameters.increment,
                            initial_stroke_length=actuator_config.actuator_parameters.initial_stroke_length,
                            offset=actuator_config.actuator_parameters.offset,
                            radius=actuator_config.actuator_parameters.radius,
                            phi_0=actuator_config.actuator_parameters.phi_0,
                        )
                    )
                else:
                    actuator_array.append(
                        actuator_object(
                            joint_number=i + 1,
                            clockwise=actuator_config.actuator_clockwise,
                            increment=torch.tensor(0.0),
                            initial_stroke_length=torch.tensor(0.0),
                            offset=torch.tensor(0.0),
                            radius=torch.tensor(0.0),
                            phi_0=torch.tensor(0.0),
                        )
                    )
            except KeyError:
                raise KeyError(
                    f"Currently the selected actuator type: {actuator_config.actuator_type} is not supported."
                )

        self.actuator_list = actuator_array
