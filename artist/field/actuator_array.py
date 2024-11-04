import logging

import torch

from artist.field.actuator_ideal import IdealActuator
from artist.field.actuator_linear import LinearActuator
from artist.util import config_dictionary
from artist.util.configuration_classes import ActuatorListConfig

actuator_type_mapping = {
    config_dictionary.ideal_actuator_key: IdealActuator,
    config_dictionary.linear_actuator_key: LinearActuator,
}
"""A type mapping dictionary that allows ``ARTIST`` to automatically infer the correct actuator type."""

log = logging.getLogger(__name__)
"""A logger for the actuators."""


class ActuatorArray(torch.nn.Module):
    """
    Wrap the list of actuators as a ``torch.nn.Module`` to allow gradient calculation.

    Attributes
    ----------
    actuator_list : List[Actuator]
        The list of actuators to be wrapped.
    """

    def __init__(
        self,
        actuator_list_config: ActuatorListConfig,
        device: torch.device = "cpu",
    ) -> None:
        """
        Initialize the actuator array.

        An actuator array is used to bundle all actuators for a specific heliostat. A heliostat can have one or
        more actuators. Different actuator types exist. The actuators are created according to their
        ``actuator_config``. If the actuator config does not contain actuator parameters, an actuator with default
        values will be initialized.

        Parameters
        ----------
        actuator_list_config : ActuatorListConfig
            The configuration parameters for the actuators.
        device : torch.device
            The device on which to initialize tensors (default is CPU).
        """
        super().__init__()
        actuator_array = []
        # Iterate through each actuator configuration in the list of actuator configurations.
        for i, actuator_config in enumerate(actuator_list_config.actuator_list):
            # Try to load an actuator from the given configuration. This will fail, if ARTIST
            # does not recognize the actuator type defined in the configuration.
            try:
                actuator_object = actuator_type_mapping[actuator_config.actuator_type]
                # Check if the actuator configuration contains actuator parameters and initialize an actuator with
                # these parameters.
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
                # If the actuator config does not contain actuator parameters, initialize an actuator with default
                # values.
                else:
                    log.warning(
                        "No actuator parameters provided. Loading an actuator with default parameters!"
                    )
                    actuator_array.append(
                        actuator_object(
                            joint_number=i + 1,
                            clockwise=actuator_config.actuator_clockwise,
                            increment=torch.tensor(0.0, device=device),
                            initial_stroke_length=torch.tensor(0.0, device=device),
                            offset=torch.tensor(0.0, device=device),
                            radius=torch.tensor(0.0, device=device),
                            phi_0=torch.tensor(0.0, device=device),
                        )
                    )
            except KeyError:
                raise KeyError(
                    f"Currently the selected actuator type: {actuator_config.actuator_type} is not supported."
                )

        self.actuator_list = actuator_array
