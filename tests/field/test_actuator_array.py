import pytest
import torch

from artist.field.actuator_array import ActuatorArray
from artist.util import config_dictionary
from artist.util.configuration_classes import ActuatorConfig, ActuatorListConfig


@pytest.fixture(
    params=[
        config_dictionary.ideal_actuator_key,
        config_dictionary.linear_actuator_key,
        "invalid",
    ]
)
def actuator_config(request: pytest.FixtureRequest) -> ActuatorConfig:
    """
    Define an actuator config.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The pytest fixture used to consider different test cases.

    Returns
    -------
    ActuatorConfig
        An actuator config.
    """
    return ActuatorConfig(
        key="actuator_1",
        type=request.param,
        clockwise_axis_movement=False,
    )


def test_actuator_array_errors(
    actuator_config: ActuatorConfig,
    device: torch.device,
) -> None:
    """
    Test that actuator array raises errors with improper initialization.

    Parameters
    ----------
    actuator_config : ActuatorConfig
        The actuator config with a specific actuator type.
    device : torch.device
        The device on which to initialize tensors (default is cuda).

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    actuator_list_config = ActuatorListConfig([actuator_config])
    if actuator_config.type == "invalid":
        with pytest.raises(KeyError) as exc_info:
            ActuatorArray(actuator_list_config, device=device)
        assert (
            f"Currently the selected actuator type: {actuator_config.type} is not supported."
            in str(exc_info.value)
        )
