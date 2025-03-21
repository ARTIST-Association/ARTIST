import logging

import h5py
import pytest
import torch
from pytest_mock import MockerFixture

from artist.util import config_dictionary, utils_load_h5


@pytest.mark.parametrize(
    "kinematic_type",
    [(
        "invalid_kinematic_type"
    )],
)
def test_load_kinematic_deviations(
        mocker: MockerFixture,
        kinematic_type: str, 
        device: torch.device
    ):
    """
    Test errors raised when loading kinematic deviations from an hdf5 file.

    Parameters
    ----------
    mocker : MockerFixture
        A pytest-mocker fixture used to create mock objects.
    kinematic_type : str
        The kinematic type to be tested.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    scenario_file = mocker.MagicMock(spec=h5py.File)

    mock_level_1 = mocker.MagicMock()

    scenario_file.__getitem__.side_effect = lambda key: {
        config_dictionary.heliostat_kinematic_key: mock_level_1
    }[key]

    log = mocker.MagicMock(spec=logging.Logger)

    with pytest.raises(ValueError) as exc_info:
        utils_load_h5.kinematic_deviations(
            prototype=False,
            kinematic_type=kinematic_type,
            scenario_file=scenario_file,
            log=log,
            device=device,
        )

    assert f"The kinematic type: {kinematic_type} is not yet implemented!" in str(
        exc_info.value
    )



@pytest.mark.parametrize(
    "actuator_type, error_message",
    [
        (
            "invalid_actuator_type",
            "The actuator type: invalid_actuator_type is not yet implemented!"
        ),
        (
            "linear",
            "This scenario file contains the wrong amount of actuators for this heliostat and its kinematic type. Expected 2 actuators, found 1 actuator(s)."
        )
    ],
)
def test_load_actuator_parameters(
        mocker: MockerFixture,
        actuator_type: str, 
        error_message: str,
        device: torch.device
    ):
    """
    Test errors raised when loading actuator parameters from an hdf5 file.

    Parameters
    ----------
    mocker : MockerFixture
        A pytest-mocker fixture used to create mock objects.
    actuator_type : str
        The actuator type to be tested.
    error_message : str
        The expected error message.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    scenario_file = mocker.MagicMock(spec=h5py.File)

    mock_level_2 = mocker.MagicMock()

    mock_level_1 = mocker.MagicMock()
    mock_level_1.__getitem__.side_effect = lambda key: {
        config_dictionary.sun_key: mock_level_2
    }[key]
    mock_level_1.keys.return_value = [config_dictionary.sun_key]

    scenario_file.__getitem__.side_effect = lambda key: {
        config_dictionary.heliostat_actuator_key: mock_level_1
    }[key]

    log = mocker.MagicMock(spec=logging.Logger)

    with pytest.raises(ValueError) as exc_info:
        utils_load_h5.actuator_parameters(
            prototype=False,
            actuator_type=actuator_type,
            scenario_file=scenario_file,
            number_of_actuators=2,
            initial_orientation=torch.tensor([0.0, -1.0, 0.0, 0.0]),
            log=log,
            device=device,
        )
    assert error_message in str(
        exc_info.value
    )

