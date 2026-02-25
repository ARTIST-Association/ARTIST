import logging

import h5py
import pytest
import torch
from pytest_mock import MockerFixture

from artist.data_parser import h5_scenario_parser
from artist.util import config_dictionary


@pytest.mark.parametrize(
    "kinematics_type",
    [("invalid_kinematics_type")],
)
def test_load_kinematics_deviations(
    mocker: MockerFixture, kinematics_type: str, device: torch.device
) -> None:
    """
    Test errors raised when loading kinematics deviations from an hdf5 file.

    Parameters
    ----------
    mocker : MockerFixture
        A pytest-mocker fixture used to create mock objects.
    kinematics_type : str
        The kinematics type to be tested.
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
        config_dictionary.heliostat_kinematics_key: mock_level_1
    }[key]

    log = mocker.MagicMock(spec=logging.Logger)

    with pytest.raises(ValueError) as exc_info:
        h5_scenario_parser.kinematics_deviations(
            prototype=False,
            kinematics_type=kinematics_type,
            scenario_file=scenario_file,
            log=log,
            device=device,
        )

    assert f"The kinematics type: {kinematics_type} is not yet implemented!" in str(
        exc_info.value
    )


@pytest.mark.parametrize(
    "actuator_type, error_message",
    [
        (
            "invalid_actuator_type",
            "The actuator type: invalid_actuator_type is not yet implemented!",
        ),
        (
            "linear",
            "This scenario file contains the wrong amount of actuators for this heliostat and its kinematics type. Expected 2 actuators, found 0 actuator(s).",
        ),
        (
            "ideal",
            "This scenario file contains the wrong amount of actuators for this heliostat and its kinematics type. Expected 2 actuators, found 0 actuator(s).",
        ),
    ],
)
def test_load_actuator_parameters(
    mocker: MockerFixture, actuator_type: str, error_message: str, device: torch.device
) -> None:
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

    mock_level_actuators = mocker.MagicMock()

    scenario_file.__getitem__.side_effect = lambda key: {
        config_dictionary.heliostat_actuator_key: mock_level_actuators
    }[key]

    log = mocker.MagicMock(spec=logging.Logger)

    with pytest.raises(ValueError) as exc_info:
        h5_scenario_parser.actuator_parameters(
            prototype=False,
            actuator_type=actuator_type,
            scenario_file=scenario_file,
            number_of_actuators=2,
            log=log,
            device=device,
        )
    assert error_message in str(exc_info.value)
