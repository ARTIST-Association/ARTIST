import logging
from unittest import mock

import h5py
import pytest
import torch

from artist.io import h5_scenario_parser
from artist.util import config_dictionary


@pytest.mark.parametrize(
    "kinematics_type",
    ["invalid_kinematics_type"],
)
def test_load_kinematics_deviations(kinematics_type: str, device: torch.device) -> None:
    """
    Test that unsupported kinematics types raise a ValueError when loading kinematics deviations from an HDF5 file.

    Parameters
    ----------
    kinematics_type : str
        The kinematics type to be tested.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    scenario_file = mock.MagicMock(spec=h5py.File)

    mock_level_1 = mock.MagicMock()

    scenario_file.__getitem__.side_effect = lambda key: {
        config_dictionary.heliostat_kinematics_key: mock_level_1
    }[key]

    log = mock.MagicMock(spec=logging.Logger)

    with pytest.raises(ValueError) as exc_info:
        h5_scenario_parser.kinematics_deviations(
            prototype=False,
            kinematics_type=kinematics_type,
            scenario_file=scenario_file,
            log=log,
            device=device,
        )

    assert "is not yet implemented" in str(exc_info.value)
    assert kinematics_type in str(exc_info.value)


@pytest.mark.parametrize(
    "actuator_type, expected_error_message_fragment",
    [
        (
            "invalid_actuator_type",
            "is not yet implemented",
        ),
        (
            "linear",
            "Expected 2 actuators, found 0 actuator(s).",
        ),
        (
            "ideal",
            "Expected 2 actuators, found 0 actuator(s).",
        ),
    ],
)
def test_load_actuator_parameters(
    actuator_type: str, expected_error_message_fragment: str, device: torch.device
) -> None:
    """
    Test that invalid actuator setup/type raises a ValueError when loading actuator parameters from an HDF5 file.

    Parameters
    ----------
    actuator_type : str
        The actuator type to be tested.
    expected_error_message_fragment : str
        The expected error message fragment.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    scenario_file = mock.MagicMock(spec=h5py.File)

    mock_level_actuators = mock.MagicMock()

    # Return an empty/unspecified actuator group to trigger count/type validation paths.
    scenario_file.__getitem__.side_effect = lambda key: {
        config_dictionary.heliostat_actuator_key: mock_level_actuators
    }[key]

    log = mock.MagicMock(spec=logging.Logger)

    with pytest.raises(ValueError) as exc_info:
        h5_scenario_parser.actuator_parameters(
            prototype=False,
            actuator_type=actuator_type,
            scenario_file=scenario_file,
            number_of_actuators=2,
            log=log,
            device=device,
        )
    assert expected_error_message_fragment in str(exc_info.value)
