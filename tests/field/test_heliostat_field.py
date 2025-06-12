from typing import Union
from unittest.mock import MagicMock

import h5py
import pytest
import torch
from pytest_mock import MockerFixture

from artist.field.heliostat_field import HeliostatField
from artist.util import config_dictionary
from artist.util.configuration_classes import SurfaceConfig


@pytest.fixture
def prototype_mock_generator(mocker: MockerFixture) -> MagicMock:
    """
    Generate a mock prototype.

    Parameters
    ----------
    mocker : MockerFixture
        A pytest-mock fixture used to create mock objects.

    Returns
    -------
    MagicMock
        A mock prototype.
    """
    return mocker.MagicMock()


@pytest.mark.parametrize(
    "prototype_surface, prototype_kinematic, prototype_actuators, error",
    [
        (
            None,
            None,
            None,
            "If the heliostat does not have individual surface parameters, a surface prototype must be provided!",
        ),
        (
            prototype_mock_generator,
            None,
            None,
            "If the heliostat does not have an individual kinematic, a kinematic prototype must be provided!",
        ),
        (
            prototype_mock_generator,
            {
                config_dictionary.kinematic_type: config_dictionary.rigid_body_key,
                config_dictionary.kinematic_initial_orientation: torch.rand(4, 4),
                config_dictionary.kinematic_deviations: torch.rand(4, 4),
            },
            None,
            "If the heliostat does not have individual actuators, an actuator prototype must be provided!",
        ),
    ],
)
def test_heliostat_field_load_from_hdf5_errors(
    mocker: MockerFixture,
    prototype_surface: SurfaceConfig,
    prototype_kinematic: dict[str, Union[str, torch.Tensor]],
    prototype_actuators: dict[str, Union[str, torch.Tensor]],
    error: str,
    device: torch.device,
) -> None:
    """
    Test the heliostat field load from hdf5 method.

    Parameters
    ----------
    mocker : MockerFixture
        A pytest-mocker fixture used to create mock objects.
    prototype_surface : SurfaceConfig
        The mock prototype surface.
    prototype_kinematic : dict[str, Union[str, torch.Tensor]]
        The mock prototype kinematic.
    prototype_actuators : dict[str, Union[str, torch.Tensor]]
        The mock prototype actuator.
    error : str
        The expected error message.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    mock_h5_file = mocker.MagicMock(spec=h5py.File)
    mock_group_heliostats = mocker.MagicMock()
    mock_group_heliostats.keys.return_value = ["heliostat_1"]
    mock_h5_file.__getitem__.return_value = mock_group_heliostats
    mock_group_heliostats.__len__.return_value = 1
    mock_group_heliostats.keys.return_value = ["heliostats"]

    with pytest.raises(ValueError) as exc_info:
        HeliostatField.from_hdf5(
            config_file=mock_h5_file,
            prototype_surface=prototype_surface,
            prototype_kinematic=prototype_kinematic,
            prototype_actuators=prototype_actuators,
            device=device,
        )
    assert error in str(exc_info.value)
