from typing import Optional
from unittest.mock import MagicMock

import h5py
import pytest
import torch
from pytest_mock import MockerFixture

from artist.field.heliostat import Heliostat


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
    "prototype_surface, prototype_kinematic, error",
    [
        (
            None,
            None,
            "If the heliostat does not have individual surface parameters, a surface prototype must be provided!",
        ),
        (
            prototype_mock_generator,
            None,
            "If the heliostat does not have an individual kinematic, a kinematic prototype must be provided!",
        ),
        (
            prototype_mock_generator,
            prototype_mock_generator,
            "If the heliostat does not have individual actuators, an actuator prototype must be provided!",
        ),
    ],
)
def test_heliostat_load_from_hdf5_errors(
    mocker: MockerFixture,
    prototype_surface: Optional[MagicMock],
    prototype_kinematic: Optional[MagicMock],
    error: str,
    device: torch.device,
) -> None:
    """
    Test the heliostat load from hdf5 method.

    Parameters
    ----------
    mocker : MockerFixture
        A pytest-mocker fixture used to create mock objects.
    prototype_surface : Optional[MagicMock]
        The mock prototype surface.
    prototype_kinematic : Optional[MagicMock]
        The mock prototype kinematic.
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

    with pytest.raises(ValueError) as exc_info:
        Heliostat.from_hdf5(
            config_file=mock_h5_file,
            prototype_surface=prototype_surface,
            prototype_kinematic=prototype_kinematic,
            device=device,
        )
    assert error in str(exc_info.value)
