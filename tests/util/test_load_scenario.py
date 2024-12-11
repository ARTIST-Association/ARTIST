import h5py
import pytest
from pytest_mock import MockerFixture
import torch

from artist.field.heliostat import Heliostat


@pytest.fixture
def prototype_mock_generator(mocker: MockerFixture):
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
def test_heliostat_load_from_hdf5(
    mocker,
    prototype_surface,
    prototype_kinematic,
    error,
    device: torch.device,
) -> None:
    """
    Parameters
    ----------
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

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
