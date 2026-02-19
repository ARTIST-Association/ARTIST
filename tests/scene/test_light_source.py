import h5py
import pytest
import torch
from pytest_mock import MockerFixture

from artist.scene.light_source import LightSource

torch.manual_seed(7)
torch.cuda.manual_seed(7)


def test_load_light_source_from_hdf5(
    mocker: MockerFixture,
    device: torch.device,
) -> None:
    """
    Test abstract light source load from hdf5 file.

    Parameters
    ----------
    mocker : MockerFixture
        A pytest-mocker fixture used to create mock objects.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    mock_h5_file = mocker.MagicMock(spec=h5py.File)
    with pytest.raises(NotImplementedError) as exc_info:
        LightSource.from_hdf5(
            config_file=mock_h5_file,
            light_source_name="Sun",
            device=device,
        )
    assert "Must be overridden!" in str(exc_info.value)


def test_light_source_distortions() -> None:
    """
    Test the abstract light source distortions.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    light_source = LightSource(number_of_rays=4)

    with pytest.raises(NotImplementedError) as exc_info:
        light_source.get_distortions(number_of_points=40, number_of_active_heliostats=5)
    assert "Must be overridden!" in str(exc_info.value)
