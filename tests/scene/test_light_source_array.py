from unittest import mock

import h5py
import pytest
import torch

from artist.scene.light_source_array import LightSourceArray
from artist.util import config_dictionary


def test_light_source_array_load_from_hdf5_errors(
    device: torch.device,
) -> None:
    """
    Test errors raised when loading a light source array from an hdf5 file.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    mock_h5_file = mock.MagicMock(spec=h5py.File)

    mock_level_3 = mock.MagicMock()
    mock_level_3.__getitem__.return_value = b"laser"

    mock_level_2 = mock.MagicMock()
    mock_level_2.__getitem__.side_effect = lambda key: {
        config_dictionary.light_source_type: mock_level_3
    }[key]
    mock_level_2.keys.return_value = [config_dictionary.light_source_type]

    mock_level_1 = mock.MagicMock()
    mock_level_1.__getitem__.side_effect = lambda key: {
        config_dictionary.sun_key: mock_level_2
    }[key]
    mock_level_1.keys.return_value = [config_dictionary.sun_key]

    mock_h5_file.__getitem__.side_effect = lambda key: {
        config_dictionary.light_source_key: mock_level_1
    }[key]

    with pytest.raises(KeyError) as exc_info:
        LightSourceArray.from_hdf5(
            config_file=mock_h5_file,
            device=device,
        )
    assert "Currently the selected light source: laser is not supported." in str(
        exc_info.value
    )
