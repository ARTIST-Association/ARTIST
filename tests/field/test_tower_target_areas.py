from unittest import mock

import h5py
import pytest
import torch

from artist.field.tower_target_areas import TowerTargetAreas


def test_abstract_tower_target_areas(
    device: torch.device,
) -> None:
    """
    Test the abstract methods of tower target areas.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    with pytest.raises(NotImplementedError) as exc_info:
        TowerTargetAreas.from_hdf5(
            config_file=mock.MagicMock(spec=h5py.File),
            device=device,
        )
    assert "Must be overridden!" in str(exc_info.value)
