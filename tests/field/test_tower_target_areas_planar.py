from unittest import mock

import h5py
import torch

from artist.field.tower_target_areas_planar import TowerTargetAreasPlanar
from artist.util import constants


def test_target_area_load_from_hdf5(
    device: torch.device,
) -> None:
    """
    Load planar target areas from mocked HDF5 and verify parsed tensors.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    target_name = constants.target_area_receiver
    mock_h5_file = mock.MagicMock(spec=h5py.File)

    mock_center = mock.MagicMock()
    mock_center.__getitem__.return_value = [0.0, 0.0, 0.0, 1.0]

    mock_normal = mock.MagicMock()
    mock_normal.__getitem__.return_value = [0.0, 1.0, 0.0, 0.0]

    mock_plane_e = mock.MagicMock()
    mock_plane_e.__getitem__.return_value = 2.0

    mock_plane_u = mock.MagicMock()
    mock_plane_u.__getitem__.return_value = 2.0

    mock_level_planar_target = mock.MagicMock()

    mock_level_planar_target.__getitem__.side_effect = lambda key: {
        constants.target_area_position_center: mock_center,
        constants.target_area_normal_vector: mock_normal,
        constants.target_area_plane_e: mock_plane_e,
        constants.target_area_plane_u: mock_plane_u,
    }[key]

    mock_level_target_areas = mock.MagicMock()
    mock_level_target_areas.__getitem__.side_effect = lambda key: {
        target_name: mock_level_planar_target
    }[key]
    mock_level_target_areas.keys.return_value = [target_name]
    mock_level_target_areas.__len__.return_value = 1

    mock_h5_file.__getitem__.side_effect = lambda key: {
        constants.target_area_planar_key: mock_level_target_areas
    }[key]

    # Perform the test.
    target_areas = TowerTargetAreasPlanar.from_hdf5(
        config_file=mock_h5_file,
        device=device,
    )

    assert isinstance(target_areas, TowerTargetAreasPlanar)
    assert target_areas.names == [target_name]
    torch.testing.assert_close(
        target_areas.centers,
        torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        target_areas.normals,
        torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        target_areas.dimensions,
        torch.tensor([[2.0, 2.0]], device=device),
        atol=1e-5,
        rtol=1e-5,
    )
    assert target_areas.number_of_target_areas == 1
