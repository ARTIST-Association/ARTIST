import h5py
import torch
from pytest_mock import MockerFixture

from artist.field.tower_target_areas_planar import TowerTargetAreasPlanar
from artist.util import config_dictionary


def test_target_area_load_from_hdf5(
    mocker: MockerFixture,
    device: torch.device,
) -> None:
    """
    Test the target area load from hdf5 method.

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

    mock_center = mocker.MagicMock()
    mock_center.__getitem__.return_value = [0.0, 0.0, 0.0, 1.0]

    mock_normal = mocker.MagicMock()
    mock_normal.__getitem__.return_value = [0.0, 1.0, 0.0, 0.0]

    mock_plane_e = mocker.MagicMock()
    mock_plane_e.__getitem__.return_value = 2.0

    mock_plane_u = mocker.MagicMock()
    mock_plane_u.__getitem__.return_value = 2.0

    mock_level_planar_target = mocker.MagicMock()

    mock_level_planar_target.__getitem__.side_effect = lambda key: {
        config_dictionary.target_area_position_center: mock_center,
        config_dictionary.target_area_normal_vector: mock_normal,
        config_dictionary.target_area_plane_e: mock_plane_e,
        config_dictionary.target_area_plane_u: mock_plane_u,
    }[key]

    mock_level_target_areas = mocker.MagicMock()
    mock_level_target_areas.__getitem__.side_effect = lambda key: {
        config_dictionary.target_area_receiver: mock_level_planar_target
    }[key]
    mock_level_target_areas.keys.return_value = [config_dictionary.target_area_receiver]
    mock_level_target_areas.__len__.return_value = 1

    mock_h5_file.__getitem__.side_effect = lambda key: {
        config_dictionary.target_area_planar_key: mock_level_target_areas
    }[key]

    # Perform the test.
    target_areas = TowerTargetAreasPlanar.from_hdf5(
        config_file=mock_h5_file,
        device=device,
    )

    assert isinstance(target_areas, TowerTargetAreasPlanar)
    assert target_areas.names == [config_dictionary.target_area_receiver]
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
    assert(target_areas.number_of_target_areas == 1)

