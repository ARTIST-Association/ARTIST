import h5py
import pytest
import torch
from pytest_mock import MockerFixture

from artist.field.tower_target_areas import TowerTargetAreas
from artist.util import config_dictionary


@pytest.mark.parametrize("curvature", [True, False])
def test_target_area_load_from_hdf5(
    mocker: MockerFixture,
    curvature: bool,
    device: torch.device,
) -> None:
    """
    Test the target area load from hdf5 method.

    Parameters
    ----------
    mocker : MockerFixture
        A pytest-mocker fixture used to create mock objects.
    curvature : bool
        Target area has a curvature or not.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    mock_h5_file = mocker.MagicMock(spec=h5py.File)

    # Define mocked objects for various keys.
    mock_geometry = mocker.MagicMock()
    mock_geometry.__getitem__.return_value = b"planar"

    mock_center = mocker.MagicMock()
    mock_center.__getitem__.return_value = [0.0, 0.0, 0.0, 1.0]

    mock_normal = mocker.MagicMock()
    mock_normal.__getitem__.return_value = [0.0, 1.0, 0.0, 0.0]

    mock_plane_e = mocker.MagicMock()
    mock_plane_e.__getitem__.return_value = 2.0

    mock_plane_u = mocker.MagicMock()
    mock_plane_u.__getitem__.return_value = 2.0

    mock_curvature_e = mocker.MagicMock()
    mock_curvature_e.__getitem__.return_value = 1.0

    mock_curvature_u = mocker.MagicMock()
    mock_curvature_u.__getitem__.return_value = 2.0

    mock_level_receiver = mocker.MagicMock()
    if curvature:
        mock_level_receiver.__getitem__.side_effect = lambda key: {
            config_dictionary.target_area_geometry: mock_geometry,
            config_dictionary.target_area_position_center: mock_center,
            config_dictionary.target_area_normal_vector: mock_normal,
            config_dictionary.target_area_plane_e: mock_plane_e,
            config_dictionary.target_area_plane_u: mock_plane_u,
            config_dictionary.target_area_curvature_e: mock_curvature_e,
            config_dictionary.target_area_curvature_u: mock_curvature_u,
        }[key]
        mock_level_receiver.keys.return_value = [
            config_dictionary.target_area_geometry,
            config_dictionary.target_area_position_center,
            config_dictionary.target_area_normal_vector,
            config_dictionary.target_area_plane_e,
            config_dictionary.target_area_plane_u,
            config_dictionary.target_area_curvature_e,
            config_dictionary.target_area_curvature_u,
        ]
    else:
        mock_level_receiver.__getitem__.side_effect = lambda key: {
            config_dictionary.target_area_geometry: mock_geometry,
            config_dictionary.target_area_position_center: mock_center,
            config_dictionary.target_area_normal_vector: mock_normal,
            config_dictionary.target_area_plane_e: mock_plane_e,
            config_dictionary.target_area_plane_u: mock_plane_u,
        }[key]

    mock_level_target_areas = mocker.MagicMock()
    mock_level_target_areas.__getitem__.side_effect = lambda key: {
        config_dictionary.target_area_receiver: mock_level_receiver
    }[key]
    mock_level_target_areas.keys.return_value = [config_dictionary.target_area_receiver]
    mock_level_target_areas.__len__.return_value = 1

    mock_h5_file.__getitem__.side_effect = lambda key: {
        config_dictionary.target_area_key: mock_level_target_areas
    }[key]

    # Perform the test.
    target_areas = TowerTargetAreas.from_hdf5(
        config_file=mock_h5_file,
        device=device,
    )

    assert isinstance(target_areas, TowerTargetAreas)
