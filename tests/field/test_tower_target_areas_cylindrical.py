from unittest import mock

import h5py
import torch

from artist.field.tower_target_areas_cylindrical import TowerTargetAreasCylindrical
from artist.util import config_dictionary


def test_target_area_load_from_hdf5(
    device: torch.device,
) -> None:
    """
    Test the cylindrical target area is correctly loaded from an HDF5 scenario file.

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

    mock_center = mock.MagicMock()
    mock_center.__getitem__.return_value = [0.0, 0.0, 0.0, 1.0]

    mock_normal = mock.MagicMock()
    mock_normal.__getitem__.return_value = [0.0, 1.0, 0.0, 0.0]

    mock_radius = mock.MagicMock()
    mock_radius.__getitem__.return_value = 4.0

    mock_height = mock.MagicMock()
    mock_height.__getitem__.return_value = 6.0

    mock_opening_angle = mock.MagicMock()
    mock_opening_angle.__getitem__.return_value = torch.pi

    mock_axis = mock.MagicMock()
    mock_axis.__getitem__.return_value = [0.0, 0.0, 1.0, 0.0]

    mock_level_cylindrical_target = mock.MagicMock()

    mock_level_cylindrical_target.__getitem__.side_effect = lambda key: {
        config_dictionary.target_area_cylinder_normal: mock_normal,
        config_dictionary.target_area_cylinder_center: mock_center,
        config_dictionary.target_area_cylinder_axis: mock_axis,
        config_dictionary.target_area_cylinder_radius: mock_radius,
        config_dictionary.target_area_cylinder_height: mock_height,
        config_dictionary.target_area_cylinder_opening_angle: mock_opening_angle,
    }[key]

    mock_level_target_areas = mock.MagicMock()
    mock_level_target_areas.__getitem__.side_effect = lambda key: {
        config_dictionary.target_area_receiver: mock_level_cylindrical_target
    }[key]
    mock_level_target_areas.keys.return_value = [config_dictionary.target_area_receiver]
    mock_level_target_areas.__len__.return_value = 1

    mock_h5_file.__getitem__.side_effect = lambda key: {
        config_dictionary.target_area_cylindrical_key: mock_level_target_areas
    }[key]

    # Perform the test.
    target_areas = TowerTargetAreasCylindrical.from_hdf5(
        config_file=mock_h5_file,
        device=device,
    )

    assert isinstance(target_areas, TowerTargetAreasCylindrical)
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
        target_areas.axes,
        torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=device),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        target_areas.radii,
        torch.tensor([4.0], device=device),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        target_areas.heights,
        torch.tensor([6.0], device=device),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        target_areas.opening_angles,
        torch.tensor([torch.pi], device=device),
        atol=1e-5,
        rtol=1e-5,
    )
    assert target_areas.number_of_target_areas == 1
