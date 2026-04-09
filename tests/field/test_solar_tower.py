

from unittest import mock

import pytest
import torch 
from artist.field.solar_tower import SolarTower
from artist.field.tower_target_areas_cylindrical import TowerTargetAreasCylindrical
from artist.field.tower_target_areas_planar import TowerTargetAreasPlanar

@pytest.mark.parametrize(
    "target_area_indices, expected",
    [
        (
            torch.tensor([0, 3]),
            torch.tensor(
            [[ 1.0,  1.0,  1.0,  1.0],
            [20.,  2.,  0.0,  1.0]])
        ),
        (
            torch.tensor([2, 3]),
            torch.tensor(
            [[ 11.0, 0.0,  0.0,  1.0],
            [20.,  2.,  0.0,  1.0]])
        ),


    ]
)
def test_get_centers_of_target_areas(
    target_area_indices,
    expected,
    device,
):

    mock_target_areas_planar = mock.MagicMock(spec=TowerTargetAreasPlanar)
    mock_target_areas_planar.names = ["plane_1", "plane_2"]
    mock_target_areas_planar.number_of_target_areas = 2
    mock_target_areas_planar.centers = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
        ],
        device=device,
    )

    mock_target_areas_cylindrical = mock.MagicMock(spec=TowerTargetAreasCylindrical)
    mock_target_areas_cylindrical.names = ["cylinder_1", "cylinder_2"]
    mock_target_areas_cylindrical.number_of_target_areas = 2
    mock_target_areas_cylindrical.centers = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0],
            [20.0, 0.0, 0.0, 0.0],
        ],
        device=device,
    )
    mock_target_areas_cylindrical.radii = torch.tensor([1.0, 2.0], device=device)
    mock_target_areas_cylindrical.normals = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        device=device,
    )

    solar_tower = SolarTower(
        target_areas=[mock_target_areas_planar, mock_target_areas_cylindrical],
        device=device,
    )

    centers = solar_tower.get_centers_of_target_areas(
        target_area_indices=target_area_indices.to(device),
        device=device,
    )
    
    torch.testing.assert_close(centers, expected.to(device), atol=1e-6, rtol=1e-6)