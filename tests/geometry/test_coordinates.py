import math
from unittest.mock import MagicMock

import pytest
import torch

from artist.field import SolarTower, TowerTargetAreasCylindrical, TowerTargetAreasPlanar
from artist.geometry import coordinates


@pytest.mark.parametrize(
    "point, expected",
    [
        (
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0, 1.0]),
        ),
        (
            torch.tensor([1.0, 0.0]),
            None,
        ),
        (
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            None,
        ),
    ],
)
def test_3d_point_converter(
    point: torch.Tensor, expected: torch.Tensor | None, device: torch.device
) -> None:
    """
    Test the 3d-to-4d point converter.

    Parameters
    ----------
    point : torch.Tensor
        A 3d point.
    expected : torch.Tensor | None
        A 4d point or ``None`` if an error is expected.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    # Check if the ValueError is thrown as expected.
    if expected is None:
        with pytest.raises(ValueError) as exc_info:
            coordinates.convert_3d_points_to_4d_format(
                points=point.to(device),
                device=device,
            )
        assert f"Expected 3D points but got points of shape {point.shape}!" in str(
            exc_info.value
        )
    else:
        # Check if the 4d point is correct.
        point_4d = coordinates.convert_3d_points_to_4d_format(
            points=point.to(device),
            device=device,
        )
        torch.testing.assert_close(point_4d, expected.to(device), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "direction, expected",
    [
        (
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
        ),
        (
            torch.tensor([1.0, 0.0]),
            None,
        ),
        (
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            None,
        ),
    ],
)
def test_3d_direction_converter(
    direction: torch.Tensor, expected: torch.Tensor | None, device: torch.device
) -> None:
    """
    Test the 3d to 4d point converter.

    Parameters
    ----------
    direction : torch.Tensor
        A 3d direction.
    expected : torch.Tensor | None
        A 4d direction or ``None`` if an error is expected.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    # Check if the ValueError is thrown as expected.
    if expected is None:
        with pytest.raises(ValueError) as exc_info:
            coordinates.convert_3d_directions_to_4d_format(
                directions=direction.to(device),
                device=device,
            )
        assert (
            f"Expected 3D directions but got directions of shape {direction.shape}!"
            in str(exc_info.value)
        )
    else:
        # Check if the 4d point is correct.
        direction_4d = coordinates.convert_3d_directions_to_4d_format(
            directions=direction.to(device),
            device=device,
        )
        torch.testing.assert_close(
            direction_4d, expected.to(device), rtol=1e-4, atol=1e-4
        )


@pytest.mark.parametrize(
    "bitmap_coordinates, bitmap_resolution, target_area_indices, expected_coordinates",
    [
        (
            torch.tensor([[127.5, 127.5], [63.75, 255.0], [0.0, 0.0]]),
            torch.tensor([256, 256]),
            torch.tensor([0, 0, 1]),
            torch.tensor(
                [
                    [0.0000, 0.0000, 0.0000, 1.0000],
                    [1.4941, 0.0000, -2.9883, 1.0000],
                    [1.9961, 0.0000, 3.9922, 1.0000],
                ]
            ),
        ),
        (
            torch.tensor([[127.5, 127.5], [127.5, 255.0], [0.0, 63.75]]),
            torch.tensor([256, 256]),
            torch.tensor([2, 2, 2]),
            torch.tensor(
                [
                    [0.0000, 2.0000, 0.0000, 1.0000],
                    [0.0000, 2.0000, -2.9883, 1.0000],
                    [2.0000, 0.0123, 1.4941, 1.0000],
                ]
            ),
        ),
        (
            torch.tensor([[255.0, 191.25], [255.0, 255.0]]),
            torch.tensor([256, 256]),
            torch.tensor([2, 0]),
            torch.tensor(
                [[-2.0000, 0.0123, -1.4941, 1.0000], [-2.9883, 0.0000, -2.9883, 1.0000]]
            ),
        ),
    ],
)
def test_bitmap_coordinates_to_target_coordinates(
    bitmap_coordinates: torch.Tensor,
    bitmap_resolution: torch.Tensor,
    target_area_indices: torch.Tensor,
    expected_coordinates: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test the conversion from bitmap coordinates to target coordinates.

    Parameters
    ----------
    bitmap_coordinates : torch.Tensor
        The 2D pixel coordinates in the bitmap/image space.
    bitmap_resolution : torch.Tensor
        The resolution of the bitmap (e.g., width and height).
    target_area_indices : torch.Tensor
        Indices indicating which target area each bitmap coordinate maps to.
    expected_coordinates : torch.Tensor
        The expected 3D coordinates corresponding to the input bitmap coordinates.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    mock_solar_tower = MagicMock(spec=SolarTower)
    mock_target_areas_planar = MagicMock(spec=TowerTargetAreasPlanar)
    mock_target_areas_planar.names = ["planar1", "planar2"]
    mock_target_areas_planar.dimensions = torch.tensor(
        [[6.0, 6.0], [2.0, 4.0]], device=device
    )
    mock_target_areas_planar.centers = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 2.0, 1.0]], device=device
    )
    mock_target_areas_cylindrical = MagicMock(spec=TowerTargetAreasCylindrical)
    mock_target_areas_cylindrical.names = ["cylinder1"]
    mock_target_areas_cylindrical.normals = torch.tensor(
        ([[0.0, 1.0, 0.0, 0.0]]), device=device
    )
    mock_target_areas_cylindrical.axes = torch.tensor(
        ([[0.0, 0.0, 1.0, 0.0]]), device=device
    )
    mock_target_areas_cylindrical.radii = torch.tensor(([[2.0]]), device=device)
    mock_target_areas_cylindrical.heights = torch.tensor(([[6.0]]), device=device)
    mock_target_areas_cylindrical.opening_angles = torch.tensor(
        ([[math.pi]]), device=device
    )
    mock_target_areas_cylindrical.centers = torch.tensor(
        ([[0.0, 0.0, 0.0, 1.0]]), device=device
    )

    mock_solar_tower.target_areas = [
        mock_target_areas_planar,
        mock_target_areas_cylindrical,
    ]
    mock_solar_tower.number_of_target_area_types = 2
    mock_solar_tower.number_of_target_areas_per_type = torch.tensor(
        [2, 1], device=device
    )
    mock_solar_tower.target_name_to_index = {"planar1": 0, "planar2": 1, "cylinder1": 2}
    mock_solar_tower.index_to_target_area = {0: "planar1", 1: "planar2", 2: "cylinder1"}

    target_coordinates = coordinates.bitmap_coordinates_to_target_coordinates(
        bitmap_coordinates=bitmap_coordinates.to(device),
        bitmap_resolution=bitmap_resolution.to(device),
        solar_tower=mock_solar_tower,
        target_area_indices=target_area_indices.to(device),
        device=device,
    )

    torch.testing.assert_close(
        target_coordinates, expected_coordinates.to(device), rtol=1e-4, atol=1e-4
    )
