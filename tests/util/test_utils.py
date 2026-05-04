import math
from unittest.mock import MagicMock

import pytest
import torch

from artist.field.solar_tower import SolarTower
from artist.field.tower_target_areas_cylindrical import TowerTargetAreasCylindrical
from artist.field.tower_target_areas_planar import TowerTargetAreasPlanar
from artist.util import utils


@pytest.mark.parametrize(
    "total_width, slope_width, plateau_width, expected",
    [
        (8, 2, 4, torch.tensor([0.25, 0.75, 1.0, 1.0, 1.0, 1.0, 0.75, 0.25])),
        (4, 2, 4, torch.tensor([1.0, 1.0, 1.0, 1.0])),
        (1, 2, 3, torch.tensor([1.0])),
        (
            10,
            2,
            2,
            torch.tensor([0.0, 0.0, 0.25, 0.75, 1.0, 1.0, 0.75, 0.25, 0.0, 0.0]),
        ),
    ],
)
def test_trapezoid_distribution(
    total_width: int,
    slope_width: int,
    plateau_width: int,
    expected: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test that the trapezoid distribution works as expected.

    Parameters
    ----------
    total_width : int
        The total width of the trapezoid.
    slope_width : int
        The width of the slope of the trapezoid.
    plateau_width : int
        The width of the plateau.
    expected : torch.Tensor
        The expected distribution.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    trapezoid = utils.trapezoid_distribution(
        total_width=total_width,
        slope_width=slope_width,
        plateau_width=plateau_width,
        device=device,
    )

    torch.testing.assert_close(trapezoid, expected.to(device), atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize(
    "image, crop_width, crop_height, target_area_indices, expected_cropped",
    [
        # Symmetric bitmaps and no change in dimensions.
        (
            torch.tensor(
                [
                    [[1.0, 2.0, 1.0], [2.0, 3.0, 2.0], [1.0, 2.0, 1.0]],
                    [[0.5, 0.0, 0.5], [0.5, 1.0, 0.5], [0.5, 0.0, 0.5]],
                ]
            ),
            3.0,
            3.0,
            torch.tensor([0, 1]),
            torch.tensor(
                [
                    [[1.0, 2.0, 1.0], [2.0, 3.0, 2.0], [1.0, 2.0, 1.0]],
                    [[0.5, 0.0, 0.5], [0.5, 1.0, 0.5], [0.5, 0.0, 0.5]],
                ]
            ),
        ),
        # Symmetric bitmaps and change in dimensions.
        (
            torch.tensor(
                [
                    [
                        [1.0, 2.0, 2.0, 1.0],
                        [2.0, 3.0, 3.0, 2.0],
                        [2.0, 3.0, 3.0, 2.0],
                        [1.0, 2.0, 2.0, 1.0],
                    ],
                    [
                        [1.0, 2.0, 2.0, 1.0],
                        [2.0, 3.0, 3.0, 2.0],
                        [2.0, 3.0, 3.0, 2.0],
                        [1.0, 2.0, 2.0, 1.0],
                    ],
                ]
            ),
            5.0,
            5.0,
            torch.tensor([0, 1]),
            torch.tensor(
                [
                    [
                        [0.0000, 0.0000, 0.0000, 0.0000],
                        [0.0000, 2.3333, 2.3333, 0.0000],
                        [0.0000, 2.3333, 2.3333, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000],
                    ],
                    [
                        [0.0000, 0.0000, 0.0000, 0.0000],
                        [0.0000, 2.3333, 2.3333, 0.0000],
                        [0.0000, 2.3333, 2.3333, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000],
                    ],
                ]
            ),
        ),
        # Asymmetric bitmaps and no change in dimensions.
        (
            torch.tensor(
                [
                    [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                ]
            ),
            3.0,
            3.0,
            torch.tensor([0, 1]),
            torch.tensor(
                [
                    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                ]
            ),
        ),
        # Asymmetric bitmaps and change in dimensions.
        (
            torch.tensor(
                [
                    [
                        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, 2.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ),
            2.0,
            2.0,
            torch.tensor([0]),
            torch.tensor(
                [
                    [
                        [0.1111, 0.3333, 0.3333, 0.3333, 0.3333, 0.1111],
                        [0.3333, 1.0000, 1.0000, 1.0000, 1.0000, 0.3333],
                        [0.3333, 1.0000, 1.4444, 1.4444, 1.0000, 0.3333],
                        [0.3333, 1.0000, 1.4444, 1.4444, 1.0000, 0.3333],
                        [0.3333, 1.0000, 1.0000, 1.0000, 1.0000, 0.3333],
                        [0.1111, 0.3333, 0.3333, 0.3333, 0.3333, 0.1111],
                    ]
                ]
            ),
        ),
    ],
)
def test_crop_flux_distributions_around_center_centering(
    image: torch.Tensor,
    crop_width: float,
    crop_height: float,
    target_area_indices: torch.Tensor,
    expected_cropped: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test that cropping is identity when the center of mass is at the geometric center.

    When the center of mass is located at the geometric center of the image
    and the crop dimensions span the full target plane, the cropping operation
    should return the image unchanged.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor to be cropped.
        Tensor of shape [number_of_bitmaps, bitmap_resolution_e, bitmap_resolution_u].
    crop_width : float
        Desired crop width in meters.
    crop_height : float
        Desired crop height in meters.
    target_area_indices : torch.Tensor
        Indices of the target areas for each active heliostat.
    expected_cropped : torch.Tensor
        The expected output image tensor after cropping.
        Tensor of shape [number_of_bitmaps, bitmap_resolution_e, bitmap_resolution_u].
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    mock_solar_tower = MagicMock(spec=SolarTower)
    mock_target_areas_planar = MagicMock(spec=TowerTargetAreasPlanar)
    mock_target_areas_planar.names = ["multi_focus_tower"]
    mock_target_areas_planar.dimensions = torch.tensor([[3.0, 3.0]], device=device)
    mock_target_areas_cylindrical = MagicMock(spec=TowerTargetAreasCylindrical)
    mock_target_areas_cylindrical.names = ["receiver"]
    mock_target_areas_cylindrical.radii = torch.tensor(([[1.0]]), device=device)
    mock_target_areas_cylindrical.heights = torch.tensor(([[3.0]]), device=device)
    mock_target_areas_cylindrical.opening_angles = torch.tensor(
        ([[3.0]]), device=device
    )

    mock_solar_tower.target_areas = [
        mock_target_areas_planar,
        mock_target_areas_cylindrical,
    ]
    mock_solar_tower.number_of_target_area_types = 2
    mock_solar_tower.number_of_target_areas_per_type = torch.tensor(
        [1, 1], device=device
    )
    mock_solar_tower.target_name_to_index = {"multi_focus_tower": 0, "receiver": 1}
    mock_solar_tower.index_to_target_area = {0: "multi_focus_tower", 1: "receiver"}

    cropped = utils.crop_flux_distributions_around_center(
        flux_distributions=image.to(device),
        solar_tower=mock_solar_tower,
        target_area_indices=target_area_indices.to(device),
        crop_width=crop_width,
        crop_height=crop_height,
        device=device,
    )
    torch.testing.assert_close(
        cropped, expected_cropped.to(device), rtol=1e-4, atol=1e-4
    )
    assert not torch.isnan(cropped).any()


@pytest.mark.parametrize(
    "wgs84_coordinates, reference_point, expected_enu_coordinates",
    [
        # Coordinates of Juelich power plant and multi-focus tower.
        (
            (
                torch.tensor(
                    [[50.91339645088695, 6.387574436728054, 138.97975]],
                    dtype=torch.float64,
                ),
                torch.tensor(
                    [50.913421630859, 6.387824755874856, 87.000000000000],
                    dtype=torch.float64,
                ),
                torch.tensor([[-17.6045, -2.8012, 51.9798]]),
            )
        ),
    ],
)
def test_wgs84_to_enu_converter(
    wgs84_coordinates: torch.Tensor,
    reference_point: torch.Tensor,
    expected_enu_coordinates: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test the WGS84 to ENU conversion.

    Parameters
    ----------
    wgs84_coordinates : torch.Tensor
        The coordinates in latitude, longitude, altitude that are to be transformed.
    reference_point : torch.Tensor
        The center of origin of the ENU coordinate system in WGS84 coordinates.
    expected_enu_coordinates : torch.Tensor
        The expected enu coordinates.
    device : torch.device| str
        The device on which to initialize tensors (default is cuda).

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    calculated_enu_coordinates = utils.convert_wgs84_coordinates_to_local_enu(
        wgs84_coordinates.to(device), reference_point.to(device), device
    )

    torch.testing.assert_close(
        calculated_enu_coordinates, expected_enu_coordinates.to(device)
    )


inv_sqrt2 = 1.0 / math.sqrt(2)


@pytest.mark.parametrize(
    "azimuth, elevation, degree, expected",
    [
        (
            torch.tensor([-45.0, -45.0, 45.0, 135.0, 225.0, 315.0]),
            torch.tensor([0.0, 45.0, 45.0, 45.0, 45.0, 45.0]),
            True,
            torch.tensor(
                [
                    [
                        -inv_sqrt2,
                        -inv_sqrt2,
                        0.0,
                    ],
                    [
                        -0.5,
                        -0.5,
                        inv_sqrt2,
                    ],
                    [0.5, -0.5, inv_sqrt2],
                    [0.5, 0.5, inv_sqrt2],
                    [-0.5, 0.5, inv_sqrt2],
                    [-0.5, -0.5, inv_sqrt2],
                ]
            ),
        ),
        (
            torch.tensor([-torch.pi / 4, torch.pi / 4]),
            torch.tensor([torch.pi / 4, torch.pi / 4]),
            False,
            torch.tensor(
                [
                    [-0.5, -0.5, inv_sqrt2],
                    [0.5, -0.5, inv_sqrt2],
                ]
            ),
        ),
    ],
)
def test_azimuth_elevation_to_enu(
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    degree: bool,
    expected: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test the azimuth, elevation to east, north, up converter.

    Parameters
    ----------
    azimuth : torch.Tensor
        The azimuth angle.
    elevation : torch.Tensor
        The elevation angle.
    degree : bool
        Angles in degree.
    expected : torch.Tensor
        The expected coordinates in the ENU (east, north, up) coordinate system.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    enu_coordinates = utils.azimuth_elevation_to_enu(
        azimuth=azimuth, elevation=elevation, degree=degree, device=device
    )
    torch.testing.assert_close(
        enu_coordinates, expected.to(device), rtol=1e-4, atol=1e-4
    )
