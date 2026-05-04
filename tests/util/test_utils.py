import math

import pytest
import torch

from artist.util import utils


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
