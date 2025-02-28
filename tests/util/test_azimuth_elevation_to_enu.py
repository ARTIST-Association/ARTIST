import pytest
import torch

from artist.util import utils


@pytest.mark.parametrize(
    "azimuth, elevation, degree, expected",
    [
        (
            torch.tensor(-45.0),
            torch.tensor(0.0),
            True,
            torch.tensor([-1/torch.sqrt(torch.tensor([2.0])), -1/torch.sqrt(torch.tensor([2.0])), 0.0])
        ),
        (
            torch.tensor(-45.0),
            torch.tensor(45.0),
            True,
            torch.tensor([-0.5, -0.5, 1/torch.sqrt(torch.tensor([2.0]))])
        ),
        (
            torch.tensor(45.0),
            torch.tensor(45.0),
            True,
            torch.tensor([0.5, -0.5, 1/torch.sqrt(torch.tensor([2.0]))])
        ),
        (
            torch.tensor(135.0),
            torch.tensor(45.0),
            True,
            torch.tensor([0.5, 0.5, 1/torch.sqrt(torch.tensor([2.0]))])
        ),
        (
            torch.tensor(225.0),
            torch.tensor(45.0),
            True,
            torch.tensor([-0.5, 0.5, 1/torch.sqrt(torch.tensor([2.0]))])

        ),
        (
            torch.tensor(315.0),
            torch.tensor(45.0),
            True,
            torch.tensor([-0.5, -0.5, 1/torch.sqrt(torch.tensor([2.0]))])
        ),
        (
            torch.tensor(-torch.pi/4),
            torch.tensor(torch.pi/4),
            False,
            torch.tensor([-0.5, -0.5, 1/torch.sqrt(torch.tensor([2.0]))])
        ),
        (
            torch.tensor(torch.pi/4),
            torch.tensor(torch.pi/4),
            False,
            torch.tensor([0.5, -0.5, 1/torch.sqrt(torch.tensor([2.0]))])
        ),
    ],
)
def test_azimuth_elevation_to_enu(
    azimuth: torch.Tensor, elevation:torch.Tensor, degree: bool, device: torch.device, expected: torch.Tensor, 
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
    device : torch.device
        The device on which to initialize tensors.
    expected : torch.Tensor
        The expected coordinates in the ENU (east, north, up) coordinate system.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    enu_coordinates = utils.azimuth_elevation_to_enu(
        azimuth=azimuth,
        elevation=elevation,
        degree=degree,
        device=device
    )
    torch.testing.assert_close(enu_coordinates, expected.to(device), rtol=1e-4, atol=1e-4)
