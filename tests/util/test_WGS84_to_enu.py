from artist.util import utils
import pytest
import torch


@pytest.fixture(params=["cpu", "cuda:3"] if torch.cuda.is_available() else ["cpu"])
def device(request: pytest.FixtureRequest) -> torch.device:
    """
    Return the device on which to initialize tensors.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The pytest fixture used to consider different test cases.

    Returns
    -------
    torch.device
        The device on which to initialize tensors.
    """
    return torch.device(request.param)

@pytest.mark.parametrize(
    "WGS84_coordinates, reference_point, expected_enu_coordinates",
    [
        # Coordinates of Jülich power plant and multifocustower
        ((torch.tensor([50.91339645088695, 6.387574436728054, 138.97975], dtype=torch.float64), 
          torch.tensor([50.913421630859, 6.387824755874856, 87.000000000000], dtype=torch.float64), 
          torch.tensor([-17.6045,  -2.8012,  51.9798]))),
    ]
)
def test_WGS84_to_enu_converter(
    WGS84_coordinates: torch.Tensor,
    reference_point: torch.Tensor,
    expected_enu_coordinates: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test the WGS84 to enu conversion.

    Parameters
    ----------
    WGS84_coordinates : torch.Tensor
        The coordinates in lat, lon, alt that are to be transformed.
    reference_point : torch.Tensor
        The center of origin of the ENU coordinate system in WGS84 coordinates.
    expected_enu_coordinates : torch.Tensor
        The expected enu coordinates.
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).
    
    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    calculated_enu_coordinates = utils.convert_WGS84_coordinates_to_local_enu(WGS84_coordinates.to(device), reference_point.to(device), device)

    torch.testing.assert_close(calculated_enu_coordinates, expected_enu_coordinates.to(device))

