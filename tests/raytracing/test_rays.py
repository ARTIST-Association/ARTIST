import pytest
import torch

from artist.raytracing.rays import Rays


@pytest.mark.parametrize(
    "ray_directions, ray_magnitudes",
    [
        (torch.rand(10, 4, 1000, 4), torch.rand(9, 4, 1000)),
        (torch.rand(10, 3, 1000, 4), torch.rand(10, 4, 1000)),
        (torch.rand(10, 4, 100, 4), torch.rand(10, 4, 1000)),
    ],
)
def test_ray_initialization_error(
    ray_directions: torch.Tensor, ray_magnitudes: torch.Tensor
) -> None:
    """
    Test the ray initialization.

    Parameters
    ----------
    ray_directions : torch.Tensor
        The direction of the rays.
    ray_magnitudes : torch.Tensor
        The magnitudes of the rays.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    # Check if the ValueError is thrown as expected.
    with pytest.raises(ValueError) as exc_info:
        Rays(ray_directions=ray_directions, ray_magnitudes=ray_magnitudes)
    assert "Ray directions and magnitudes have differing sizes!" in str(exc_info.value)
