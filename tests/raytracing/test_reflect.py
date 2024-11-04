import pytest
import torch

from artist.raytracing import raytracing_utils


@pytest.fixture(params=["cpu", "cuda:3"] if torch.cuda.is_available() else ["cpu"])
def device(request: pytest.FixtureRequest) -> torch.device:
    """Return the device on which to initialize tensors."""
    return torch.device(request.param)


@pytest.mark.parametrize(
    "incoming_ray_direction, surface_normals, expected_reflection",
    [
        (
            torch.tensor([1.0, 1.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
            torch.tensor([1.0, 1.0, -1.0, 0.0]),
        ),
        (
            torch.tensor([1.0, 1.0, 1.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0, 0.0]),
            torch.tensor([1.0, -1.0, 1.0, 0.0]),
        ),
        (
            torch.tensor([1.0, 1.0, 1.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([-1.0, 1.0, 1.0, 0.0]),
        ),
        (
            torch.tensor([2.0, 1.0, 3.0, 0.0]),
            torch.tensor([0.3, 0.6, 0.7, 0.0]),
            torch.tensor([0.0200, -2.9600, -1.6200, 0.0000]),
        ),
        (
            torch.tensor([1.0, 1.0, 1.0, 0.0]),
            torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
            ),
            torch.tensor(
                [[-1.0, 1.0, 1.0, 0.0], [1.0, -1.0, 1.0, 0.0], [1.0, 1.0, -1.0, 0.0]]
            ),
        ),
    ],
)
def test_reflect_function(
    incoming_ray_direction: torch.Tensor,
    surface_normals: torch.Tensor,
    expected_reflection: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test the reflection function by reflection various rays from different surfaces.

    Parameters
    ----------
    incoming_ray_direction : torch.Tensor
        The direction of the incoming ray to be reflected.
    surface_normals : torch.Tensor
        The surface normals of the reflective surface.
    expected_reflection : torch.Tensor
        The expected direction of the reflected rays.
    device : torch.device
        The device on which to initialize tensors.
    """
    reflection = raytracing_utils.reflect(
        incoming_ray_direction=incoming_ray_direction.to(device),
        reflection_surface_normals=surface_normals.to(device),
    )

    torch.testing.assert_close(
        reflection, expected_reflection.to(device), rtol=1e-4, atol=1e-4
    )
