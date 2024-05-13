import pytest
import torch

from artist.raytracing import raytracing_utils


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
    """
    reflection = raytracing_utils.reflect(
        incoming_ray_direction=incoming_ray_direction,
        reflection_surface_normals=surface_normals,
    )

    torch.testing.assert_close(reflection, expected_reflection, rtol=1e-4, atol=1e-4)
