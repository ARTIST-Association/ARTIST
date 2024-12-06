from typing import Optional

import pytest
import torch

from artist.raytracing import raytracing_utils

@pytest.mark.parametrize(
    "ray_directions, plane_normal_vectors, plane_center, points_at_ray_origin, expected_intersections",
    [
        (  # Single intersection with ray perpendicular to plane.
            torch.tensor([0.0, 0.0, -1.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
        ),
        (  # Single intersection not perpendicular to plane.
            torch.tensor([1.0, 1.0, -1.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
            torch.tensor([1.0, 1.0, 0.0, 1.0]),
        ),
        (  # Single intersection with tilted plane.
            torch.tensor([-1.0, -2.0, -1.0, 0.0]),
            torch.tensor([0.5, 2.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
            torch.tensor([2.0, 2.0, 2.0, 1.0]),
            torch.tensor([0.7273, -0.5455, 0.7273, 1.0]),
        ),
        (  # Multiple intersections with multiple rays.
            torch.tensor(
                [[0.0, 0.0, -1.0, 0.0], [1.0, 1.0, -1.0, 0.0], [-1.0, -2.0, -1.0, 0.0]]
            ),
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
            torch.tensor(
                [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 2.0, 1.0]]
            ),
            torch.tensor(
                [[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], [0.0, -2.0, 0.0, 1.0]]
            ),
        ),
        (  # AssertionError - no intersection since ray is parallel to plane.
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
            None,
        ),
        (  # AssertionError - no intersection since ray is within the plane.
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
            None,
        ),
    ],
)
def test_line_plane_intersection(
    ray_directions: torch.Tensor,
    plane_normal_vectors: torch.Tensor,
    plane_center: torch.Tensor,
    points_at_ray_origin: torch.Tensor,
    expected_intersections: Optional[torch.Tensor],
    device: torch.device,
) -> None:
    """
    Test the line plane intersection function by computing the intersections between various rays and planes.

    Parameters
    ----------
    ray_directions : torch.Tensor
        The direction of the rays being considered for the intersection.
    plane_normal_vectors : torch.Tensor
        The normal vectors of the plane being considered for the intersection.
    plane_center : torch.Tensor
        The center of the plane being considered for the intersection.
    points_at_ray_origin : torch.Tensor
        The surface points of the ray origin.
    expected_intersections : torch.Tensor, None
        The expected intersections between the rays and the plane, or ``None`` if no intersections are expected.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    # Check if the ValueError is thrown as expected.
    if expected_intersections is None:
        with pytest.raises(ValueError) as exc_info:
            raytracing_utils.line_plane_intersections(
                ray_directions=ray_directions.to(device),
                plane_normal_vectors=plane_normal_vectors.to(device),
                plane_center=plane_center.to(device),
                points_at_ray_origin=points_at_ray_origin.to(device),
            )
        assert "No intersection or line is within plane." in str(exc_info.value)
    else:
        # Check if the intersections match the expected intersections.
        intersections = raytracing_utils.line_plane_intersections(
            ray_directions=ray_directions.to(device),
            plane_normal_vectors=plane_normal_vectors.to(device),
            plane_center=plane_center.to(device),
            points_at_ray_origin=points_at_ray_origin.to(device),
        )
        torch.testing.assert_close(
            intersections, expected_intersections.to(device), rtol=1e-4, atol=1e-4
        )
