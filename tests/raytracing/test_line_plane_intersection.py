from typing import Union

import pytest
import torch

from artist.raytracing import raytracing_utils
from artist.raytracing.rays import Rays


@pytest.fixture
def rays(request: pytest.FixtureRequest, device: torch.device) -> Rays:
    """
    Define rays with directions and magnitudes used in tests.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The pytest fixture used to consider different test cases.
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    Rays
        The rays.
    """
    directions, magnitudes = request.param
    return Rays(
        ray_directions=directions.to(device), ray_magnitudes=magnitudes.to(device)
    )


@pytest.mark.parametrize(
    (
        "rays",
        "plane_normal_vectors",
        "plane_center",
        "points_at_ray_origin",
        "expected_intersections",
        "expected_absolute_intensities",
    ),
    [
        (  # Single intersection with ray perpendicular to plane.
            (torch.tensor([[[[0.0, 0.0, -1.0, 0.0]]]]), torch.tensor([[[1.0]]])),
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
            torch.tensor([[[0.0, 0.0, 1.0, 1.0]]]),
            torch.tensor([[[[0.0, 0.0, 0.0, 1.0]]]]),
            torch.tensor([[[1.0]]]),
        ),
        (  # Single intersection not perpendicular to plane.
            (torch.tensor([[[[1.0, 1.0, -1.0, 0.0]]]]), torch.tensor([[[1.0]]])),
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
            torch.tensor([[[0.0, 0.0, 1.0, 1.0]]]),
            torch.tensor([[[[1.0, 1.0, 0.0, 1.0]]]]),
            torch.tensor([[[1.0]]]),
        ),
        (  # Single intersection with tilted plane.
            (torch.tensor([[[[-1.0, -2.0, -1.0, 0.0]]]]), torch.tensor([[[1.0]]])),
            torch.tensor([0.5, 2.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
            torch.tensor([[[2.0, 2.0, 2.0, 1.0]]]),
            torch.tensor([[[[0.7273, -0.5455, 0.7273, 1.0]]]]),
            torch.tensor([[[0.458333343267]]]),
        ),
        (  # Multiple intersections with multiple rays.
            (
                torch.tensor(
                    [
                        [
                            [
                                [0.0, 0.0, -1.0, 0.0],
                                [1.0, 1.0, -1.0, 0.0],
                                [-1.0, -2.0, -1.0, 0.0],
                            ]
                        ]
                    ]
                ),
                torch.tensor([[[1.0, 1.0, 1.0]]]),
            ),
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
            torch.tensor(
                [[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 2.0, 1.0]]]
            ),
            torch.tensor(
                [[[[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], [0.0, -2.0, 0.0, 1.0]]]]
            ),
            torch.tensor([[[1.0000, 1.0000, 0.0833]]]),
        ),
        (  # ValueError - no intersection since ray is parallel to plane.
            (torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]]), torch.tensor([[[1.0]]])),
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
            torch.tensor([[[0.0, 0.0, 1.0, 1.0]]]),
            None,
            None,
        ),
        (  # ValueError - no intersection since ray is within the plane.
            (torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]]), torch.tensor([[[1.0]]])),
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
            torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]),
            None,
            None,
        ),
    ],
    indirect=["rays"],
)
def test_line_plane_intersection(
    rays: Rays,
    plane_normal_vectors: torch.Tensor,
    plane_center: torch.Tensor,
    points_at_ray_origin: torch.Tensor,
    expected_intersections: Union[torch.Tensor, None],
    expected_absolute_intensities: Union[torch.Tensor, None],
    device: torch.device,
) -> None:
    """
    Test the line plane intersection function by computing the intersections between various rays and planes.

    Parameters
    ----------
    rays : Rays
        The rays with directions and magnitudes.
    plane_normal_vectors : torch.Tensor
        The normal vectors of the plane being considered for the intersection.
    plane_center : torch.Tensor
        The center of the plane being considered for the intersection.
    points_at_ray_origin : torch.Tensor
        The surface points of the ray origin.
    expected_intersections : Union[torch.Tensor, None]
        The expected intersections between the rays and the plane, or ``None`` if no intersections are expected.
    expected_absolute_intensities : Union[torch.Tensor, None]
        The expected absolute intensities of the ray intersections, or ``None`` if no intersections are expected.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    # Check if the ValueError is thrown as expected.
    if expected_intersections is None or expected_absolute_intensities is None:
        with pytest.raises(ValueError) as exc_info:
            raytracing_utils.line_plane_intersections(
                rays=rays,
                plane_normal_vector=plane_normal_vectors.to(device),
                plane_center=plane_center.to(device),
                points_at_ray_origin=points_at_ray_origin.to(device),
            )
        assert "No ray intersection on the front of the target area plane." in str(
            exc_info.value
        )
    else:
        # Check if the intersections match the expected intersections.
        intersections, absolute_intensities = raytracing_utils.line_plane_intersections(
            rays=rays,
            plane_normal_vector=plane_normal_vectors.to(device),
            plane_center=plane_center.to(device),
            points_at_ray_origin=points_at_ray_origin.to(device),
        )
        torch.testing.assert_close(
            intersections, expected_intersections.to(device), rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            absolute_intensities,
            expected_absolute_intensities.to(device),
            rtol=1e-4,
            atol=1e-4,
        )
