import pytest
import torch

from artist.field.tower_target_areas import TowerTargetAreas
from artist.scene.rays import Rays
from artist.util import raytracing_utils


@pytest.mark.parametrize(
    "incident_ray_directions, surface_normals, expected_reflection",
    [
        (
            torch.tensor(
                [
                    [1.0, 1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0],
                    [2.0, 1.0, 3.0, 0.0],
                ]
            ),
            torch.tensor(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.3, 0.6, 0.7, 0.0],
                ]
            ),
            torch.tensor(
                [
                    [1.0, 1.0, -1.0, 0.0],
                    [1.0, -1.0, 1.0, 0.0],
                    [-1.0, 1.0, 1.0, 0.0],
                    [0.0200, -2.9600, -1.6200, 0.0000],
                ]
            ),
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
    incident_ray_directions: torch.Tensor,
    surface_normals: torch.Tensor,
    expected_reflection: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test the reflection function by reflecting various rays from different surfaces.

    Parameters
    ----------
    incident_ray_directions : torch.Tensor
        The direction of the incoming ray to be reflected.
    surface_normals : torch.Tensor
        The surface normals of the reflective surface.
    expected_reflection : torch.Tensor
        The expected direction of the reflected rays.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    reflection = raytracing_utils.reflect(
        incident_ray_directions=incident_ray_directions.to(device),
        reflection_surface_normals=surface_normals.to(device),
    )

    torch.testing.assert_close(
        reflection, expected_reflection.to(device), rtol=1e-4, atol=1e-4
    )


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


@pytest.fixture
def target_area_1(
    device: torch.device,
) -> TowerTargetAreas:
    """
    Create target areas to use in the test.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    TowerTargetAreas
        The target areas.
    """
    normal_vectors = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=device)
    centers = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
    dimensions = torch.tensor([[1, 1]], device=device)
    curvatures = torch.tensor([[0, 0]], device=device)

    return TowerTargetAreas(
        names=["first"],
        geometries=["planar"],
        centers=centers,
        normal_vectors=normal_vectors,
        dimensions=dimensions,
        curvatures=curvatures,
    )


@pytest.fixture
def target_area_2(
    device: torch.device,
) -> TowerTargetAreas:
    """
    Create target areas to use in the test.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    TowerTargetAreas
        The target areas.
    """
    normal_vectors = torch.tensor([[0.5, 2.0, 1.0, 0.0]], device=device)
    centers = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
    dimensions = torch.tensor([[1, 1]], device=device)
    curvatures = torch.tensor([[0, 0]], device=device)

    return TowerTargetAreas(
        names=["first"],
        geometries=["planar_tilted"],
        centers=centers,
        normal_vectors=normal_vectors,
        dimensions=dimensions,
        curvatures=curvatures,
    )


@pytest.fixture(params=[torch.tensor([0]), None])
def target_area_mask(
    request: pytest.FixtureRequest, device: torch.device
) -> torch.Tensor | None:
    """
    Create a target area mask or None to use in the test.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The pytest fixture used to consider different test cases.
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    torch.Tensor | None
        The target area mask.
    """
    mask = request.param

    if mask is not None:
        mask = mask.to(device)
    return mask


@pytest.mark.parametrize(
    (
        "rays",
        "target_areas_fixture",
        "points_at_ray_origins",
        "expected_intersections",
        "expected_absolute_intensities",
    ),
    [
        (  # Single intersection with ray perpendicular to plane.
            (torch.tensor([[[[0.0, 0.0, -1.0, 0.0]]]]), torch.tensor([[[1.0]]])),
            "target_area_1",
            torch.tensor([[[0.0, 0.0, 1.0, 1.0]]]),
            torch.tensor([[[[0.0, 0.0, 0.0, 1.0]]]]),
            torch.tensor([[[1.0]]]),
        ),
        (  # Single intersection not perpendicular to plane.
            (torch.tensor([[[[1.0, 1.0, -1.0, 0.0]]]]), torch.tensor([[[1.0]]])),
            "target_area_1",
            torch.tensor([[[0.0, 0.0, 1.0, 1.0]]]),
            torch.tensor([[[[1.0, 1.0, 0.0, 1.0]]]]),
            torch.tensor([[[1.0]]]),
        ),
        (  # Single intersection with tilted plane.
            (torch.tensor([[[[-1.0, -2.0, -1.0, 0.0]]]]), torch.tensor([[[1.0]]])),
            "target_area_2",
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
            "target_area_1",
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
            "target_area_1",
            torch.tensor([[[0.0, 0.0, 1.0, 1.0]]]),
            None,
            None,
        ),
        (  # ValueError - no intersection since ray is within the plane.
            (torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]]), torch.tensor([[[1.0]]])),
            "target_area_1",
            torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]),
            None,
            None,
        ),
    ],
    indirect=["rays"],
)
def test_line_plane_intersection(
    request: pytest.FixtureRequest,
    target_area_mask: torch.Tensor | None,
    rays: Rays,
    target_areas_fixture: str,
    points_at_ray_origins: torch.Tensor,
    expected_intersections: torch.Tensor | None,
    expected_absolute_intensities: torch.Tensor | None,
    device: torch.device,
) -> None:
    """
    Test the line plane intersection function by computing the intersections between various rays and planes.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The pytest fixture used to consider different test cases.
    target_area_mask : torch.Tensor | None
        The target area mask.
    rays : Rays
        The rays with directions and magnitudes.
    plane_normal_vectors : torch.Tensor
        The normal vectors of the plane being considered for the intersection.
    plane_center : torch.Tensor
        The center of the plane being considered for the intersection.
    points_at_ray_origin : torch.Tensor
        The surface points of the ray origin.
    expected_intersections : torch.Tensor | None
        The expected intersections between the rays and the plane, or ``None`` if no intersections are expected.
    expected_absolute_intensities : torch.Tensor | None
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
                points_at_ray_origins=points_at_ray_origins.to(device),
                target_areas=request.getfixturevalue(target_areas_fixture),
                target_area_mask=target_area_mask,
                device=device,
            )
        assert "No ray intersections on the front of the target area planes." in str(
            exc_info.value
        )
    else:
        # Check if the intersections match the expected intersections.
        intersections, absolute_intensities = raytracing_utils.line_plane_intersections(
            rays=rays,
            points_at_ray_origins=points_at_ray_origins.to(device),
            target_areas=request.getfixturevalue(target_areas_fixture),
            target_area_mask=target_area_mask,
            device=device,
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
