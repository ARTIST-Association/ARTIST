import pytest
import torch

from artist.field.tower_target_areas_cylindrical import TowerTargetAreasCylindrical
from artist.field.tower_target_areas_planar import TowerTargetAreas, TowerTargetAreasPlanar
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
def target_area_1_planar(
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
    TowerTargetAreasPlanar
        The target areas.
    """
    normals = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device)
    centers = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
    dimensions = torch.tensor([[2, 2]], device=device)

    return TowerTargetAreasPlanar(
        names=["planar1"],
        centers=centers,
        normals=normals,
        dimensions=dimensions,
    )


@pytest.fixture
def target_area_2_planar(
    device: torch.device,
) -> TowerTargetAreasPlanar:
    """
    Create target areas to use in the test.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    TowerTargetAreasPlanar
        The target areas.
    """
    normals = torch.tensor([[0.2182, 0.7071, 0.7071, 0.0]], device=device)
    centers = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
    dimensions = torch.tensor([[3, 3]], device=device)

    return TowerTargetAreasPlanar(
        names=["planar2"],
        centers=centers,
        normals=normals,
        dimensions=dimensions,
    )


@pytest.fixture(params=[torch.tensor([0]), None])
def target_area_indices(
    request: pytest.FixtureRequest, device: torch.device
) -> torch.Tensor | None:
    """
    Create target area indices or None to use in the test.

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
        "expected_intersections_e",
        "expected_intersections_u",
        "expected_intersection_distances",
        "expected_absolute_intensities",
    ),
    [
        (  # Single intersection with ray perpendicular to plane.
            (torch.tensor([[[[0.0, -1.0, 0.0, 0.0]]]]), torch.tensor([[[1.0]]])),
            "target_area_1_planar",
            torch.tensor([[[0.0, 1.0, 0.0, 1.0]]]),
            torch.tensor([[[127.5]]]),
            torch.tensor([[[127.5]]]),
            torch.tensor([[[1.0]]]),
            torch.tensor([[[1.0]]]),
        ),
        (  # Single intersection not perpendicular to plane.
            (torch.tensor([[[[ 0.5774,  -0.5774, 0.5774, 0.0]]]]), torch.tensor([[[1.0]]])),
            "target_area_1_planar",
            torch.tensor([[[0.0, 1.0, 0.0, 1.0]]]),
            torch.tensor([[[0.0]]]),
            torch.tensor([[[255.0]]]),
            torch.tensor([[[1.7319]]]),
            torch.tensor([[[0.5774]]]),
        ),
        (  # Single intersection with tilted plane and reduced magnitude.
            (torch.tensor([[[[0.0, -1.0, 0.0, 0.0]]]]), torch.tensor([[[0.5]]])),
            "target_area_2_planar",
            torch.tensor([[[0.0, 1.0, 0.0, 1.0]]]),
            torch.tensor([[[127.5]]]),
            torch.tensor([[[127.5]]]),
            torch.tensor([[[1.0]]]),
            torch.tensor([[[0.3535]]]),
        ),
        (  # Multiple intersections with multiple rays and some rays do not hit the plane.
            (
                torch.tensor(
                    [
                        [
                            [
                                [0.0, 0.0, -1.0, 0.0],
                                [0.4472, -0.8944, 0.0, 0.0],
                                [0.7071, -0.7071, 0.0, 0.0],
                            ]
                        ]
                    ]
                ),
                torch.tensor([[[1.0, 2.0, 1.0]]]),
            ),
            "target_area_1_planar",
            torch.tensor(
                [[[1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]]
            ),
            torch.tensor([[[255.0, 63.75, 0.0]]]),
            torch.tensor([[[0.0, 127.5, 127.5]]]),
            torch.tensor([[[0.0, 1.1181, 1.4142]]]),
            torch.tensor([[[-0.0, 1.7888, 0.7071]]]),
        ),
    ],
    indirect=["rays"],
)
def test_line_plane_intersection(
    request: pytest.FixtureRequest,
    target_area_indices: torch.Tensor | None,
    rays: Rays,
    target_areas_fixture: str,
    points_at_ray_origins: torch.Tensor,
    expected_intersections_e: torch.Tensor,
    expected_intersections_u: torch.Tensor,
    expected_intersection_distances: torch.Tensor,
    expected_absolute_intensities: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test the line plane intersection function by computing the intersections between various rays and planes.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The pytest fixture used to consider different test cases.
    target_area_indices : torch.Tensor | None
        The target area indices.
    rays : Rays
        The rays with directions and magnitudes.
    target_areas_fixture : str
        Name of fixture to get target areas.
    points_at_ray_origins : torch.Tensor
        The surface points of the ray origin.
    expected_intersections : torch.Tensor
        The expected intersections between the rays and the plane.
    expected_absolute_intensities : torch.Tensor
        The expected absolute intensities of the ray intersections.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    bitmap_intersections_e, bitmap_intersections_u, intersection_distances, intensities = raytracing_utils.line_plane_intersections(
        rays=rays,
        points_at_ray_origins=points_at_ray_origins.to(device),
        target_areas=request.getfixturevalue(target_areas_fixture),
        target_area_indices=target_area_indices,
        device=device,
    )

    torch.testing.assert_close(
        bitmap_intersections_e, expected_intersections_e.to(device), rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        bitmap_intersections_u, expected_intersections_u.to(device), rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        intersection_distances, expected_intersection_distances.to(device), rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        intensities,
        expected_absolute_intensities.to(device),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.fixture
def target_area_1_cylindrical(device: torch.device) -> TowerTargetAreasCylindrical:
    centers = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
    normals = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device)
    axes = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=device)
    radii = torch.tensor([1.0], device=device)
    heights = torch.tensor([2.0], device=device)
    opening_angles = torch.tensor([2*torch.pi], device=device)

    return TowerTargetAreasCylindrical(
        names=["cylinder1"],
        centers=centers,
        normals=normals,
        axes=axes,
        radii=radii,
        heights=heights,
        opening_angles=opening_angles
    )

@pytest.fixture
def target_area_2_cylindrical(device: torch.device) -> TowerTargetAreasCylindrical:
    centers = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
    normals = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device)
    axes = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=device)
    radii = torch.tensor([1.0], device=device)
    heights = torch.tensor([2.0], device=device)
    opening_angles = torch.tensor([torch.pi / 2], device=device)

    return TowerTargetAreasCylindrical(
        names=["cylinder2"],
        centers=centers,
        normals=normals,
        axes=axes,
        radii=radii,
        heights=heights,
        opening_angles=opening_angles
    )


@pytest.mark.parametrize(
    (
        "rays",
        "target_areas_fixture",
        "points_at_ray_origins",
        "expected_intersections_e",
        "expected_intersections_u",
        "expected_intersection_distances",
        "expected_absolute_intensities",
    ),
    [
        # Single intersection with ray perpendicular to plane.
        (
            (torch.tensor([[[[0.0, -1.0, 0.0, 0.0]]]]), torch.tensor([[[1.0]]])),
            "target_area_1_cylindrical",
            torch.tensor([[[0.0, 5.0, 0.0, 1.0]]]),
            torch.tensor([[[127.5]]]),
            torch.tensor([[[127.5]]]),
            torch.tensor([[[4.0]]]),
            torch.tensor([[[1.0]]]),
        ),
        # Single ray, hits outside valid height.
        (
            (torch.tensor([[[[0.0, -0.2425, 0.9701, 0.0]]]]), torch.tensor([[[1.0]]])),
            "target_area_1_cylindrical",
            torch.tensor([[[0.0, 2.0, 0.0, 1.0]]]),
            torch.tensor([[[0.0]]]),
            torch.tensor([[[0.0]]]),
            torch.tensor([[[0.0]]]),
            torch.tensor([[[0.0]]]),
        ),
        # Single ray outside of opening angle.
        (
            (torch.tensor([[[[-1.0, 0.0, 0.0, 0.0]]]]), torch.tensor([[[1.0]]])),
            "target_area_2_cylindrical",
            torch.tensor([[[3.0, 0.0, 0.0, 1.0]]]),
            torch.tensor([[[0.0]]]),
            torch.tensor([[[0.0]]]),
            torch.tensor([[[0.0]]]),
            torch.tensor([[[0.0]]]),
        ),
        # Single ray completely misses the cylinder.
        (
            (torch.tensor([[[[-1.0, 1.0, 0.0, 0.0]]]]), torch.tensor([[[1.0]]])),
            "target_area_1_cylindrical",
            torch.tensor([[[-10.0, 0.0, 0.0, 1.0]]]),
            torch.tensor([[[0.0]]]),
            torch.tensor([[[0.0]]]),
            torch.tensor([[[0.0]]]),
            torch.tensor([[[0.0]]]),
        ),
        # Single ray with origin on the cylinder.
        (
            (torch.tensor([[[[-1.0, 1.0, 0.0, 0.0]]]]), torch.tensor([[[1.0]]])),
            "target_area_1_cylindrical",
            torch.tensor([[[0.0, 1.0, 0.0, 1.0]]]),
            torch.tensor([[[0.0]]]),
            torch.tensor([[[0.0]]]),
            torch.tensor([[[0.0]]]),
            torch.tensor([[[0.0]]]),
        ),
    ],
    indirect=["rays"]
)
def test_line_cylinder_intersection(
    request: pytest.FixtureRequest,
    target_area_indices: torch.Tensor | None,
    rays: Rays,
    target_areas_fixture: str,
    points_at_ray_origins: torch.Tensor,
    expected_intersections_e: torch.Tensor,
    expected_intersections_u: torch.Tensor,
    expected_intersection_distances: torch.Tensor,
    expected_absolute_intensities: torch.Tensor,
    device: torch.device,
):
    bitmap_intersections_e, bitmap_intersections_u, intersection_distances, intensities = raytracing_utils.line_cylinder_intersections(
        rays=rays,
        points_at_ray_origins=points_at_ray_origins.to(device),
        target_areas=request.getfixturevalue(target_areas_fixture),
        target_area_indices=target_area_indices,
        device=device,
    )

    torch.testing.assert_close(bitmap_intersections_e, expected_intersections_e.to(device), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(bitmap_intersections_u, expected_intersections_u.to(device), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(intersection_distances, expected_intersection_distances.to(device), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(intensities, expected_absolute_intensities.to(device), rtol=1e-4, atol=1e-4)