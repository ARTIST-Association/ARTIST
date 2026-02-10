import pathlib

import h5py
import pytest
import torch

from artist import ARTIST_ROOT
from artist.core import blocking
from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.scenario.scenario import Scenario
from artist.util import utils


@pytest.fixture
def surface_at_origin() -> torch.Tensor:
    """
    Define a surface at the origin.

    Returns
    -------
    torch.Tensor
        The surface.
    """
    corner_points = torch.tensor(
        [
            [-1.0, -0.5, 0.0, 1.0],
            [1.0, -0.5, 0.0, 1.0],
            [1.0, 0.5, 0.0, 1.0],
            [-1.0, 0.5, 0.0, 1.0],
        ],
    )

    interior_points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.5, 0.0, 0.0, 1.0],
        ],
    )

    surface_points = torch.cat([corner_points, interior_points], dim=0)[None, :, :]

    return surface_points


@pytest.fixture
def surface_rotated_and_translated(surface_at_origin: torch.Tensor) -> torch.Tensor:
    """
    Define a rotated and translated surface.

    Parameters
    ----------
    surface_at_origin : torch.Tensor
        A surface at the origin.

    Returns
    -------
    torch.Tensor
        The surface.
    """
    device = surface_at_origin.device
    rotation_e = utils.rotate_e(e=torch.tensor([0.5]), device=device)
    rotation_n = utils.rotate_n(n=torch.tensor([0.2]), device=device)

    translation = torch.tensor(
        [
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, 3.0],
            [0.0, 0.0, 1.0, 1.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
    )

    transform = translation.T @ rotation_n @ rotation_e

    transformed_surface = surface_at_origin @ transform

    return transformed_surface


@pytest.mark.parametrize(
    "surface, transformed_surface, expected",
    [
        (
            "surface_at_origin",
            "surface_at_origin",
            [
                torch.tensor(
                    [
                        [
                            [-1.0, -0.5, 0.0, 1.0],
                            [1.0, -0.5, 0.0, 1.0],
                            [1.0, 0.5, 0.0, 1.0],
                            [-1.0, 0.5, 0.0, 1.0],
                        ]
                    ],
                ),
                torch.tensor([[[2.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]),
                torch.tensor([[0.0, 0.0, 1.0]]),
            ],
        ),
        (
            "surface_at_origin",
            "surface_rotated_and_translated",
            [
                torch.tensor(
                    [
                        [
                            [1.2781, 2.8035, -0.0828, 1.0000],
                            [3.2382, 2.6130, -0.4315, 1.0000],
                            [3.2382, 3.4906, -0.9109, 1.0000],
                            [1.2781, 3.6811, -0.5622, 1.0000],
                        ]
                    ]
                ),
                torch.tensor(
                    [
                        [
                            [1.9601, -0.1905, -0.3487, 0.0000],
                            [0.0000, 0.8776, -0.4794, 0.0000],
                        ]
                    ]
                ),
                torch.tensor([[0.1987, 0.4699, 0.8601]]),
            ],
        ),
    ],
)
def test_create_blocking_primitives_rectangle(
    surface: torch.Tensor,
    transformed_surface: torch.Tensor,
    expected: list[torch.Tensor],
    request: pytest.FixtureRequest,
    device: torch.device,
) -> None:
    """
    Test that the creation of blocking primitives works as desired.

    Parameter
    ---------
    surface : torch.Tensor
        Surface at the origin.
    transformed_surface : torch.Tensor
        Surface that has been transformed.
    expected : torch.Tensor
        The expected tensors.
    request : pytest.FixtureRequest
        The pytest fixture used to consider different test cases.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    blocking_surface_points = request.getfixturevalue(surface)
    active_surface_points = request.getfixturevalue(transformed_surface)

    corners, spans, normals = blocking.create_blocking_primitives_rectangle(
        blocking_heliostats_surface_points=blocking_surface_points.to(device),
        blocking_heliostats_active_surface_points=active_surface_points.to(device),
        epsilon=0.1,
        device=device,
    )

    torch.testing.assert_close(corners, expected[0].to(device), atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(spans, expected[1].to(device), atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(normals, expected[2].to(device), atol=5e-4, rtol=5e-4)


@pytest.fixture
def surface_for_index_test(device: torch.device) -> torch.Tensor:
    """
    Define a rotated and translated surface.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    torch.Tensor
        The surface.
    """
    points_per_axis = 5

    facet_coordinates: list[torch.Tensor] = []
    facet_origins = [
        (0.0, 0.0),
        (2.0, 0.0),
        (2.0, 1.0),
        (0.0, 2.0),
    ]

    facet_size = (2.0, 2.0)

    for x, y in facet_origins:
        xs = torch.linspace(x, x + facet_size[0], points_per_axis, device=device)
        ys = torch.linspace(y, y + facet_size[1], points_per_axis, device=device)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        grid_z = torch.zeros_like(grid_x)
        points = torch.stack(
            [grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=-1
        )
        facet_coordinates.append(points)

    all_points = torch.cat(facet_coordinates, dim=0)
    all_points = torch.cat(
        [all_points, torch.ones(all_points.shape[0], 1, device=device)], dim=-1
    )
    all_points = all_points[None, :, :]

    return all_points


@pytest.fixture
def surface_for_index_test_rotated_and_translated(
    surface_for_index_test,
) -> torch.Tensor:
    """
    Define a rotated and translated surface.

    Parameters
    ----------
    surface_for_index_test : torch.Tensor
        Surface at the origin, not rotated.

    Returns
    -------
    torch.Tensor
        The surface.
    """
    device = surface_for_index_test.device
    rotation_e = utils.rotate_e(e=torch.tensor([0.5]), device=device)
    rotation_n = utils.rotate_n(n=torch.tensor([0.2]), device=device)

    translation = torch.tensor(
        [
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, 3.0],
            [0.0, 0.0, 1.0, 1.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
    )

    transform = translation.T @ rotation_n @ rotation_e

    active_surface_points = surface_for_index_test @ transform
    return active_surface_points


@pytest.mark.parametrize(
    "surface, expected",
    [
        (
            "surface_for_index_test",
            [
                torch.tensor(
                    [
                        [
                            [2.0, 1.0, 0.0, 1.0],
                            [0.0, 2.0, 0.0, 1.0],
                            [4.0, 2.0, 0.0, 1.0],
                            [2.0, 2.0, 0.0, 1.0],
                        ]
                    ]
                ),
                torch.tensor([[[-2.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]),
                torch.tensor([[0.0, 0.0, -1.0]]),
            ],
        ),
        (
            "surface_for_index_test_rotated_and_translated",
            [
                torch.tensor(
                    [
                        [
                            [4.2183, 3.8341, -1.3250, 1.0000],
                            [2.2581, 4.9022, -1.4557, 1.0000],
                            [6.1784, 4.5212, -2.1531, 1.0000],
                            [4.2183, 4.7117, -1.8044, 1.0000],
                        ]
                    ]
                ),
                torch.tensor(
                    [
                        [
                            [-1.9601, 1.0681, -0.1307, 0.0000],
                            [0.0000, 0.8776, -0.4794, 0.0000],
                        ]
                    ]
                ),
                torch.tensor([[-0.1987, -0.4699, -0.8601]]),
            ],
        ),
    ],
)
def test_create_blocking_primitives_rectangles_by_index(
    surface: torch.Tensor,
    expected: list[torch.Tensor],
    request: pytest.FixtureRequest,
    device: torch.device,
) -> None:
    """
    Test that the creation of blocking primitives works as desired.

    Parameter
    ---------
    surface : torch.Tensor
        Surface randomly transformed.
    expected : torch.Tensor
        The expected tensors.
    request : pytest.FixtureRequest
        The pytest fixture used to consider different test cases.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    active_surface_points = request.getfixturevalue(surface)

    corners, spans, normals = blocking.create_blocking_primitives_rectangles_by_index(
        blocking_heliostats_active_surface_points=active_surface_points.to(device),
        device=device,
    )

    torch.testing.assert_close(corners, expected[0].to(device), atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(spans, expected[1].to(device), atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(normals, expected[2].to(device), atol=5e-4, rtol=5e-4)


def test_blocking_integration(device: torch.device) -> None:
    """
    Test all blocking methods in an integration test.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    # Load the scenario.
    with h5py.File(
        pathlib.Path(ARTIST_ROOT) / "tests/data/scenarios/test_blocking.h5",
        "r",
    ) as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file,
            device=device,
        )

    incident_ray_direction = torch.nn.functional.normalize(
        torch.tensor([0.0, 1.0, 0.0, 0.0], device=device), dim=-1
    )

    heliostat_group = scenario.heliostat_field.heliostat_groups[0]
    heliostat_target_light_source_mapping = [
        ("heliostat_0", "target_0", incident_ray_direction),
        ("heliostat_1", "target_0", incident_ray_direction),
        ("heliostat_2", "target_0", incident_ray_direction),
        ("heliostat_3", "target_0", incident_ray_direction),
        ("heliostat_4", "target_0", incident_ray_direction),
        ("heliostat_5", "target_0", incident_ray_direction),
    ]

    (
        active_heliostats_mask,
        target_area_mask,
        incident_ray_directions,
    ) = scenario.index_mapping(
        heliostat_group=heliostat_group,
        string_mapping=heliostat_target_light_source_mapping,
        device=device,
    )

    heliostat_group.activate_heliostats(
        active_heliostats_mask=active_heliostats_mask, device=device
    )

    heliostat_group.align_surfaces_with_incident_ray_directions(
        aim_points=scenario.target_areas.centers[target_area_mask],
        incident_ray_directions=incident_ray_directions,
        active_heliostats_mask=active_heliostats_mask,
        device=device,
    )

    scenario.set_number_of_rays(number_of_rays=200)

    ray_tracer = HeliostatRayTracer(
        scenario=scenario,
        heliostat_group=heliostat_group,
        blocking_active=True,
        batch_size=10,
    )

    bitmaps_per_heliostat = ray_tracer.trace_rays(
        incident_ray_directions=incident_ray_directions,
        active_heliostats_mask=active_heliostats_mask,
        target_area_mask=target_area_mask,
        device=device,
    )

    expected_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/data/expected_bitmaps_blocking"
        / f"bitmaps_{device.type}.pt"
    )

    expected = torch.load(expected_path, map_location=device, weights_only=True)

    torch.testing.assert_close(bitmaps_per_heliostat, expected, atol=5e-4, rtol=5e-4)


def test_ray_extinction(device: torch.device) -> None:
    """
    Test the ray extinction.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    with h5py.File(
        pathlib.Path(ARTIST_ROOT)
        / "tests/data/scenarios/test_scenario_paint_single_heliostat.h5",
        "r",
    ) as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file,
            device=device,
        )

    heliostat_group = scenario.heliostat_field.heliostat_groups[0]

    (
        active_heliostats_mask,
        target_area_mask,
        incident_ray_directions,
    ) = scenario.index_mapping(
        heliostat_group=heliostat_group,
        device=device,
    )

    heliostat_group.activate_heliostats(
        active_heliostats_mask=active_heliostats_mask, device=device
    )

    heliostat_group.align_surfaces_with_incident_ray_directions(
        aim_points=scenario.target_areas.centers[target_area_mask],
        incident_ray_directions=incident_ray_directions,
        active_heliostats_mask=active_heliostats_mask,
        device=device,
    )

    scenario.set_number_of_rays(number_of_rays=200)

    ray_tracer = HeliostatRayTracer(
        scenario=scenario,
        heliostat_group=heliostat_group,
        blocking_active=True,
        batch_size=10,
    )

    ray_extinction_factor = 0.9

    bitmaps_per_heliostat_no_extinction = ray_tracer.trace_rays(
        incident_ray_directions=incident_ray_directions,
        active_heliostats_mask=active_heliostats_mask,
        target_area_mask=target_area_mask,
        device=device,
    )

    bitmaps_per_heliostat_extinction = ray_tracer.trace_rays(
        incident_ray_directions=incident_ray_directions,
        active_heliostats_mask=active_heliostats_mask,
        target_area_mask=target_area_mask,
        ray_extinction_factor=ray_extinction_factor,
        device=device,
    )

    bitmaps = torch.cat(
        (bitmaps_per_heliostat_no_extinction, bitmaps_per_heliostat_extinction)
    )

    expected_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/data/expected_bitmaps_ray_extinction"
        / f"bitmaps_{device.type}.pt"
    )

    expected = torch.load(expected_path, map_location=device, weights_only=True)

    torch.testing.assert_close(bitmaps, expected, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(
        bitmaps_per_heliostat_no_extinction[0].sum() * (1 - ray_extinction_factor),
        bitmaps_per_heliostat_extinction[0].sum(),
        atol=5e-4,
        rtol=5e-4,
    )
