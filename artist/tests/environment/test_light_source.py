"""This pytest tests the correctness of the light source."""

import math
import pathlib
from typing import Any, Dict, Tuple

import pytest
import torch

from artist import ARTIST_ROOT
from artist.environment.sun import Sun
from artist.physics_objects.heliostats.heliostat import HeliostatModule
from artist.util import config_dictionary


def generate_data(
    incident_ray_direction: torch.Tensor,
    expected_value: str,
) -> Dict[str, torch.Tensor]:
    """
    Generate all the relevant data for this test.

    This includes the deviation parameters of the kinematic module, the position of the heliostat, the position of the receiver,
    the sun as a light source, and five manually created surface points and normals.

    The surface points and normals are aligned by the kinematic module of the Heliost module.

    Parameters
    ----------
    incident_ray_direction : torch.Tensor
        The direction of the light.
    expected_value : str
        The expected bitmaps for the given test-cases.

    Returns
    -------
    dict[str, torch.Tensor]
        A dictionary containing all the data.
    """
    deviation_parameters = {
        config_dictionary.first_joint_translation_e: torch.tensor(0.0),
        config_dictionary.first_joint_translation_n: torch.tensor(0.0),
        config_dictionary.first_joint_translation_u: torch.tensor(0.0),
        config_dictionary.first_joint_tilt_e: torch.tensor(0.0),
        config_dictionary.first_joint_tilt_n: torch.tensor(0.0),
        config_dictionary.first_joint_tilt_u: torch.tensor(0.0),
        config_dictionary.second_joint_translation_e: torch.tensor(0.0),
        config_dictionary.second_joint_translation_n: torch.tensor(0.0),
        config_dictionary.second_joint_translation_u: torch.tensor(0.0),
        config_dictionary.second_joint_tilt_e: torch.tensor(0.0),
        config_dictionary.second_joint_tilt_n: torch.tensor(0.0),
        config_dictionary.second_joint_tilt_u: torch.tensor(0.0),
        config_dictionary.concentrator_translation_e: torch.tensor(0.0),
        config_dictionary.concentrator_translation_n: torch.tensor(0.0),
        config_dictionary.concentrator_translation_u: torch.tensor(0.0),
        config_dictionary.concentrator_tilt_e: torch.tensor(0.0),
        config_dictionary.concentrator_tilt_n: torch.tensor(0.0),
        config_dictionary.concentrator_tilt_u: torch.tensor(0.0),
    }
    initial_orientation_offset: float = -math.pi / 2

    heliostat_position = torch.tensor([0.0, 5.0, 0.0, 1.0])
    receiver_center = torch.tensor([0.0, -10.0, 0.0, 1.0])

    sun = Sun()

    surface_normals = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    surface_points = torch.tensor(
        [
            [-1.0, -1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [-1.0, 1.0, 0.0, 1.0],
            [1.0, -1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    heliostat = HeliostatModule(
        id=1,
        position=heliostat_position,
        alignment_type="rigid_body",
        actuator_type="ideal_actuator",
        aim_point=receiver_center,
        facet_type="point_cloud_facet",
        surface_points=surface_points,
        surface_normals=surface_normals,
        incident_ray_direction=incident_ray_direction,
        kinematic_deviation_parameters=deviation_parameters,
        kinematic_initial_orientation_offset=initial_orientation_offset,
    )

    aligned_surface_points, aligned_surface_normals = heliostat.get_aligned_surface()

    return {
        "sun": sun,
        "aligned_surface_points": aligned_surface_points,
        "aligned_surface_normals": aligned_surface_normals,
        "receiver_center": receiver_center,
        "incident_ray_direction": incident_ray_direction,
        "expected_value": expected_value,
    }


@pytest.fixture(
    params=[
        (torch.tensor([0.0, 0.0, 1.0, 0.0]), "above.pt"),
        (torch.tensor([1.0, 0.0, 0.0, 0.0]), "east.pt"),
        (torch.tensor([-1.0, 0.0, 0.0, 0.0]), "west.pt"),
        (torch.tensor([0.0, -1.0, 0.0, 0.0]), "south.pt"),
    ],
    name="environment_data",
)
def data(request: Tuple[torch.Tensor, str]) -> Dict[str, Any]:
    """
    Compute the data required for the test.

    Parameters
    ----------
    request : (torch.Tensor, str)
        The pytest.fixture request with the incident ray direction and bitmap name required for the test.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the data required for the test.
    """
    return generate_data(*request.param)


def test_compute_bitmaps(environment_data: dict[str, torch.Tensor]) -> None:
    """
    Compute resulting flux density distribution (bitmap) for the given test case.

    With the aligned surface and the light direction, calculate the reflected rays on the heliostat surface.
    Calculate the intersection on the receiver.
    Compute the bitmaps and normalize them.
    Compare the calculated bitmaps with the expected ones.

    Parameters
    ----------
    environment_data : dict[str, torch.Tensor]
        The dictionary containing all the data to compute the bitmaps.
    """
    torch.manual_seed(7)
    sun = environment_data["sun"]
    aligned_surface_points = environment_data["aligned_surface_points"]
    aligned_surface_normals = environment_data["aligned_surface_normals"]
    receiver_center = environment_data["receiver_center"]
    incident_ray_direction = environment_data["incident_ray_direction"]
    expected_value = environment_data["expected_value"]

    receiver_plane_normal = torch.tensor([0.0, 1.0, 0.0, 0.0])
    receiver_plane_x = 10
    receiver_plane_y = 10
    receiver_resolution_x = 256
    receiver_resolution_y = 256

    preferred_ray_directions = sun.get_preferred_reflection_direction(
        -incident_ray_direction, aligned_surface_normals
    )

    distortions_n, distortions_u = sun.sample(preferred_ray_directions.shape[0])

    rays = sun.scatter_rays(
        preferred_ray_directions,
        distortions_n,
        distortions_u,
    )

    intersections = sun.line_plane_intersections(
        receiver_plane_normal, receiver_center, rays, aligned_surface_points
    )

    dx_ints = intersections[:, :, 0] + receiver_plane_x / 2 - receiver_center[0]
    dy_ints = intersections[:, :, 2] + receiver_plane_y / 2 - receiver_center[2]

    indices = (
        (-1 <= dx_ints)
        & (dx_ints < receiver_plane_x + 1)
        & (-1 <= dy_ints)
        & (dy_ints < receiver_plane_y + 1)
    )

    total_bitmap = sun.sample_bitmap(
        dx_ints,
        dy_ints,
        indices,
        receiver_plane_x,
        receiver_plane_y,
        receiver_resolution_x,
        receiver_resolution_y,
    )

    total_bitmap = sun.normalize_bitmap(
        total_bitmap,
        distortions_n.numel(),
        receiver_plane_x,
        receiver_plane_y,
    )

    total_bitmap = total_bitmap.T

    expected_path = (
        pathlib.Path(ARTIST_ROOT) / "artist/tests/environment/bitmaps" / expected_value
    )

    expected = torch.load(expected_path)

    torch.testing.assert_close(total_bitmap, expected, atol=5e-3, rtol=5e-3)