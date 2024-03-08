"""
This pytest tests the correctness of the light source.
"""

import pathlib
from typing import Dict

import pytest
import torch
import h5py

from artist import ARTIST_ROOT
from artist.environment.light_source.sun import Sun
from artist.io.datapoint import HeliostatDataPoint, HeliostatDataPointLabel
from artist.physics_objects.heliostats.alignment.alignment import AlignmentModule
from artist.physics_objects.heliostats.heliostat import HeliostatModule


def generate_data(
    incident_ray_direction: torch.Tensor,
    expected_value: str,
    scenario_config: str,
) -> Dict[str, torch.Tensor]:
    """
    Generate all the relevant data for this test.

    This includes the position of the heliostat, the position of the receiver,
    the sun as a light source, and five manually created surface points and normals.

    The surface points and normals are aligned by the kinematic module.

    Parameters
    ----------
    light_direction : torch.Tensor
        The direction of the light.
    expected_value : torch.Tensor
        The expected bitmaps for the given test-cases.
    scenario_config : str
        The name of the scenario config that should be loaded.

    Returns
    -------
    dict[str, torch.Tensor]
        A dictionary containing all the data.
    """
    heliostat_position = torch.tensor([[0.0], [5.0], [0.0], [1.0]])
    receiver_center = torch.tensor([[0.0], [-10.0], [0.0], [1.0]])
    
    sun = Sun()

    surface_normals = torch.tensor(
        [
            [[0.0], [0.0], [1.0], [1.0]],
            [[0.0], [0.0], [1.0], [1.0]],
            [[0.0], [0.0], [1.0], [1.0]],
            [[0.0], [0.0], [1.0], [1.0]],
            [[0.0], [0.0], [1.0], [1.0]],
        ]
    )
    surface_points = torch.tensor(
        [
            [[-1.0], [-1.0], [0.0], [1.0]],
            [[1.0], [1.0], [0.0], [1.0]],
            [[-1.0], [1.0], [0.0], [1.0]],
            [[1.0], [-1.0], [0.0], [1.0]],
            [[0.0], [0.0], [0.0], [1.0]],
        ]
    )

    heliostat = HeliostatModule(id=1,
                                position=heliostat_position,
                                alignment_type="rigid_body",
                                actuator_type="ideal_actuator",
                                aim_point=receiver_center,
                                facet_type="point_cloud_facet",
                                surface_points=surface_points,
                                surface_normals=surface_normals,
                                incident_ray_direction=incident_ray_direction)


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
        (torch.tensor([[0.0], [0.0], [1.0], [1.0]]), "above.pt", "test_scenario"),
        (torch.tensor([[1.0], [0.0], [0.0], [1.0]]), "east.pt", "test_scenario"),
        (torch.tensor([[-1.0], [0.0], [0.0], [1.0]]), "west.pt", "test_scenario"),
        (torch.tensor([[0.0], [-1.0], [0.0], [1.0]]), "south.pt", "test_scenario"),
    ],
    name="environment_data",
)
def data(request):
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

    receiver_plane_normal = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    receiver_plane_x = 10
    receiver_plane_y = 10
    receiver_resolution_x = 256
    receiver_resolution_y = 256

    preferred_ray_directions = sun.get_preferred_reflection_direction(
        -incident_ray_direction, aligned_surface_normals
    )

    distortion_x, distortion_y = sun.sample(preferred_ray_directions.shape[1])

    rays = sun.scatter_rays(
        preferred_ray_directions,
        distortion_x,
        distortion_y,
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
        distortion_x.numel(),
        receiver_plane_x,
        receiver_plane_y,
    )

    total_bitmap = total_bitmap.T

    expected_path = pathlib.Path(ARTIST_ROOT) / pathlib.Path(
        f"artist/tests/environment/light_source/bitmaps/{expected_value}"
    )

    expected = torch.load(expected_path)

    torch.testing.assert_close(total_bitmap, expected)
