"""This pytest considers loading a heliostat surface from a point cloud."""

import pathlib
from typing import Dict

import pytest
import torch

from artist import ARTIST_ROOT
from artist.io.datapoint import HeliostatDataPoint, HeliostatDataPointLabel
from artist.physics_objects.heliostats.heliostat import HeliostatModule
from artist.physics_objects.heliostats.concentrator.tests import surface_defaults
from artist.scenario.light_source.sun import Sun


def generate_data(
    light_direction: torch.Tensor, expected_value: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Generate all the relevant data for this test.

    This includes the position of the heliostat, the position of the receiver,
    the sun as a light source, and the point cloud as the heliostat surface.

    The facets of the heliostat surface are loaded from a point cloud.
    The surface points and surface normals are calculated.
    The surface points and normals are aligned.

    Parameters
    ----------
    light_direction : torch.Tensor
        The direction of the light.
    expected_value : torch.Tensor
        The expected bitmaps for the given test-cases.

    Returns
    -------
    Dict[str, torch.Tensor]
        A dictionary containing all the data.
    """
    cfg_default_surface = surface_defaults.get_cfg_defaults()
    surface_config = surface_defaults.load_config_file(cfg_default_surface)
    
    receiver_center = torch.tensor([0.0, -50.0, 0.0]).reshape(-1, 1)
    
    datapoint = HeliostatDataPoint(
        point_id=1,
        light_directions=light_direction,#.reshape(-1, 1),
        desired_aimpoint=receiver_center,#.reshape(-1, 1),
        label=HeliostatDataPointLabel(),
    )
    
    covariance = 4.3681e-06  # circum-solar ratio
    sun = Sun(
        "Normal", ray_count=100, mean=[0.0, 0.0], covariance=[[covariance, 0.0], [0.0, covariance]]
    )

    heliostat = HeliostatModule(surface_config)
    
    aligned_surface_points, aligned_surface_normals = heliostat.get_aligned_surface(datapoint=datapoint)

    return {
        "sun": sun,
        "aligned_surface_points": aligned_surface_points,
        "aligned_surface_normals": aligned_surface_normals,
        "receiver_center": receiver_center,
        "light_direction": light_direction,
        "expected_value": expected_value,
    }


@pytest.fixture(
    params=[
        (torch.tensor([0.0, -1.0, 0.0]), "south.pt"),
        (torch.tensor([1.0, 0.0, 0.0]), "east.pt"),
        (torch.tensor([-1.0, 0.0, 0.0]), "west.pt"),
        (torch.tensor([0.0, 0.0, 1.0]), "above.pt"),
    ],
    name="environment_data",
)
def data(request) -> Dict[str, torch.Tensor]:
    """
    [INSERT DESCRIPTION HERE!].

    Parameters
    ----------
    request :
        [INSERT DESCRIPTION HERE!]

    Returns
    -------
    Dict[str, torch.Tensor]
        [INSERT DESCRIPTION HERE!]
    """
    return generate_data(*request.param)


def test_compute_bitmaps(environment_data: Dict[str, torch.Tensor]) -> None:
    """
    Compute resulting flux density distribution (bitmap) for the given test case.

    With the aligned surface and the light direction, calculate the reflected rays on the heliostat surface.
    Calculate the intersection on the receiver.
    Compute the bitmaps and normalize them.
    Compare the calculated bitmaps with the expected ones.

    Parameters
    ----------
    environment_data : Dict[str, torch.Tensor]
        The dictionary containing all the data to compute the bitmaps.
    """
    torch.manual_seed(7)
    sun = environment_data["sun"]
    aligned_surface_points = environment_data["aligned_surface_points"]
    aligned_surface_normals = environment_data["aligned_surface_normals"]
    receiver_center = environment_data["receiver_center"]
    light_direction = environment_data["light_direction"]
    expected_value = environment_data["expected_value"]

    receiver_plane_normal = torch.tensor([0.0, 1.0, 0.0])
    receiver_plane_x = 8.629666667
    receiver_plane_y = 7.0
    receiver_resolution_x = 256
    receiver_resolution_y = 256
    sun_position = light_direction

    ray_directions = sun.reflect_rays_(-sun_position, aligned_surface_normals)

    xi, zi = sun.sample_distortions(len(ray_directions))

    rays = sun.scatter_rays(
        ray_directions,
        xi,
        zi,
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
        xi.numel(),
        receiver_plane_x,
        receiver_plane_y,
    )
    
    expected_path = (
        pathlib.Path(ARTIST_ROOT)
        / "artist/physics_objects/heliostats/tests/test_bitmaps"
        / expected_value
    )

    expected = torch.load(expected_path)

    torch.testing.assert_close(total_bitmap, expected, atol=5e-5, rtol=5e-5)
