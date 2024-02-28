"""
This pytest considers loading a heliostat surface from a pointcloud.
"""

import pathlib

import pytest
import torch

from artist import ARTIST_ROOT
from artist.environment.light_source.sun import Sun
from artist.physics_objects.heliostats.heliostat import HeliostatModule
from artist.tests.physics_objects.heliostats.concentrator import concentrator_defaults


def generate_data(
    incident_ray_direction: torch.Tensor, expected_value: str
) -> dict[str, torch.Tensor]:
    """
    Generate all the relevant data for this test.

    This includes the position of the heliostat, the position of the receiver,
    the sun as a light source, and the pointcloud as the heliostat surface.

    The facets of the heliostat surface are loaded from a pointcloud.
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
    dict[str, torch.Tensor]
        A dictionary containing all the data.
    """
    cfg_default_surface = concentrator_defaults.get_cfg_defaults()
    surface_config = concentrator_defaults.load_config_file(cfg_default_surface)

    receiver_center = torch.tensor([0.0, -50.0, 0.0]).reshape(-1, 1)

    sun_parameters = {
        "distribution_type": "Normal",
        "mean": 0.0,
        "covariance": 4.3681e-06,  # circum-solar ratio
    }

    sun = Sun(distribution_parameters=sun_parameters, ray_count=100)

    heliostat = HeliostatModule(incident_ray_direction=incident_ray_direction,
                                config_file=surface_config)

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
        (torch.tensor([0.0, -1.0, 0.0]), "south.pt"),
        (torch.tensor([1.0, 0.0, 0.0]), "east.pt"),
        (torch.tensor([-1.0, 0.0, 0.0]), "west.pt"),
        (torch.tensor([0.0, 0.0, 1.0]), "above.pt"),
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

    receiver_plane_normal = torch.tensor([0.0, 1.0, 0.0])
    receiver_plane_x = 8.629666667
    receiver_plane_y = 7.0
    receiver_resolution_x = 256
    receiver_resolution_y = 256

    # Calculate preferred directions of the (?) rays sent out by the heliostat surface.
    # These rays originate from reflection of the `num_rays_heliostat` (?) incoming sun
    # rays hitting that heliostat's surface. The heliostat surface is described by `aligned_surface_normals`.
    # For each normal used to describe the heliostat surface, only the outgoing rays'
    # preferred directions are returned, which are to be scattered or distorted in the next step.

    # heliostat besteht aus 4 surface_points (4, 3) und dazugehörig (4, 3) normals
    # Berechne mit sonnenvektor (incident_ray_direction) shape: (1, 3) die preferred_directions (4, 3)
    # sende ray_count strahlen pro normal_vector aus und störe sie um xi, yi, shape: (ray_count, 4, 3)
    # Beispiel ray_count = 2 -> ray_directions shape: (8, 3)
    #

    preferred_ray_directions = sun.get_preferred_reflection_direction(
        -incident_ray_direction, aligned_surface_normals
    )

    distortion_x, distortion_y = sun.sample(len(preferred_ray_directions))

    rays = sun.compute_rays(
        receiver_plane_normal,
        receiver_center,
        preferred_ray_directions,
        aligned_surface_points,
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

    expected_path = (
        pathlib.Path(ARTIST_ROOT)
        / "artist/tests/physics_objects/heliostats/test_bitmaps"
        / expected_value
    )

    expected = torch.load(expected_path)

    torch.testing.assert_close(total_bitmap, expected)
