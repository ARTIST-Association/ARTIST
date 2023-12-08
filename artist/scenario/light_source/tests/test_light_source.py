import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import pytest
from artist import ARTIST_ROOT

from artist.scenario.light_source.sun import Sun
from artist.io.datapoint import HeliostatDataPoint, HeliostatDataPointLabel
from artist.physics_objects.heliostats.alignment.alignment import AlignmentModule


def generate_sun_data(light_direction, expected_value):
    heliostat_position = torch.tensor([0.0, 5.0, 0.0])
    receiver_center = torch.tensor([0.0, -10.0, 0.0])

    cov = 1e-12  # 4.3681e-06
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sun = Sun("Normal", 300, [0, 0], [[cov, 0], [0, cov]], device)

    surface_normals = torch.tensor(
        [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
    )
    surface_points = torch.tensor(
        [[-1, -1, 0], [1, 1, 0], [-1, 1, 0], [1, -1, 0], [0, 0, 0]]
    )
    alignment_model = AlignmentModule(position=heliostat_position)

    datapoint = HeliostatDataPoint(
        point_id=1,
        light_directions=torch.tensor(light_direction),
        desired_aimpoint=receiver_center,
        label=HeliostatDataPointLabel(),
    )

    aligned_surface_points, aligned_surface_normals = alignment_model.align_surface(
        datapoint=datapoint,
        surface_points=surface_points,
        surface_normals=surface_normals,
    )

    return {
        "sun": sun,
        "aligned_surface_points": aligned_surface_points,
        "aligned_surface_normals": aligned_surface_normals,
        "receiver_center": receiver_center,
        "light_direction": torch.tensor(light_direction),
        "expected_value": expected_value,
    }


@pytest.fixture(
    params=[
        ([0.0, 0.0, 1.0], "above.pt"),
        ([1.0, 0.0, 0.0], "east.pt"),
        ([-1.0, 0.0, 0.0], "west.pt"),
        ([0.0, -1.0, 0.0], "south.pt"),
    ],
    name="sun_data",
)
def sun_data(request):
    return generate_sun_data(*request.param)


def test_compute_bitmaps(sun_data):
    torch.manual_seed(7)
    sun = sun_data["sun"]
    aligned_surface_points = sun_data["aligned_surface_points"]
    aligned_surface_normals = sun_data["aligned_surface_normals"]
    receiver_center = sun_data["receiver_center"]
    light_direction = sun_data["light_direction"]
    expected_value = sun_data["expected_value"]

    receiver_plane_normal = torch.tensor([0.0, 1.0, 0.0])
    receiver_plane_x = 10
    receiver_plane_y = 10
    receiver_resolution_x = 256
    receiver_resolution_y = 256
    sun_position = light_direction

    ray_directions = sun.reflect_rays_(-sun_position, aligned_surface_normals)

    xi, yi = sun.sample(len(ray_directions))

    rays = sun.compute_rays(
        receiver_plane_normal,
        receiver_center,
        ray_directions,
        aligned_surface_points,
        xi,
        yi,
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

    total_bitmap = total_bitmap.T

    expected_path = os.path.join(
        ARTIST_ROOT,
        "artist",
        "scenario",
        "light_source",
        "tests",
        "bitmaps",
        expected_value,
    )

    expected = torch.load(expected_path)

    torch.testing.assert_close(total_bitmap, expected)
