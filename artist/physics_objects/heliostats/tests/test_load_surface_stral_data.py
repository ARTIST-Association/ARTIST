import os
from matplotlib import pyplot as plt
import pytest

import torch

from artist import ARTIST_ROOT
from artist.io.datapoint import HeliostatDataPoint, HeliostatDataPointLabel
from artist.physics_objects.heliostats.alignment.alignment import AlignmentModule
from artist.physics_objects.heliostats.surface.concentrator import ConcentratorModule
from artist.physics_objects.heliostats.surface.facets.point_cloud_facets import (
    PointCloudFacetModule,
)
from artist.physics_objects.heliostats.surface.tests import surface_defaults
from artist.scenario.light_source.sun import Sun


def generate_data(light_direction, expected_value):
    position = torch.Tensor([0.0, 5.0, 0.0])
    receiver_center = torch.Tensor([0.0, -50.0, 0.0])

    cfg_default_surface = surface_defaults.get_cfg_defaults()
    surface_config = surface_defaults.load_config_file(cfg_default_surface)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cov = 4.3681e-06
    sun = Sun("Normal", 100, [0, 0], [[cov, 0], [0, cov]], device)

    point_cloud_facets = PointCloudFacetModule(
        surface_config, receiver_center, torch.tensor(light_direction), device
    )
    facets = point_cloud_facets.make_facets_list()
    concentrator = ConcentratorModule(facets)
    surface_points, surface_normals = concentrator.get_surface()

    alignment_model = AlignmentModule(position=position)
    datapoint = HeliostatDataPoint(
        point_id=1,
        light_directions=torch.tensor(light_direction),
        desired_aimpoint=receiver_center,
        label=HeliostatDataPointLabel(),
    )
    (
        aligned_surface_points,
        aligned_surface_normals,
    ) = alignment_model.align_surface(
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
        ([0.0, -1.0, 0.0], "south.pt"),
        ([1.0, 0.0, 0.0], "east.pt"),
        ([-1.0, 0.0, 0.0], "west.pt"),
        ([0.0, 0.0, 1.0], "above.pt"),
    ],
    name="sun_data",
)
def sun_data(request):
    return generate_data(*request.param)


def test_compute_bitmaps(sun_data):
    torch.manual_seed(7)
    sun = sun_data["sun"]
    aligned_surface_points = sun_data["aligned_surface_points"]
    aligned_surface_normals = sun_data["aligned_surface_normals"]
    receiver_center = sun_data["receiver_center"]
    light_direction = sun_data["light_direction"]
    expected_value = sun_data["expected_value"]

    receiver_plane_normal = torch.tensor([0.0, 1.0, 0.0])
    receiver_plane_x = 8.629666667
    receiver_plane_y = 7.0
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

    expected_path = os.path.join(
        "artist",
        "physics_objects",
        "heliostats",
        "tests",
        "test_bitmaps",
        expected_value,
    )

    expected = torch.load(expected_path)

    # plt.imshow(total_bitmap.T, cmap="jet", origin="lower")
    # plt.show()
    # plt.imshow(expected.T, cmap="jet", origin="lower")
    # plt.show()

    # loss = torch.nn.L1Loss()
    # loss = loss(total_bitmap, expected)
    # torch.le(loss, torch.tensor(0.0000001))

    torch.testing.assert_close(total_bitmap, expected)
