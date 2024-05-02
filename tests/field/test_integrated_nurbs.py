import pathlib
import torch
import h5py
import pytest

from artist import ARTIST_ROOT
from artist.field.heliostat import Heliostat
from artist.scene.sun import Sun
from artist.util import config_dictionary
from artist.util import nurbs_converters


@pytest.fixture(scope="module")
def common_setup():
    with h5py.File(f"{ARTIST_ROOT}/scenarios/test_scenario.h5", "r") as config_h5:
        receiver_center = torch.tensor(
            config_h5[config_dictionary.receiver_prefix][
                config_dictionary.receiver_center
            ][()],
            dtype=torch.float,
        )
        sun = Sun.from_hdf5(config_file=config_h5)
        heliostat = Heliostat.from_hdf5(
            heliostat_name="Single_Heliostat",
            incident_ray_direction=torch.tensor([0.0, 0.0, 0.0, 0.0]),
            config_file=config_h5,
        )
    
    width = 2
    height = 2
    nurbs_surface = nurbs_converters.deflectometry_to_nurbs(heliostat.concentrator.facets.surface_points[::100], heliostat.concentrator.facets.surface_normals[::100], width, height)
    
    surface_points, surface_normals = nurbs_surface.calculate_surface_points_and_normals()

    heliostat.concentrator.facets.surface_points = surface_points
    heliostat.concentrator.facets.surface_normals = surface_normals

    return receiver_center, sun, heliostat


@pytest.mark.parametrize(
        "incident_ray_direction, expected_value",
        [
            (torch.tensor([0.0, -1.0, 0.0, 0.0]), "south.pt"),
            (torch.tensor([1.0, 0.0, 0.0, 0.0]), "east.pt"),
            (torch.tensor([-1.0, 0.0, 0.0, 0.0]), "west.pt"),
            (torch.tensor([0.0, 0.0, 1.0, 0.0]), "above.pt"),
        ]
)
def test_nurbs(common_setup, incident_ray_direction, expected_value):

    receiver_center, sun, heliostat = common_setup
    
    heliostat.incident_ray_direction = incident_ray_direction

    aligned_surface_points, aligned_surface_normals = heliostat.get_aligned_surface()

    torch.manual_seed(7)

    receiver_plane_normal = torch.tensor([0.0, 1.0, 0.0, 0.0])
    receiver_plane_x = 8.629666667
    receiver_plane_y = 7.0
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

    expected_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/physics_objects/test_bitmaps_integrated_nurbs"
        / expected_value
    )

    expected = torch.load(expected_path)

    torch.testing.assert_close(total_bitmap.T, expected, atol=5e-4, rtol=5e-4)

