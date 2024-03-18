"""
This pytest considers loading a heliostat surface from a pointcloud.
"""

import pathlib

import h5py
from matplotlib import pyplot as plt
import numpy as np
import pytest
import torch

from artist import ARTIST_ROOT
from artist.environment.light_source.sun import Sun
from artist.physics_objects.heliostats.heliostat import HeliostatModule
from artist.util import config_dictionary


def generate_data(
    incident_ray_direction: torch.Tensor,
    expected_value: str,
    scenario_config: str,
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
    scenario_config : str
        The name of the scenario config that should be loaded.

    Returns
    -------
    dict[str, torch.Tensor]
        A dictionary containing all the data.
    """
    with h5py.File(f"{ARTIST_ROOT}/scenarios/{scenario_config}.h5", "r") as config_h5:
        receiver_center = torch.tensor(
            config_h5[config_dictionary.receiver_prefix][
                config_dictionary.receiver_center
            ][()],
            dtype=torch.float,
        )
        sun = Sun.from_hdf5(config_file=config_h5)
        heliostat = HeliostatModule.from_hdf5(
            heliostat_name="Single_Heliostat",
            incident_ray_direction=incident_ray_direction,
            config_file=config_h5,
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
        (torch.tensor([0.0, -1.0, 0.0, 0.0]), "south.pt", "test_scenario_2"),
        (torch.tensor([1.0, 0.0, 0.0, 0.0]), "east.pt", "test_scenario"),
        (torch.tensor([-1.0, 0.0, 0.0, 0.0]), "west.pt", "test_scenario"),
        (torch.tensor([0.0, 0.0, 1.0, 0.0]), "above.pt", "test_scenario"),
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

    receiver_plane_normal = torch.tensor([0.0, 1.0, 0.0, 0.0])
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

    selected_indices = np.arange(0, 161512, 300)
    selected_points = aligned_surface_points[selected_indices, :]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # ax.quiver(0, -50, 0, receiver_plane_normal[0], receiver_plane_normal[1], receiver_plane_normal[2], color='r')
    ax.quiver(
        0,
        0,
        0,
        -incident_ray_direction[0],
        -incident_ray_direction[1],
        -incident_ray_direction[2],
        color="g",
    )
    ax.scatter(
        selected_points[:, 0].detach().numpy(),
        selected_points[:, 1].detach().numpy(),
        selected_points[:, 2].detach().numpy(),
        c="b",
        marker="o",
    )
    ax.set_xlabel("E")
    ax.set_ylabel("N")
    ax.set_zlabel("U")
    ax.set_xlim(-3, 3)
    # ax.set_ylim(-50, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    plt.show()

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

    plt.imshow(total_bitmap.T.detach().numpy(), origin="lower", cmap="jet")
    plt.show()

    expected_path = (
        pathlib.Path(ARTIST_ROOT)
        / "artist/tests/physics_objects/heliostats/test_bitmaps"
        / expected_value
    )

    expected = torch.load(expected_path)
    plt.imshow(expected.T, origin="lower", cmap="jet")
    plt.show()

    torch.testing.assert_close(total_bitmap, expected)
