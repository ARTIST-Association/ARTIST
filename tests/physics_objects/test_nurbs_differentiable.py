import datetime
import math
import pathlib
from typing import Dict

from matplotlib import pyplot as plt
import pytest
import torch
import h5py

from artist import ARTIST_ROOT
from artist.field.nurbs_differentiable import NURBSSurface
from artist.field.heliostat import Heliostat
from artist.scene.sun import Sun
from artist.util import config_dictionary

def generate_data(
    incident_ray_direction: torch.Tensor,
    expected_value: str,
    scenario_config: str,
) -> Dict[str, torch.Tensor]:
    """
    Generate all the relevant data for this test.

    This includes the position of the heliostat, the position of the receiver,
    the sun as a light source, and the point cloud as the heliostat surface.

    The facets/points of the heliostat surface are loaded from a point cloud.
    The surface points and normals are aligned.

    Parameters
    ----------
    incident_ray_direction : torch.Tensor
        The direction of the light.
    expected_value : str
        The expected bitmaps for the given test-cases.
    scenario_config : str
        The name of the scenario config that should be loaded.

    Returns
    -------
    Dict[str, torch.Tensor]
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
        heliostat = Heliostat.from_hdf5(
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
        (torch.tensor([0.0, -1.0, 0.0, 0.0]), "south.pt", "test_scenario"),
        (torch.tensor([1.0, 0.0, 0.0, 0.0]), "east.pt", "test_scenario"),
        (torch.tensor([-1.0, 0.0, 0.0, 0.0]), "west.pt", "test_scenario"),
        (torch.tensor([0.0, 0.0, 1.0, 0.0]), "above.pt", "test_scenario"),
    ],
    name="environment_data",
)
def data(request):
    """
    Compute the data required for the test.

    Parameters
    ----------
    request : Tuple[torch.Tensor, str, str]
        The pytest.fixture request with the incident ray direction and bitmap name required for the test.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the data required for the test.
    """
    return generate_data(*request.param)


def test_nurbs(environment_data: Dict[str, torch.Tensor]):
    torch.manual_seed(7)
    sun = environment_data["sun"]
    receiver_center = environment_data["receiver_center"]
    incident_ray_direction = environment_data["incident_ray_direction"]
    expected_value = environment_data["expected_value"]
    aligned_surface_points = environment_data["aligned_surface_points"]
    aligned_surface_normals = environment_data["aligned_surface_normals"]

    receiver_plane_normal = torch.tensor([0.0, 1.0, 0.0, 0.0])
    receiver_plane_x = 8.629666667
    receiver_plane_y = 7.0
    receiver_resolution_x = 256
    receiver_resolution_y = 256

    degree_x = 2
    degree_y = 2
    num_control_points_x = 7
    num_control_points_y = 7

    origin = torch.tensor([0.0, 5.0, 0.0]) # heliostat position in field                                                                                    
    
    control_points_shape = (num_control_points_x, num_control_points_y)                       
    control_points = torch.zeros(
        control_points_shape + (3,),                                                  
    )

    origin_offsets_x = torch.linspace(
        -receiver_plane_x / 2, receiver_plane_x / 2, num_control_points_x)
    origin_offsets_y = torch.linspace(
        -receiver_plane_y / 2, receiver_plane_y / 2, num_control_points_y)
    origin_offsets = torch.cartesian_prod(origin_offsets_x, origin_offsets_y)
    origin_offsets = torch.hstack((
        origin_offsets,
        torch.zeros((len(origin_offsets), 1)),
    ))
    control_points = torch.nn.parameter.Parameter((origin + origin_offsets).reshape(control_points.shape))

    # num_control_points_x = 401
    # num_control_points_y = 401
    # control_points_real = torch.reshape(aligned_surface_points[:160801], (num_control_points_x, num_control_points_y, 4))
    # control_points_real = control_points_real[:,:, :3]
    # control_points_shape = (num_control_points_x, num_control_points_y)      
    # control_points_real = torch.nn.parameter.Parameter(control_points_real)                
  
    knots_x = torch.zeros(num_control_points_x + degree_x + 1)                                                                                              
    num_knot_vals = len(knots_x[degree_x:-degree_x])
    knot_vals = torch.linspace(0, 1, num_knot_vals)
    knots_x[:degree_x] = 0
    knots_x[degree_x:-degree_x] = knot_vals
    knots_x[-degree_x:] = 1

    knots_y = torch.zeros(num_control_points_y + degree_y + 1)                                                                                        
    num_knot_vals = len(knots_y[degree_y:-degree_y])
    knot_vals = torch.linspace(0, 1, num_knot_vals)
    knots_y[:degree_y] = 0
    knots_y[degree_y:-degree_y] = knot_vals
    knots_y[-degree_y:] = 1

    surface = NURBSSurface(degree_x,
                           degree_y,
                           100,
                           100,)
    
    optimizer = torch.optim.Adam([control_points])
    
    input = (control_points, knots_x.unsqueeze(0), knots_y.unsqueeze(0))

    for epoch in range(10):
        points, normals = surface(input)
        print(points)
        print(normals)

        optimizer.zero_grad()

        normals = torch.cat((normals, torch.zeros(normals.shape[0], 1)), dim = 1)

        preferred_ray_directions = sun.get_preferred_reflection_direction(
            -incident_ray_direction, normals
        )

        distortions_n, distortions_u = sun.sample(preferred_ray_directions.shape[0])

        rays = sun.scatter_rays(
            preferred_ray_directions,
            distortions_n,
            distortions_u,
        )

        intersections = sun.line_plane_intersections(
            receiver_plane_normal, receiver_center, rays, points
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

        plt.imshow(total_bitmap.detach())
        plt.show()

        expected_path = (
            pathlib.Path(ARTIST_ROOT)
            / "tests/physics_objects/test_bitmaps_load_surface_stral"
            / expected_value
        )

        expected = torch.load(expected_path)

        plt.imshow(expected.detach())
        plt.show()

        loss = total_bitmap - expected
        loss.abs().sum().backward()

        optimizer.step()

        print(loss.abs().sum())

        #loss = (output-control_points_real).sum()
        #print(loss)
        
