import datetime
import math
from typing import Dict

import pytest
import torch
import h5py

from artist import ARTIST_ROOT
from artist.field.nurbs_diff import NURBSSurface
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
    aligned_surface_points = environment_data["aligned_surface_points"]

    # build control point vector from aligned surface points
    # remove some elements
    #control_points = torch.reshape(aligned_surface_points[:160801], (401, -1, 4)).unsqueeze(0)
    
    aligned_surface_normals = environment_data["aligned_surface_normals"]

    degree_x = 3
    degree_y = 3
    num_control_points_x = 401
    num_control_points_y = 401

    next_degree_x = degree_x + 1                                                                # == Order
    next_degree_y = degree_y + 1                                                                # == Order
    assert num_control_points_x > degree_x, \
        f'need at least {next_degree_x} control points in x direction'                          # num_control_points > degree + 1       
    assert num_control_points_y > degree_y, \
        f'need at least {next_degree_y} control points in y direction'                          # num_control_points > degree + 1
    
    control_points_real = torch.reshape(aligned_surface_points[:160801], (num_control_points_x, num_control_points_y, 4))
    control_points_shape = (num_control_points_x, num_control_points_y)                         # grid of control points
    control_points = torch.empty(
        control_points_shape + (4,),                                                            # each control point is 3 dimensional
    )   
    knots_x = torch.zeros(num_control_points_x + next_degree_x)                                 # num_control_points + order == num_control_points + degree + 1                                                             # fill knot vector with full multiplicity knots ranging from 0 to 1
    knots_x[next_degree_x:-next_degree_x] = 0.5                                                 # Why full multiplicity knots in the middle?
    knots_x[-next_degree_x:] = 1

    knots_y = torch.zeros(num_control_points_y + next_degree_y)                                 # num_control_points + order == num_control_points + degree + 1                                                           # fill knot vector with full multiplicity knots ranging from 0 to 1
    knots_y[next_degree_y:-next_degree_y] = 0.5                                                 # Why full multiplicity knots in the middle?
    knots_y[-next_degree_y:] = 1

    surface = NURBSSurface(degree_x,
                           degree_y,
                           401,
                           401,
                           4)
    
    input = (control_points_real, knots_x.unsqueeze(0), knots_y.unsqueeze(0))
    output = surface(input)
    print(output)
    print(output.grad_fn)

    loss = (output-control_points_real).sum()
    print(loss)
    
