"""This pytest considers loading a heliostat surface from a point cloud."""

import pathlib
from typing import Dict

import h5py
import pytest
import torch
from torch.utils.data import DataLoader

from artist import ARTIST_ROOT, Scenario
from artist.raytracing.heliostat_tracing import DistortionsDataset, HeliostatRayTracer


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
        scenario = Scenario.load_scenario_from_hdf5(scenario_file=config_h5)

    receiver = scenario.receiver
    sun = scenario.light_source
    heliostat = scenario.heliostats.heliostat_list[0]

    return {
        "sun": sun,
        "heliostats": heliostat,
        "receiver": receiver,
        "incident_ray_direction": incident_ray_direction,
        "expected_value": expected_value,
    }


@pytest.fixture(
    params=[
        (
            torch.tensor([0.0, -1.0, 0.0, 0.0]),
            "south.pt",
            "test_scenario",
        ),
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


def test_compute_bitmaps(environment_data: Dict[str, torch.Tensor]) -> None:
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
    heliostat = environment_data["heliostats"]
    receiver = environment_data["receiver"]
    incident_ray_direction = environment_data["incident_ray_direction"]
    expected_value = environment_data["expected_value"]

    heliostat.set_aligned_surface(incident_ray_direction=incident_ray_direction)
    heliostat.set_preferred_reflection_direction(rays=-incident_ray_direction)

    # Currently heliostat raytracing is only possible for heliostats with the same number of surface points/normals
    number_of_surface_points = heliostat.preferred_reflection_direction.size(0)

    # Create data set
    distortions_dataset = DistortionsDataset(
        light_source=sun,
        number_of_points=number_of_surface_points,
        random_seed=7,
    )

    # Create dataloader
    distortions_loader = DataLoader(distortions_dataset, batch_size=5, shuffle=False)

    # Create raytracer
    raytracer = HeliostatRayTracer()

    final_bitmap = torch.zeros((receiver.resolution_x, receiver.resolution_y))
    for batch_u, batch_e in distortions_loader:
        rays = raytracer.scatter_rays(
            heliostat.preferred_reflection_direction,
            batch_u,
            batch_e,
        )

        intersections = raytracer.line_plane_intersections(
            receiver.plane_normal,
            receiver.center,
            rays,
            heliostat.current_aligned_surface_points,
        )

        dx_ints = intersections[:, :, 0] + receiver.plane_x / 2 - receiver.center[0]
        dy_ints = intersections[:, :, 2] + receiver.plane_y / 2 - receiver.center[2]

        indices = (
            (-1 <= dx_ints)
            & (dx_ints < receiver.plane_x + 1)
            & (-1 <= dy_ints)
            & (dy_ints < receiver.plane_y + 1)
        )

        total_bitmap = raytracer.sample_bitmap(
            dx_ints,
            dy_ints,
            indices,
            receiver.plane_x,
            receiver.plane_y,
            receiver.resolution_x,
            receiver.resolution_y,
        )

        final_bitmap += total_bitmap

    final_bitmap = raytracer.normalize_bitmap(
        final_bitmap,
        distortions_dataset.distortions_u.numel(),
        receiver.plane_x,
        receiver.plane_y,
    )

    expected_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/field/test_bitmaps_load_surface_stral"
        / expected_value
    )

    expected = torch.load(expected_path)

    torch.testing.assert_close(final_bitmap.T, expected, atol=5e-4, rtol=5e-4)
