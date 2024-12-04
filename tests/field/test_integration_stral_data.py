"""Tests loading a heliostat surface from STRAL data and performing ray tracing."""

import pathlib

import h5py
import pytest
import torch

from artist import ARTIST_ROOT
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.scenario import Scenario


@pytest.fixture(params=["cpu", "cuda:1"] if torch.cuda.is_available() else ["cpu"])
def device(request: pytest.FixtureRequest) -> torch.device:
    """
    Return the device on which to initialize tensors.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The pytest fixture used to consider different test cases.

    Returns
    -------
    torch.device
        The device on which to initialize tensors.
    """
    return torch.device(request.param)


@pytest.mark.parametrize(
    "incident_ray_direction, expected_value, scenario_config",
    [
        (torch.tensor([0.0, -1.0, 0.0, 0.0]), "south", "test_scenario_single_prototype_heliostat_stral"),
        (torch.tensor([1.0, 0.0, 0.0, 0.0]), "east", "test_scenario_single_prototype_heliostat_stral"),
        (torch.tensor([-1.0, 0.0, 0.0, 0.0]), "west", "test_scenario_single_prototype_heliostat_stral"),
        (torch.tensor([0.0, 0.0, 1.0, 0.0]), "above", "test_scenario_single_prototype_heliostat_stral"),
        (
            torch.tensor([0.0, -1.0, 0.0, 0.0]),
            "individual_south",
            "test_scenario_single_individual_heliostat_stral",
        ),  # Test if loading with individual measurements works
    ],
)
def test_compute_bitmaps(
    incident_ray_direction: torch.Tensor,
    expected_value: str,
    scenario_config: str,
    device: torch.device,
) -> None:
    """
    Compute the resulting flux density distribution (bitmap) for the given test case.

    With the aligned surface and the light direction, reflect the rays at every normal on the heliostat surface to
    calculate the preferred reflection direction.
    Then perform heliostat based raytracing. This uses distortions based on the model of the sun to generate additional
    rays, calculates the intersections on the receiver, and computes the bitmap.
    Then normalize the bitmaps and compare them with the expected value.

    Parameters
    ----------
    incident_ray_direction : torch.Tensor
        The incident ray direction used for the test.
    expected_value : str
        The path to the expected value bitmap.
    scenario_config : str
        The name of the scenario to be loaded.
    device : torch.device
        The device on which to initialize tensors.
    distributed_environment : tuple[bool, int, int]
        Fixture to setup and destroy the process group before and after each test.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    # Load the scenario.
    with h5py.File(
        pathlib.Path(ARTIST_ROOT) / "tests/data" / f"{scenario_config}.h5", "r"
    ) as config_h5:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=config_h5, device=device
        )

    # Align heliostat.
    scenario.heliostats.heliostat_list[
        0
    ].set_aligned_surface_with_incident_ray_direction(
        incident_ray_direction=incident_ray_direction.to(device), device=device
    )

    # Create raytracer
    raytracer = HeliostatRayTracer(
        scenario=scenario
    )

    # Perform heliostat-based raytracing.
    final_bitmap = raytracer.trace_rays(
        incident_ray_direction=incident_ray_direction.to(device), device=device
    )

    final_bitmap = raytracer.normalize_bitmap(final_bitmap)

    expected_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/data/expected_bitmaps_integration"
        / f"{expected_value}_{device.type}.pt"
    )
    expected = torch.load(expected_path, map_location=device, weights_only=True)
    torch.testing.assert_close(final_bitmap.T, expected, atol=5e-4, rtol=5e-4)
