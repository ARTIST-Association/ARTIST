"""Tests loading a heliostat and performing ray tracing."""

import pathlib

import h5py
import pytest
import torch

from artist import ARTIST_ROOT
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util.scenario import Scenario


@pytest.mark.parametrize(
    "incident_ray_directions, ray_direction_string, scenario_config",
    [
        (
            torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
            "stral_south",
            "test_scenario_stral_prototypes",
        ),
        (
            torch.tensor([[-1.0, 0.0, 0.0, 0.0]]),
            "stral_east",
            "test_scenario_stral_prototypes",
        ),
        (
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "stral_west",
            "test_scenario_stral_prototypes",
        ),
        (
            torch.tensor([[0.0, 0.0, -1.0, 0.0]]),
            "stral_above",
            "test_scenario_stral_prototypes",
        ),
        (
            torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
            "individual_south",
            "test_scenario_stral_individual_measurements",
        ),
        (
            torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
            "paint_south",
            "test_scenario_paint_single_heliostat",
        ),
    ],
)
def test_integration_alignment(
    incident_ray_directions: torch.Tensor,
    ray_direction_string: str,
    scenario_config: str,
    device: torch.device,
) -> None:
    """
    Align heliostats from different scenarios using the kinematic module to test the alignment process.

    With the aligned surface and the light direction, reflect the rays at every normal on the heliostat surface to
    calculate the preferred reflection direction. Then perform heliostat based ray tracing.
    This uses distortions based on the model of the sun to generate additional rays, calculates the intersections
    on the receiver, and computes the bitmap.

    Parameters
    ----------
    incident_ray_directions : torch.Tensor
        The incident ray directions used for the test.
    ray_direction_string : str
        String value describing the ray direction.
    scenario_config : str
        The name of the scenario to be loaded.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    # Load the scenario.
    with h5py.File(
        pathlib.Path(ARTIST_ROOT) / "tests/data/scenarios" / f"{scenario_config}.h5",
        "r",
    ) as config_h5:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=config_h5, device=device
        )

    # Align heliostat.
    scenario.heliostat_field.heliostat_groups[0].align_surfaces_with_incident_ray_directions(
        incident_ray_directions=incident_ray_directions.to(device),
        active_heliostats_indices=torch.tensor([0], device=device),
        device=device
    )

    # Create a ray tracer.
    ray_tracer = HeliostatRayTracer(
        scenario=scenario,
        heliostat_group=scenario.heliostat_field.heliostat_groups[0]
    )

    # Perform heliostat-based ray tracing.
    final_bitmaps = ray_tracer.trace_rays(
        incident_ray_directions=incident_ray_directions.to(device),
        active_heliostats_indices=torch.tensor([0], device=device),
        target_area_indices=torch.tensor([scenario.target_areas.names.index("receiver")], device=device),
        device=device,
    )

    expected_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/data/expected_bitmaps_integration"
        / f"{ray_direction_string}_{device.type}.pt"
    )
    expected = torch.load(expected_path, map_location=device, weights_only=True)
    torch.testing.assert_close(final_bitmaps[0], expected, atol=5e-4, rtol=5e-4)
