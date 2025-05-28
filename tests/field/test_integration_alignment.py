"""Tests loading a heliostat and performing ray tracing."""

import pathlib

import h5py
import pytest
import torch

from artist import ARTIST_ROOT
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util.scenario import Scenario


@pytest.mark.parametrize(
    "mapping, scenario_config",
    [
        (
            [
                ("heliostat_1", "receiver", torch.tensor([0.0, 1.0, 0.0, 0.0])),
                ("heliostat_1", "receiver", torch.tensor([-1.0, 0.0, 0.0, 0.0])),
                ("heliostat_1", "receiver", torch.tensor([1.0, 0.0, 0.0, 0.0])),
                ("heliostat_1", "receiver", torch.tensor([0.0, 0.0, -1.0, 0.0])),
            ],
            "test_scenario_stral_prototypes",
        ),
        (
            [
                ("heliostat_1", "receiver", torch.tensor([0.0, 1.0, 0.0, 0.0])),
            ],
            "test_scenario_stral_individual_measurements",
        ),
        (
            [
                ("AA39", "receiver", torch.tensor([0.0, 1.0, 0.0, 0.0])),
                ("AA39", "receiver", torch.tensor([-1.0, 0.0, 0.0, 0.0])),
                ("AA39", "receiver", torch.tensor([1.0, 0.0, 0.0, 0.0])),
                ("AA39", "receiver", torch.tensor([0.0, 0.0, -1.0, 0.0])),
                ("AA39", "multi_focus_tower", torch.tensor([0.0, 1.0, 0.0, 0.0])),
                (
                    "AA39",
                    "solar_tower_juelich_lower",
                    torch.tensor([0.0, 1.0, 0.0, 0.0]),
                ),
                (
                    "AA39",
                    "solar_tower_juelich_upper",
                    torch.tensor([0.0, 1.0, 0.0, 0.0]),
                ),
            ],
            "test_scenario_paint_single_heliostat",
        ),
    ],
)
def test_integration_alignment(
    mapping: list[tuple[str, str, torch.Tensor]],
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
    mapping : list[tuple[str, str, torch.Tensor]]
        The mapping from heliostat to target area to incident ray direction.
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

    bitmap_resolution_e, bitmap_resolution_u = 256, 256
    flux_distributions = torch.zeros(
        (
            scenario.target_areas.number_of_target_areas,
            bitmap_resolution_e,
            bitmap_resolution_u,
        ),
        device=device,
    )

    for heliostat_group in scenario.heliostat_field.heliostat_groups:
        (
            active_heliostats_mask,
            target_area_mask,
            incident_ray_directions,
        ) = scenario.index_mapping(
            heliostat_group=heliostat_group,
            string_mapping=mapping,
            device=device,
        )

        # Activate heliostats
        heliostat_group.activate_heliostats(
            active_heliostats_mask=active_heliostats_mask
        )

        # Align heliostats.
        heliostat_group.align_surfaces_with_incident_ray_directions(
            aim_points=scenario.target_areas.centers[target_area_mask],
            incident_ray_directions=incident_ray_directions,
            device=device,
        )

        # Create a ray tracer.
        ray_tracer = HeliostatRayTracer(
            scenario=scenario,
            heliostat_group=heliostat_group,
            bitmap_resolution_e=bitmap_resolution_e,
            bitmap_resolution_u=bitmap_resolution_u,
            batch_size=10,
        )

        # Perform heliostat-based ray tracing.
        group_bitmaps = ray_tracer.trace_rays(
            incident_ray_directions=incident_ray_directions,
            target_area_mask=target_area_mask,
            device=device,
        )

        group_bitmaps_per_target = torch.zeros(
            (
                scenario.target_areas.number_of_target_areas,
                bitmap_resolution_e,
                bitmap_resolution_u,
            ),
            device=device,
        )
        for index in range(scenario.target_areas.number_of_target_areas):
            mask = target_area_mask == index
            if mask.any():
                group_bitmaps_per_target[index] = group_bitmaps[mask].sum(dim=0)

        flux_distributions = flux_distributions + group_bitmaps_per_target

    expected_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/data/expected_bitmaps_integration"
        / f"{scenario_config}_{device.type}.pt"
    )

    expected = torch.load(expected_path, map_location=device, weights_only=True)
    torch.testing.assert_close(flux_distributions, expected, atol=5e-4, rtol=5e-4)
