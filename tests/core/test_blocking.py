import os
import pathlib

import h5py
import torch
from matplotlib import pyplot as plt

from artist import ARTIST_ROOT
from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, set_logger_config
from artist.util.environment_setup import get_device, setup_distributed_environment


def test_blocking(device: torch.device):  
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    # Load the scenario.
    with h5py.File(
        pathlib.Path(ARTIST_ROOT) / "tests/data/scenarios/test_blocking.h5", "r",
    ) as scenario_file:

        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file,
            device=device,
        )

    incident_ray_direction = torch.nn.functional.normalize(torch.tensor([0.0, 1.0, 0.0, 0.0], device=device), dim=-1)

    heliostat_group = scenario.heliostat_field.heliostat_groups[0]
    heliostat_target_light_source_mapping = [
        ("heliostat_0", "target_0", incident_ray_direction),
        ("heliostat_1", "target_0", incident_ray_direction),
        ("heliostat_2", "target_0", incident_ray_direction),
        ("heliostat_3", "target_0", incident_ray_direction),
        ("heliostat_4", "target_0", incident_ray_direction),
        ("heliostat_5", "target_0", incident_ray_direction),
    ]

    (
        active_heliostats_mask,
        target_area_mask,
        incident_ray_directions,
    ) = scenario.index_mapping(
        heliostat_group=heliostat_group,
        string_mapping=heliostat_target_light_source_mapping,
        device=device,
    )

    heliostat_group.activate_heliostats(
        active_heliostats_mask=active_heliostats_mask, device=device
    )

    heliostat_group.align_surfaces_with_incident_ray_directions(
        aim_points=scenario.target_areas.centers[target_area_mask],
        incident_ray_directions=incident_ray_directions,
        active_heliostats_mask=active_heliostats_mask,
        device=device,
    )

    scenario.set_number_of_rays(number_of_rays=200)
    
    ray_tracer = HeliostatRayTracer(
        scenario=scenario,
        heliostat_group=heliostat_group,
        batch_size=10,
    )

    bitmaps_per_heliostat = ray_tracer.trace_rays(
        incident_ray_directions=incident_ray_directions,
        active_heliostats_mask=active_heliostats_mask,
        target_area_mask=target_area_mask,
        device=device,
    )

    expected_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/data/expected_bitmaps_blocking"
        / f"bitmaps_{device.type}.pt"
    )

    expected = torch.load(expected_path, map_location=device, weights_only=True)

    torch.testing.assert_close(bitmaps_per_heliostat, expected, atol=5e-4, rtol=5e-4)



