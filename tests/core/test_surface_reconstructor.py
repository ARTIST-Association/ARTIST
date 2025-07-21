import pathlib

import h5py
import torch

from artist import ARTIST_ROOT
from artist.core.surface_reconstructor import SurfaceReconstructor
from artist.scenario.scenario import Scenario


def test_surface_reconstructor(
    device: torch.device,
) -> None:
    """
    Test the surface reconstructor.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    scenario_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/data/scenarios/test_scenario_paint_four_heliostats.h5"
    )

    heliostat_data_mapping = [
        (
            "AA39",
            [
                pathlib.Path(ARTIST_ROOT)
                / "tests/data/field_data/AA39-calibration-properties_1.json",
                pathlib.Path(ARTIST_ROOT)
                / "tests/data/field_data/AA39-calibration-properties_2.json",
            ],
            [
                pathlib.Path(ARTIST_ROOT)
                / "tests/data/field_data/AA39-flux_centered_1.png",
                pathlib.Path(ARTIST_ROOT)
                / "tests/data/field_data/AA39-flux-centered_2.png",
            ],
        ),
        (
            "AA31",
            [
                pathlib.Path(ARTIST_ROOT)
                / "tests/data/field_data/AA31-calibration-properties_1.json"
            ],
            [
                pathlib.Path(ARTIST_ROOT)
                / "tests/data/field_data/AA31-flux-centered_1.png"
            ],
        ),
    ]

    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file, device=device
        )

    for index, heliostat_group in enumerate(scenario.heliostat_field.heliostat_groups):
        surface_reconstructor = SurfaceReconstructor(
            scenario=scenario,
            heliostat_group=heliostat_group,
            heliostat_data_mapping=heliostat_data_mapping,
            max_epoch=2,
            num_log=1,
            device=device,
        )

        surface_reconstructor.reconstruct_surfaces(device=device)

        expected_path = (
            pathlib.Path(ARTIST_ROOT)
            / "tests/data/expected_reconstructed_surfaces"
            / f"group_{index}_{device.type}.pt"
        )

        expected = torch.load(expected_path, map_location=device, weights_only=True)

        torch.testing.assert_close(
            heliostat_group.active_surface_points,
            expected["active_surface_points"],
            atol=5e-2,
            rtol=5e-2,
        )
        torch.testing.assert_close(
            heliostat_group.active_surface_normals,
            expected["active_surface_normals"],
            atol=5e-2,
            rtol=5e-2,
        )
