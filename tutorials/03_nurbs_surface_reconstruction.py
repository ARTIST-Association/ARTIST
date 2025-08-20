import pathlib

import h5py
import torch

from artist.core.surface_reconstructor import SurfaceReconstructor
from artist.scenario.scenario import Scenario
from artist.util import set_logger_config
from artist.util.environment_setup import get_device, setup_distributed_environment

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set up logger
set_logger_config()

# Set the device
device = get_device()

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path("/workVERLEIHNIX/mb/ARTIST/tutorials/data/scenarios/test_scenario_paint_multiple_heliostat_groups_deflectometry.h5")

# Also specify the heliostats to be calibrated and the paths to your calibration-properties.json files.
# Please use the following style: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
heliostat_data_mapping = [
    (
        "AA39",
        [
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/270398-calibration-properties.json"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/271633-calibration-properties.json"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/275564-calibration-properties.json"
            ),
        ],
        [
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/270398-flux-centered.png"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/271633-flux-centered.png"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/275564-flux-centered.png"
            ),
        ],
    ),
    # (
    #     "AA31",
    #     [
    #         pathlib.Path(
    #             "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA31/125284-calibration-properties.json"
    #         ),
    #         pathlib.Path(
    #             "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA31/126372-calibration-properties.json"
    #         ),
    #     ],
    #     [
    #         pathlib.Path(
    #             "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA31/125284-flux-centered.png"
    #         ),
    #         pathlib.Path(
    #             "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA31/126372-flux-centered.png"
    #         ),
    #     ],
    # ),
    (
        "AC43",
        [
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AC43/62900-calibration-properties.json"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AC43/72752-calibration-properties.json"
            ),
        ],
        [
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AC43/62900-flux-centered.png"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AC43/72752-flux-centered.png"
            ),
        ],
    ),
]

number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
    scenario_path=scenario_path
)

with setup_distributed_environment(
    number_of_heliostat_groups=number_of_heliostat_groups,
    device=device,
) as ddp_setup:
    device = ddp_setup["device"]

    # Load the scenario.
    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file, device=device
        )

    # Set optimizer parameters.
    scenario.light_sources.light_source_list[0].number_of_rays = 20
    tolerance = 0.00005
    max_epoch = 10
    initial_learning_rate = 1e-5
    number_of_surface_points = torch.tensor([100, 100], device=device)
    resolution = torch.tensor([256, 256], device=device)

    # Create the surface reconstructor.
    surface_reconstructor = SurfaceReconstructor(
        ddp_setup=ddp_setup,
        scenario=scenario,
        heliostat_data_mapping=heliostat_data_mapping,
        number_of_surface_points=number_of_surface_points,
        bitmap_resolution=resolution,
        initial_learning_rate=initial_learning_rate,
        tolerance=tolerance,
        max_epoch=max_epoch,
        num_log=max_epoch,
        device=device,
    )

    # Reconstruct surfaces.
    surface_reconstructor.reconstruct_surfaces(device=device)
