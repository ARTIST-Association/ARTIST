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
scenario_path = pathlib.Path("please/insert/the/path/to/the/scenario/here/scenario.h5")

# Also specify the heliostats to be calibrated and the paths to your calibration-properties.json files.
# Please use the following style: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
heliostat_data_mapping = [
    (
        "heliostat_name_1",
        [
            pathlib.Path(
                "please/insert/the/path/to/the/paint/data/here/calibration-properties.json"
            ),
            # ....
        ],
        [
            pathlib.Path("please/insert/the/path/to/the/paint/data/here/flux.png"),
            # ....
        ],
    ),
    (
        "heliostat_name_2",
        [
            pathlib.Path(
                "please/insert/the/path/to/the/paint/data/here/calibration-properties.json"
            ),
            # ....
        ],
        [
            pathlib.Path("please/insert/the/path/to/the/paint/data/here/flux.png"),
            # ....
        ],
    ),
    # ...
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
    scenario.light_sources.light_source_list[0].number_of_rays = 200
    tolerance = 0.00005
    max_epoch = 4000
    initial_learning_rate = 1e-6
    number_of_surface_points = torch.tensor([80, 80], device=device)
    resolution = torch.tensor([256, 256], device=device)

    # Create the surface reconstructor.
    surface_reconstructor = SurfaceReconstructor(
        ddp_setup=ddp_setup,
        scenario=scenario,
        heliostat_data_mapping=heliostat_data_mapping,
        number_of_surface_points=number_of_surface_points,
        resolution=resolution,
        initial_learning_rate=initial_learning_rate,
        tolerance=tolerance,
        max_epoch=max_epoch,
        num_log=10,
        device=device,
    )

    # Reconstruct surfaces.
    surface_reconstructor.reconstruct_surfaces(device=device)
