import pathlib

import h5py
import torch

from artist.core.kinematic_optimizer import KinematicOptimizer
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, set_logger_config
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

    # Choose calibration method.
    kinematic_calibration_method = (
        config_dictionary.kinematic_calibration_raytracing
    )

    # Set optimizer parameters.
    tolerance = 0.0005
    max_epoch = 1000
    initial_learning_rate = 0.0005

    # Create the kinematic optimizer.
    kinematic_optimizer = KinematicOptimizer(
        ddp_setup=ddp_setup,
        scenario=scenario,
        heliostat_data_mapping=heliostat_data_mapping,
        calibration_method=kinematic_calibration_method,
        initial_learning_rate=initial_learning_rate,
        tolerance=tolerance,
        max_epoch=max_epoch,
        num_log=10,
        device=device,
    )

    # Calibrate the kinematic.
    kinematic_optimizer.optimize(device=device)
