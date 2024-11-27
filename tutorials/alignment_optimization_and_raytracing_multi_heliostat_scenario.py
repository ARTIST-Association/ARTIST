import pathlib

import h5py
import torch

from artist import ARTIST_ROOT
from artist.scenario import Scenario
from artist.util import set_logger_config, utils

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set up logger
set_logger_config()

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The distributed environment is setup and destroyed using a Generator object.
environment_generator = utils.setup_distributed_environment(device=device)

is_distributed, rank, world_size = next(environment_generator)

if device.type == "cuda":
    torch.cuda.set_device(rank % torch.cuda.device_count())

# If you have already generated the tutorial scenario yourself, you can leave this boolean as False. If not, set it to
# true and a pre-generated scenario file will be used for this tutorial!
use_pre_generated_scenario = True
scenario_path = (
    pathlib.Path(ARTIST_ROOT) / "please/insert/the/path/to/the/scenario/here/name.h5"
)
if use_pre_generated_scenario:
    scenario_path = (
        pathlib.Path(ARTIST_ROOT) / "tutorials/data/four_heliostat_scenario.h5"
    )

# Load the scenario.
with h5py.File(scenario_path, "r") as config_h5:
    scenario = Scenario.load_scenario_from_hdf5(scenario_file=config_h5, device=device)

# Choose calibration data for all heliostats
calibration_properties_paths = []
calibration_properties_path = (
    pathlib.Path(ARTIST_ROOT) / "tests/data/calibration_properties.json"
)
