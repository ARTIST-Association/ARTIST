
import pathlib
from artist import ARTIST_ROOT
from artist.scenario import Scenario
import h5py
from scripts import utils
import torch


if __name__ == "__main__":
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # The distributed environment is setup and destroyed using a Generator object.
    environment_generator = utils.setup_distributed_environment(device=device)
    
    is_distributed, rank, world_size = next(environment_generator)

    if device.type == "cuda":
        torch.cuda.set_device(rank % torch.cuda.device_count())
    
    scenario_path = pathlib.Path(ARTIST_ROOT) / "tests/data/four_heliostat_scenario.h5"
    
    # Load the scenario.
    with h5py.File(scenario_path, "r") as config_h5:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=config_h5, device=device
        )
        
    calibration_properties_paths = []
    calibration_properties_path = (
        pathlib.Path(ARTIST_ROOT) / "tests/data/calibration_properties.json"
    )