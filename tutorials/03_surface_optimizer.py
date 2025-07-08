import pathlib

import h5py
import torch

from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.scenario.scenario import Scenario
from artist.util import set_logger_config
from artist.util.environment_setup import get_device, setup_distributed_environment

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set up logger
set_logger_config()

# Set device type.
device = get_device()

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path(
    "/workVERLEIHNIX/mb/ARTIST/tutorials/data/scenarios/test_scenario_paint_multiple_heliostat_groups.h5"
)

# Set the number of heliostat groups, this is needed for process group assignment.
number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
    scenario_path=scenario_path
)

with setup_distributed_environment(
    number_of_heliostat_groups=number_of_heliostat_groups,
    device=device,
) as (
    device,
    is_distributed,
    is_nested,
    rank,
    world_size,
    process_subgroup,
    groups_to_ranks_mapping,
    heliostat_group_rank,
    heliostat_group_world_size,
):
    # # Load the scenario.
    # with h5py.File(scenario_path) as scenario_file:
    #     scenario = Scenario.load_scenario_from_hdf5(
    #         scenario_file=scenario_file, device=device
    #     )

    scenario = Scenario(
        
    )
    
    for group_index in groups_to_ranks_mapping[rank]:
        heliostat_group = scenario.heliostat_field.heliostat_groups[group_index]



    