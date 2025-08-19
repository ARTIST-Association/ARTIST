import logging
import pathlib

import h5py
import torch

from artist.core.surface_reconstructor import SurfaceReconstructor
from artist.data_loader.paint_loader import build_heliostat_data_mapping
from artist.scenario.scenario import Scenario
from artist.util import set_logger_config
from artist.util.environment_setup import get_device, setup_distributed_environment

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set up logger
set_logger_config()
log = logging.getLogger(__name__)
# Set the device
device = get_device()

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path("please/insert/the/path/to/the/scenario/here/name")

# Also specify the heliostats to be calibrated and the paths to your calibration-properties.json files.
# Please use the following style: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
# heliostat_data_mapping = [
#     (
#         "heliostat_name_1",
#         [
#             pathlib.Path(
#                 "please/insert/the/path/to/the/paint/data/here/calibration-properties.json"
#             ),
#             # ....
#         ],
#         [
#             pathlib.Path("please/insert/the/path/to/the/paint/data/here/flux.png"),
#             # ....
#         ],
#     ),
#     (
#         "heliostat_name_2",
#         [
#             pathlib.Path(
#                 "please/insert/the/path/to/the/paint/data/here/calibration-properties.json"
#             ),
#             # ....
#         ],
#         [
#             pathlib.Path("please/insert/the/path/to/the/paint/data/here/flux.png"),
#             # ....
#         ],
#     ),
#     # ...
# ]


heliostat_data_mapping = build_heliostat_data_mapping(
    base_path="/base/path/to/PAINT",
    heliostat_names=["AA39"],
    number_of_measurements=4,
    image_variant="flux-centered",
)


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
    # Load the scenario.
    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file, device=device
        )

    for heliostat_group_index in groups_to_ranks_mapping[rank]:
        # If there are more ranks than heliostat groups, some ranks will be left idle.
        if rank < scenario.heliostat_field.number_of_heliostat_groups:
            heliostat_group = scenario.heliostat_field.heliostat_groups[
                heliostat_group_index
            ]

            # Set parameters.
            scenario.light_sources.light_source_list[0].number_of_rays = 200
            tolerance = 0.00005
            max_epoch = 4000
            initial_learning_rate = 1e-6
            number_of_surface_points = torch.tensor([80, 80], device=device)
            resolution = torch.tensor([256, 256], device=device)

            # Create the surface reconstructor.
            surface_reconstructor = SurfaceReconstructor(
                scenario=scenario,
                heliostat_group=heliostat_group,
                heliostat_data_mapping=heliostat_data_mapping,
                number_of_surface_points=number_of_surface_points,
                resolution=resolution,
                initial_learning_rate=initial_learning_rate,
                tolerance=tolerance,
                max_epoch=max_epoch,
                num_log=max_epoch,
                device=device,
            )

            surface_reconstructor.reconstruct_surfaces(device=device)
