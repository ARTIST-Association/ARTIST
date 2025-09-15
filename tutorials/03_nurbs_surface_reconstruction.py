import logging
import pathlib

import h5py
import torch

from artist.core.loss_functions import KLDivergenceLoss
from artist.core.regularizers import IdealSurfaceRegularizer, TotalVariationRegularizer
from artist.core.surface_reconstructor import SurfaceReconstructor
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, set_logger_config
from artist.util.environment_setup import get_device, setup_distributed_environment

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set up logger
set_logger_config()
log = logging.getLogger(__name__)
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

# Create dict for the data source name and the heliostat_data_mapping.
data: dict[str, str | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]] = {
    config_dictionary.data_source: config_dictionary.paint,
    config_dictionary.heliostat_data_mapping: heliostat_data_mapping,
}

number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
    scenario_path=scenario_path
)

with setup_distributed_environment(
    number_of_heliostat_groups=number_of_heliostat_groups,
    device=device,
) as ddp_setup:
    device = ddp_setup[config_dictionary.device]

    # Load the scenario.
    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file, device=device
        )

    # Set loss function.
    loss_definition = KLDivergenceLoss()
    # Another possibility would be the pixel loss:
    # loss_definition = PixelLoss(scenario=scenario)

    # Configure regularizers and their weights.
    ideal_surface_regularizer = IdealSurfaceRegularizer(
        weight=0.5, reduction_dimensions=(1, 2, 3, 4)
    )
    total_variation_regularizer_points = TotalVariationRegularizer(
        weight=0.5,
        reduction_dimensions=(1,),
        surface=config_dictionary.surface_points,
        number_of_neighbors=64,
        sigma=1e-3,
    )
    total_variation_regularizer_normals = TotalVariationRegularizer(
        weight=0.5,
        reduction_dimensions=(1,),
        surface=config_dictionary.surface_points,
        number_of_neighbors=64,
        sigma=1e-3,
    )

    regularizers = [
        ideal_surface_regularizer,
        total_variation_regularizer_points,
        total_variation_regularizer_normals,
    ]

    # Configure the learning rate scheduler. The example scheduler parameter dict includes
    # example parameters for all three possible schedulers.
    scheduler = (
        config_dictionary.exponential
    )  # exponential, cyclic or reduce_on_plateau
    scheduler_parameters = {
        config_dictionary.gamma: 0.9,
        config_dictionary.min: 1e-6,
        config_dictionary.max: 1e-3,
        config_dictionary.step_size_up: 500,
        config_dictionary.reduce_factor: 0.3,
        config_dictionary.patience: 10,
        config_dictionary.threshold: 1e-3,
        config_dictionary.cooldown: 10,
    }

    # Set optimizer parameters.
    optimization_configuration = {
        config_dictionary.initial_learning_rate: 1e-4,
        config_dictionary.tolerance: 0.00005,
        config_dictionary.max_epoch: 500,
        config_dictionary.num_log: 50,
        config_dictionary.early_stopping_delta: 1e-4,
        config_dictionary.early_stopping_patience: 10,
        config_dictionary.scheduler: scheduler,
        config_dictionary.scheduler_parameters: scheduler_parameters,
        config_dictionary.regularizers: regularizers,
    }

    scenario.light_sources.light_source_list[0].number_of_rays = 20
    number_of_surface_points = torch.tensor([100, 100], device=device)
    resolution = torch.tensor([256, 256], device=device)

    # Create the surface reconstructor.
    surface_reconstructor = SurfaceReconstructor(
        ddp_setup=ddp_setup,
        scenario=scenario,
        data=data,
        optimization_configuration=optimization_configuration,
        number_of_surface_points=number_of_surface_points,
        bitmap_resolution=resolution,
        device=device,
    )

    # Reconstruct surfaces.
    _ = surface_reconstructor.reconstruct_surfaces(
        loss_definition=loss_definition, device=device
    )
