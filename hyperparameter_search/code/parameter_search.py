import itertools
import pathlib
import time

import h5py
import torch

from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.scenario.scenario import Scenario
from artist.util import set_logger_config, utils
from artist.util.environment_setup import get_device, setup_distributed_environment
from artist.util.nurbs import NURBSSurfaces
from hyperparameter_search.code import helper
from hyperparameter_search.code.surface_reconstructor2 import SurfaceReconstructor2

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set up logger
set_logger_config()

# Set the device
device = get_device()

# Specify the path to your scenario.h5 file.
scenario_paths = [
    pathlib.Path(
        "/workVERLEIHNIX/mb/ARTIST/hyperparameter_search/ideal_cp_5.h5"
    ),
    pathlib.Path(
        "/workVERLEIHNIX/mb/ARTIST/hyperparameter_search/ideal_cp_6.h5"
    ),
    pathlib.Path(
        "/workVERLEIHNIX/mb/ARTIST/hyperparameter_search/ideal_cp_10.h5"
    ),
    pathlib.Path(
        "/workVERLEIHNIX/mb/ARTIST/hyperparameter_search/ideal_cp_20.h5"
    ),
    pathlib.Path(
        "/workVERLEIHNIX/mb/ARTIST/hyperparameter_search/ideal_cp_50.h5"
    ),
    pathlib.Path(
        "/workVERLEIHNIX/mb/ARTIST/hyperparameter_search/ideal_cp_100.h5"
    ),
]

# Also specify the heliostats to be calibrated and the paths to your measured flux density distributions.
# Please follow the following style: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
heliostat_data_mapping = [
    (
        "AA39",
        [
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/275564-calibration-properties.json"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/271633-calibration-properties.json"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/270398-calibration-properties.json"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/246955-calibration-properties.json"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/218385-calibration-properties.json"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/223788-calibration-properties.json"
            ),
        ],
        [
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/275564-flux-centered.png"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/271633-flux-centered.png"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/270398-flux-centered.png"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/246955-flux-centered.png"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/218385-flux-centered.png"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/223788-flux-centered.png"
            ),
        ],
    ),
]

number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
    scenario_path=scenario_paths[0]
)

with setup_distributed_environment(
    number_of_heliostat_groups=number_of_heliostat_groups,
    device=device,
) as ddp_setup:
    
    device = ddp_setup["device"]

    points_and_rays = [(torch.tensor([50, 50], device=device), 220),
                       (torch.tensor([120, 120], device=device), 50),
                       (torch.tensor([90, 90], device=device), 100),
                       (torch.tensor([100, 100], device=device), 90),
                       (torch.tensor([80, 80], device=device), 200),
                       (torch.tensor([150, 150], device=device), 57)]

    resolution = [
        torch.tensor([256, 256], device=device)
    ]

    number_of_measurements = [2, 4, 6]

    learning_rates = [1e-4, 1e-5, 1e-6]

    parameter_combinations = list(
        itertools.product(points_and_rays, resolution, scenario_paths, number_of_measurements, learning_rates)
    )

    keys = ["points_and_rays", "resolution", "scenario_paths", "number_of_measurements", "learning_rates"]
    parameter_combinations_dicts = [
        dict(zip(keys, values)) for values in parameter_combinations
    ]

    for parameter_combination in parameter_combinations_dicts:
        # Load the scenario.
        with h5py.File(parameter_combination["scenario_paths"], "r") as scenario_file:
            scenario = Scenario.load_scenario_from_hdf5(
                scenario_file=scenario_file, device=device,
            )

        scenario.light_sources.light_source_list[
            0
        ].number_of_rays = parameter_combination["points_and_rays"][1]

        # Set parameters.
        tolerance = 0.00005
        max_epoch = 3000
        initial_learning_rate = parameter_combination["learning_rates"]
        number_of_surface_points = parameter_combination["points_and_rays"][0]
        resolution = parameter_combination["resolution"]

        start_reconstruction = time.perf_counter()
        # Create the surface reconstructor.
        surface_reconstructor = SurfaceReconstructor2(
            scenario=scenario,
            heliostat_data_mapping=heliostat_data_mapping,
            number_of_surface_points=number_of_surface_points,
            resolution=resolution,
            initial_learning_rate=initial_learning_rate,
            tolerance=tolerance,
            max_epoch=max_epoch,
            num_log=max_epoch,
            device=device,
        )

        surface_reconstructor.reconstruct_surfaces(ddp_setup=ddp_setup, device=device)
        
        end_reconstruction = time.perf_counter()
        elapsed_ms = (end_reconstruction - start_reconstruction) * 1000 
