import pathlib

import h5py
import torch

from artist.core.kinematic_optimizer import KinematicOptimizer
from artist.core.motor_position_optimizer import MotorPositionsOptimizer
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, set_logger_config, utils
from artist.util.environment_setup import get_device, setup_distributed_environment

torch.manual_seed(7)
torch.cuda.manual_seed(7)
torch.autograd.set_detect_anomaly(True)

# Set up logger
set_logger_config()

# Set the device
device = get_device()

torch.autograd.set_detect_anomaly(True)

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path(
    "/workVERLEIHNIX/mb/ARTIST/tutorials/data/scenarios/test_scenario_paint_multiple_heliostat_groups_deflectometry.h5"
)

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
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/270398-flux.png"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/271633-flux.png"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/275564-flux.png"
            ),
        ],
    ),
    (
        "AA31",
        [
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA31/125284-calibration-properties.json"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA31/126372-calibration-properties.json"
            ),
        ],
        [
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA31/125284-flux.png"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA31/126372-flux.png"
            ),
        ],
    ),
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
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AC43/62900-flux.png"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AC43/72752-flux.png"
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
            scenario_file=scenario_file,
            device=device,
        )

    # Choose calibration method.
    kinematic_calibration_method = config_dictionary.kinematic_calibration_raytracing

    # Set optimizer parameters.
    tolerance = 0.0005
    max_epoch = 60
    initial_learning_rate = 1e-4

    # Choose motor position optimization method and set optimization goal.
    motor_position_optimization_method = config_dictionary.optimization_to_distribution

    scenario.light_sources.light_source_list[0].number_of_rays = 4

    if motor_position_optimization_method == config_dictionary.optimization_to_focal_spot:
        optimization_goal = torch.tensor(
            [[1.1493, -0.5030, 57.0474, 1.0000]], device=device
        )
    if motor_position_optimization_method == config_dictionary.optimization_to_distribution:
        e_trapezoid = utils.trapezoid_1d_distribution(
            total_width=256, slope_width=30, plateau_width=180, device=device
        )
        u_trapezoid = utils.trapezoid_1d_distribution(
            total_width=256, slope_width=30, plateau_width=180, device=device
        )
        optimization_goal = u_trapezoid.unsqueeze(1) * e_trapezoid.unsqueeze(0)

    # Create the motor positions optimizer.
    motor_positions_optimizer = MotorPositionsOptimizer(
        ddp_setup=ddp_setup,
        scenario=scenario,
        incident_ray_direction=torch.tensor([0.0, 1.0, 0.0, 0.0], device=device),
        target_area_index=1,
        method=config_dictionary.optimization_to_distribution,
        optimization_goal=optimization_goal,
        bitmap_resolution=torch.tensor([256, 256], device=device),
        initial_learning_rate=initial_learning_rate,
        max_epoch=max_epoch,
        num_log=max_epoch,
        device=device,
    )

    # Optimize the motor positions.
    motor_positions_optimizer.optimize(device=device)

# P || Q    Penalizes extra mass where target has none (avoids hallucinated bits).
# Q || P    Penalizes missing mass in the target regions.
# -> We should do kl-div for surface reconstruction as well!