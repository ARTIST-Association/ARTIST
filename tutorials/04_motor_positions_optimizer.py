import pathlib

import h5py
import torch

from artist.core import loss_functions
from artist.core.motor_position_optimizer import MotorPositionsOptimizer
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, set_logger_config, utils
from artist.util.environment_setup import get_device, setup_distributed_environment

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set up logger
set_logger_config()

# Set the device
device = get_device()

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path("please/insert/the/path/to/the/scenario/here/scenario.h5")

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

    # Choose motor position optimization method and set optimization goal.
    motor_position_optimization_method = config_dictionary.optimization_to_distribution

    scenario.light_sources.light_source_list[0].number_of_rays = 4

    # For an optimization using a focal spot as target use this loss function definition:
    # optimization_goal = torch.tensor(
    #     [1.1493, -0.5030, 57.0474, 1.0000], device=device
    # )
    # loss_function = loss_functions.focal_spot_loss

    # For an optimization using a distribution as target use this loss function definition:
    e_trapezoid = utils.trapezoid_distribution(
        total_width=256, slope_width=30, plateau_width=180, device=device
    )
    u_trapezoid = utils.trapezoid_distribution(
        total_width=256, slope_width=30, plateau_width=180, device=device
    )
    optimization_goal = u_trapezoid.unsqueeze(1) * e_trapezoid.unsqueeze(0)
    loss_function = loss_functions.distribution_loss_kl_divergence

    # Set optimizer paramteres.
    initial_learning_rate = 1e-3
    max_epoch = 20

    # Create the motor positions optimizer.
    motor_positions_optimizer = MotorPositionsOptimizer(
        ddp_setup=ddp_setup,
        scenario=scenario,
        incident_ray_direction=torch.tensor([0.0, 1.0, 0.0, 0.0], device=device),
        target_area_index=1,
        optimization_goal=optimization_goal,
        bitmap_resolution=torch.tensor([256, 256], device=device),
        initial_learning_rate=initial_learning_rate,
        max_epoch=max_epoch,
        num_log=max_epoch,
        device=device,
    )

    # Optimize the motor positions.
    motor_positions_optimizer.optimize(loss_function=loss_function, device=device)
