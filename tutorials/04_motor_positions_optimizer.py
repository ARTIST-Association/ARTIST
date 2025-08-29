import pathlib

import h5py
import torch

from artist.core import loss_functions_old
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
    device = ddp_setup[config_dictionary.device]

    # Load the scenario.
    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file,
            device=device,
        )

    scenario.light_sources.light_source_list[0].number_of_rays = 4

    # Set loss function and define the optimization "target" (which is called optimization goal
    # to prevent confusion with the tower "target" areas).
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
    loss_function = loss_functions_old.distribution_loss_kl_divergence

    # Configure the learning rate scheduler. The example scheduler parameter dict includes
    # example parameters for all three possible schedulers.
    scheduler = (
        config_dictionary.exponential
    )  # Choose from: exponential, cyclic or reduce_on_plateau
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
        config_dictionary.initial_learning_rate: 1e-3,
        config_dictionary.tolerance: 0.0005,
        config_dictionary.max_epoch: 50,
        config_dictionary.num_log: 10,
        config_dictionary.early_stopping_delta: 1e-4,
        config_dictionary.early_stopping_patience: 10,
        config_dictionary.scheduler: scheduler,
        config_dictionary.scheduler_parameters: scheduler_parameters,
    }

    # Create the motor positions optimizer.
    motor_positions_optimizer = MotorPositionsOptimizer(
        ddp_setup=ddp_setup,
        scenario=scenario,
        optimization_configuration=optimization_configuration,
        incident_ray_direction=torch.tensor([0.0, 1.0, 0.0, 0.0], device=device),
        target_area_index=1,
        optimization_goal=optimization_goal,
        bitmap_resolution=torch.tensor([256, 256], device=device),
        device=device,
    )

    # Optimize the motor positions.
    _ = motor_positions_optimizer.optimize(loss_function=loss_function, device=device)
