import argparse
import json
import logging
import pathlib
import pickle
import random
import re
import warnings
from functools import partial

import h5py
import torch
import yaml
from mpi4py import MPI
from propulate import Propulator
from propulate.utils import get_default_propagator, set_logger_config

from artist.core import loss_functions
from artist.core.motor_position_optimizer import MotorPositionsOptimizer
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, utils
from artist.util.environment_setup import get_device

log = logging.getLogger(__name__)
"""A logger for the hyper parameter search."""


def motor_position_optimizer_for_hpo(
    params: dict[str, float],
    scenario_path: pathlib.Path,
) -> float:
    """
    Set up a motor position optimizer used in a hyperparameter search.

    Parameters
    ----------
    params : dict[str, float]
        Combination of reconstruction parameters.
    scenario_path : pathlib.Path
        Path to the surface reconstruction scenario.

    Returns
    -------
    float
        The loss for a specific parameter configuration.
    """
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    # Get device.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    device = get_device(device)

    # Set up ARTIST to run in single device mode.
    ddp_setup = {
        config_dictionary.device: device,
        config_dictionary.is_distributed: False,
        config_dictionary.is_nested: False,
        config_dictionary.rank: 0,
        config_dictionary.world_size: 1,
        config_dictionary.process_subgroup: None,
        config_dictionary.groups_to_ranks_mapping: {0: [0]},
        config_dictionary.heliostat_group_rank: 0,
        config_dictionary.heliostat_group_world_size: 1,
        config_dictionary.ranks_to_groups_mapping: {0: [0]},
    }

    # Load a scenario from an .h5 file.
    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file,
            change_number_of_control_points_per_facet=torch.tensor(
                [7, 7], device=device
            ),
            device=device,
        )

    # Set number of rays.
    scenario.set_number_of_rays(number_of_rays=5)

    scheduler = params["scheduler"]
    scheduler_parameters = {
        config_dictionary.min: params["min_learning_rate"],
        config_dictionary.max: params["max_learning_rate"],
        config_dictionary.step_size_up: params["step_size_up"],
        config_dictionary.reduce_factor: params["reduce_factor"],
        config_dictionary.patience: params["patience"],
        config_dictionary.threshold: params["threshold"],
        config_dictionary.cooldown: params["cooldown"],
        config_dictionary.gamma: params["gamma"],
    }

    # Set optimizer parameters.
    optimization_configuration = {
        config_dictionary.initial_learning_rate: params["initial_learning_rate"],
        config_dictionary.tolerance: 0.0005,
        config_dictionary.max_epoch: 100,
        config_dictionary.batch_size: 250,
        config_dictionary.log_step: 0,
        config_dictionary.early_stopping_delta: 1e-4,
        config_dictionary.early_stopping_patience: 15,
        config_dictionary.early_stopping_window: 10,
        config_dictionary.scheduler: scheduler,
        config_dictionary.scheduler_parameters: scheduler_parameters,
    }

    # Random, somewhere in the south-west.
    baseline_incident_ray_direction = torch.nn.functional.normalize(
        torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
        - torch.tensor([-0.411, -0.706, 0.576, 1.0], device=device),
        dim=0,
    )

    # Receiver.
    target_area_index = 1

    # Target distribution.
    e_trapezoid = utils.trapezoid_distribution(
        total_width=256, slope_width=30, plateau_width=180, device=device
    )
    u_trapezoid = utils.trapezoid_distribution(
        total_width=256, slope_width=30, plateau_width=180, device=device
    )
    eu_trapezoid = u_trapezoid.unsqueeze(1) * e_trapezoid.unsqueeze(0)

    target_distribution = eu_trapezoid / eu_trapezoid.sum()

    # Create the surface reconstructor.
    motor_positions_optimizer = MotorPositionsOptimizer(
        ddp_setup=ddp_setup,
        scenario=scenario,
        optimization_configuration=optimization_configuration,
        incident_ray_direction=baseline_incident_ray_direction,
        target_area_index=target_area_index,
        ground_truth=target_distribution,
        bitmap_resolution=torch.tensor([256, 256]),
        device=device,
    )
    loss = motor_positions_optimizer.optimize(
        loss_definition=loss_functions.KLDivergenceLoss(), device=device
    )

    return loss[torch.isfinite(loss)].mean().item()


if __name__ == "__main__":
    """
    Perform the hyperparameter search for the motor position optimization and save the results.

    This script executes the hyperparameter search with ``propulate`` and saves the result for
    further inspection.

    Parameters
    ----------
    config : str
        Path to the configuration file.
    device : str
        Device to use for the computation.
    data_dir : str
        Path to the data directory.
    results_dir : str
        Path to where the results will be saved.
    scenarios_dir : str
        Path to the directory containing the scenarios.
    propulate_logs_dir : str
        Path to the directory where propulate will write log messages.
    parameter_ranges_motor_positions : dict[str, int | float]
        The reconstruction parameters.
    """
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()

    # Set default location for configuration file.
    script_dir = pathlib.Path(__file__).resolve().parent
    default_config_path = script_dir / "config.yaml"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML configuration file.",
        default=default_config_path,
    )

    # Parse the config argument first to load the configuration.
    args, unknown = parser.parse_known_args()
    config_path = pathlib.Path(args.config)
    config = {}
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            warnings.warn(f"Error parsing YAML file: {exc}")
    else:
        warnings.warn(
            f"Warning: Configuration file not found at {config_path}. Using defaults."
        )

    # Add remaining arguments to the parser with defaults loaded from the config.
    data_dir_default = config.get("data_dir", "./paint_data")
    device_default = config.get("device", "cuda")
    scenarios_dir_default = config.get(
        "scenarios_dir", "./examples/hyperparameter_optimization/scenarios"
    )
    results_dir_default = config.get(
        "results_dir", "./examples/hyperparameter_optimization/results"
    )
    propulate_logs_dir_default = config.get(
        "propulate_logs_dir", "./examples/hyperparameter_optimization/logs"
    )
    parameter_ranges_default = config.get(
        "parameter_ranges_motor_positions",
        {
            "initial_learning_rate": [1e-7, 1e-3],
            "scheduler": ["exponential", "reduce_on_plateau", "cyclic"],
            "min_learning_rate": [1e-9, 1e-6],
            "max_learning_rate": [1e-4, 1e-2],
            "step_size_up": [100, 500],
            "reduce_factor": [0.05, 0.5],
            "patience": [3, 50],
            "threshold": [1e-6, 1e-3],
            "cooldown": [2, 20],
            "gamma": [0.85, 0.999],
        },
    )

    parser.add_argument(
        "--device",
        type=str,
        help="Device to use.",
        default=device_default,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to downloaded paint data.",
        default=data_dir_default,
    )
    parser.add_argument(
        "--scenarios_dir",
        type=str,
        help="Path to directory containing the generated scenarios.",
        default=scenarios_dir_default,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Path to save the results.",
        default=results_dir_default,
    )
    parser.add_argument(
        "--propulate_logs_dir",
        type=str,
        help="Path to save propulate log messages.",
        default=propulate_logs_dir_default,
    )
    parser.add_argument(
        "--parameter_ranges_motor_positions",
        type=eval,
        help="Parameters used for the reconstruction.",
        default=parameter_ranges_default,
    )

    # Re-parse the full set of arguments.
    args = parser.parse_args(args=unknown)

    device = get_device(torch.device(args.device))
    data_dir = pathlib.Path(args.data_dir)
    propulate_logs_dir = pathlib.Path(args.propulate_logs_dir) / "motor_positions"
    results_dir = pathlib.Path(args.results_dir)

    # Define scenario path.
    scenario_file = pathlib.Path(args.scenarios_dir) / "ideal_scenario_500.h5"
    if not scenario_file.exists():
        raise FileNotFoundError(
            f"The reconstruction scenario located at {scenario_file} could not be found! Please run the ``generate_scenarios.py`` to generate this scenario, or adjust the file path and try again."
        )

    # Set up separate logger for Propulate optimization.
    set_logger_config(
        level=logging.INFO,
        log_file=f"{propulate_logs_dir}/{pathlib.Path(__file__).stem}.log",
        log_to_stdout=False,
        log_rank=True,
        colors=True,
    )

    log = logging.getLogger(__name__)
    rank = comm.Get_rank()
    log.info(rank)

    seed = 7
    rng = random.Random(seed + comm.rank)

    reconstruction_parameter_ranges = {}
    for key, value in args.parameter_ranges_motor_positions.items():
        if all(isinstance(x, (int, float)) for x in value):
            if all(
                isinstance(x, int) or (isinstance(x, float) and x.is_integer())
                for x in value
            ):
                tuple_range = tuple(int(x) for x in value)
            else:
                tuple_range = tuple(float(x) for x in value)
        else:
            tuple_range = tuple(value)

        reconstruction_parameter_ranges[key] = tuple_range

    # Set up evolutionary operator.
    num_generations = 400
    pop_size = 2 * comm.size
    propagator = get_default_propagator(
        pop_size=pop_size,
        limits=reconstruction_parameter_ranges,
        crossover_prob=0.7,
        mutation_prob=0.4,
        random_init_prob=0.1,
        rng=rng,
    )

    loss_fn = partial(
        motor_position_optimizer_for_hpo,
        scenario_path=scenario_file,
    )

    # Set up propulator performing actual optimization.
    propulator = Propulator(
        loss_fn=loss_fn,
        propagator=propagator,
        rng=rng,
        island_comm=comm,
        generations=num_generations,
        checkpoint_path=propulate_logs_dir,
    )

    # Run optimization and print summary of results.
    propulator.propulate(
        logging_interval=1,
        debug=2,
    )
    propulator.summarize(
        top_n=10,
        debug=50,
    )

    hpo_result_file = propulate_logs_dir / "island_0_ckpt.pickle"
    optimized_parameters_file = results_dir / "hpo_results_motor_positions.json"

    # Save hpo results in format to be used by plots.
    if not hpo_result_file.exists():
        raise FileNotFoundError(
            f"The hpo results located at {hpo_result_file} could not be not found! Please run the hpo script again to generate the results."
        )

    with open(hpo_result_file, "rb") as results:
        data = pickle.load(results)

    data_dict = data[-1]
    parameters_dict = {}

    for key, value in data_dict.items():
        if (
            isinstance(value, str)
            and re.fullmatch(r"[+-]?\d+(\.\d+)?[eE][+-]?\d+", value) is not None
        ):
            parameters_dict[key] = float(value)
        else:
            parameters_dict[key] = value

    if not results_dir.parent.is_dir():
        results_dir.parent.mkdir(parents=True, exist_ok=True)

    with open(optimized_parameters_file, "w") as output_file:
        json.dump(parameters_dict, output_file, indent=2)
