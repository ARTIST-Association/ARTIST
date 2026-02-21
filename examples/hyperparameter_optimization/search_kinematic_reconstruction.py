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
from artist.core.kinematic_reconstructor import KinematicReconstructor
from artist.data_parser.calibration_data_parser import CalibrationDataParser
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary
from artist.util.environment_setup import get_device

log = logging.getLogger(__name__)
"""A logger for the hyper parameter search."""


def kinematic_reconstructor_for_hpo(
    params: dict[str, float],
    scenario_path: pathlib.Path,
    heliostat_data_mapping: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
) -> float:
    """
    Set up a kinematic reconstructor used in a hyperparameter search.

    Parameters
    ----------
    params : dict[str, float]
        Combination of reconstruction parameters.
    scenario_path : pathlib.Path
        Path to the kinematic reconstruction scenario.
    heliostat_data_mapping : list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
        Data mapping from heliostat to calibration files used to reconstruct the kinematic.

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
    # The scenario .h5 file should contain a setup with at least one heliostat (with the same name(s)
    # as the heliostat(s) for which reconstruction data is provided).
    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file,
            number_of_surface_points_per_facet=torch.tensor([5, 5], device=device),
            device=device,
        )

    # Set number of rays.
    scenario.set_number_of_rays(number_of_rays=4)

    data_parser = PaintCalibrationDataParser(
        sample_limit=int(params["sample_limit"]),
        centroid_extraction_method="UTIS",
    )
    data: dict[
        str,
        CalibrationDataParser
        | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    ] = {
        config_dictionary.data_parser: data_parser,
        config_dictionary.heliostat_data_mapping: heliostat_data_mapping,
    }
    optimizer_dict = {
        config_dictionary.initial_learning_rate: params["initial_learning_rate"],
        config_dictionary.tolerance: 0.0005,
        config_dictionary.max_epoch: 100,
        config_dictionary.batch_size: 945,
        config_dictionary.log_step: 0,
        config_dictionary.early_stopping_delta: 1e-5,
        config_dictionary.early_stopping_patience: 8,
        config_dictionary.early_stopping_window: 10,
    }
    scheduler_dict = {
        config_dictionary.scheduler_type: params["scheduler"],
        config_dictionary.min: params["min_learning_rate"],
        config_dictionary.max: params["max_learning_rate"],
        config_dictionary.step_size_up: params["step_size_up"],
        config_dictionary.reduce_factor: params["reduce_factor"],
        config_dictionary.patience: params["patience"],
        config_dictionary.threshold: params["threshold"],
        config_dictionary.cooldown: params["cooldown"],
        config_dictionary.gamma: params["gamma"],
    }
    optimization_configuration = {
        config_dictionary.optimization: optimizer_dict,
        config_dictionary.scheduler: scheduler_dict,
    }

    # Create the kinematic reconstructor.
    kinematic_reconstructor = KinematicReconstructor(
        ddp_setup=ddp_setup,
        scenario=scenario,
        data=data,
        optimization_configuration=optimization_configuration,
        reconstruction_method=config_dictionary.kinematic_reconstruction_raytracing,
    )

    loss_definition = loss_functions.FocalSpotLoss(scenario=scenario)

    # Reconstruct the kinematic.
    final_loss_per_heliostat = kinematic_reconstructor.reconstruct_kinematic(
        loss_definition=loss_definition, device=device
    )

    return (
        final_loss_per_heliostat[torch.isfinite(final_loss_per_heliostat)].mean().item()
    )


if __name__ == "__main__":
    """
    Perform the hyperparameter search for the kinematic reconstruction and save the results.

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
    heliostat_for_reconstruction : dict[str, list[int]]
        The heliostat and its calibration numbers.
    results_dir : str
        Path to where the results will be saved.
    scenarios_dir : str
        Path to the directory containing the scenarios.
    propulate_logs_dir : str
        Path to the directory where propulate will write log messages.
    parameter_ranges_kinematic : dict[str, int | float]
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
        "parameter_ranges_kinematic",
        {
            "initial_learning_rate": [1e-9, 1e-2],
            "scheduler": ["exponential", "reduce_on_plateau", "cyclic"],
            "min_learning_rate": [1e-12, 1e-6],
            "max_learning_rate": [1e-4, 1e-2],
            "step_size_up": [100, 500],
            "reduce_factor": [0.05, 0.5],
            "patience": [3, 50],
            "threshold": [1e-6, 1e-2],
            "cooldown": [2, 20],
            "gamma": [0.85, 0.999],
            "sample_limit": [2, 6],
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
        "--parameter_ranges_kinematic",
        type=eval,
        help="Parameters used for the reconstruction.",
        default=parameter_ranges_default,
    )

    # Re-parse the full set of arguments.
    args = parser.parse_args(args=unknown)

    device = get_device(torch.device(args.device))
    data_dir = pathlib.Path(args.data_dir)
    propulate_logs_dir = pathlib.Path(args.propulate_logs_dir) / "kinematic"
    results_dir = pathlib.Path(args.results_dir)

    # Define scenario path.
    scenario_file = pathlib.Path(args.scenarios_dir) / "ideal_scenario_hpo.h5"
    if not scenario_file.exists():
        raise FileNotFoundError(
            f"The reconstruction scenario located at {scenario_file} could not be found! Please run the ``generate_scenario.py`` to generate this scenario, or adjust the file path and try again."
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

    viable_heliostats_data = (
        pathlib.Path(args.results_dir) / "viable_heliostats_hpo.json"
    )
    if not viable_heliostats_data.exists():
        raise FileNotFoundError(
            f"The viable heliostat list located at {viable_heliostats_data} could not be not found! Please run the ``generate_viable_heliostat_list.py`` script to generate this list, or adjust the file path and try again."
        )

    # Load viable heliostats data.
    with open(viable_heliostats_data, "r") as f:
        viable_heliostats = json.load(f)

    heliostat_data_mapping: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]] = [
        (
            item["name"],
            [pathlib.Path(p) for p in item["calibrations"]],
            [pathlib.Path(p) for p in item["kinematic_reconstruction_flux_images"]],
        )
        for item in viable_heliostats
    ]

    reconstruction_parameter_ranges: dict[
        str, tuple[int, ...] | tuple[float, ...] | tuple[str, ...]
    ] = {}

    for key, value in args.parameter_ranges_kinematic.items():
        if all(isinstance(x, (int, float)) for x in value):
            if all(
                isinstance(x, int) or (isinstance(x, float) and x.is_integer())
                for x in value
            ):
                int_tuple: tuple[int, ...] = tuple(int(x) for x in value)
                reconstruction_parameter_ranges[key] = int_tuple
            else:
                float_tuple: tuple[float, ...] = tuple(float(x) for x in value)
                reconstruction_parameter_ranges[key] = float_tuple
        else:
            str_tuple: tuple[str, ...] = tuple(value)
            reconstruction_parameter_ranges[key] = str_tuple

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
        kinematic_reconstructor_for_hpo,
        scenario_path=scenario_file,
        heliostat_data_mapping=heliostat_data_mapping,
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
        top_n=20,
        debug=2,
    )

    hpo_result_file = propulate_logs_dir / "island_0_ckpt.pickle"
    optimized_parameters_file = results_dir / "hpo_results_kinematic.json"

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
