import argparse
import json
import logging
import pathlib
import pickle
import random
import warnings
from functools import partial

import h5py
import torch
import yaml
from mpi4py import MPI
from propulate import Propulator
from propulate.utils import get_default_propagator, set_logger_config

from artist.core import loss_functions
from artist.core.regularizers import IdealSurfaceRegularizer
from artist.core.surface_reconstructor import SurfaceReconstructor
from artist.data_parser.calibration_data_parser import CalibrationDataParser
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary
from artist.util.environment_setup import get_device

log = logging.getLogger(__name__)
"""A logger for the hyper parameter search."""


def surface_reconstructor_for_hpo(
    params: dict[str, float],
    scenario_path: pathlib.Path,
    heliostat_data_mapping: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
) -> float:
    """
    Set up a surface reconstructor used in a hyperparameter search.

    Parameters
    ----------
    params : dict[str, float]
        Combination of reconstruction parameters.
    scenario_path : pathlib.Path
        Path to the surface reconstruction scenario.
    heliostat_data_mapping : list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
        Data mapping from heliostat to calibration files used to reconstruct the surfaces.

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

    # For parameter combinations with too many rays directly return a default loss,
    # to avoid running such combination as they cause "out of memory" errors.
    total_number_of_rays = (
        params["number_of_surface_points"] * 2 * 4 * params["number_of_rays"] * 4
    )
    if total_number_of_rays >= 1500000:
        loss = 987987
        return loss

    number_of_surface_points_per_facet = torch.tensor(
        [params["number_of_surface_points"], params["number_of_surface_points"]],
        device=device,
    )

    number_of_control_points_per_facet = torch.tensor(
        [params["number_of_control_points"], params["number_of_control_points"]],
        device=device,
    )

    # Load a scenario from an .h5 file.
    # The scenario .h5 file should contain a setup with at least one heliostat (with the same name(s)
    # as the heliostat(s) for which reconstruction data is provided). The heliostat(s) in this scenario
    # should be initialized with an ideal surface, do not provide deflectometry data!
    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file,
            number_of_surface_points_per_facet=number_of_surface_points_per_facet,
            change_number_of_control_points_per_facet=number_of_control_points_per_facet,
            device=device,
        )

    # Set number of rays.
    scenario.set_number_of_rays(number_of_rays=int(params["number_of_rays"]))

    # Set nurbs degree.
    for heliostat_group in scenario.heliostat_field.heliostat_groups:
        heliostat_group.nurbs_degrees = torch.tensor(
            [params["nurbs_degree"], params["nurbs_degree"]], device=device
        )

    data: dict[
        str,
        CalibrationDataParser
        | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    ] = {
        config_dictionary.data_parser: PaintCalibrationDataParser(),
        config_dictionary.heliostat_data_mapping: heliostat_data_mapping,
    }

    scheduler = config_dictionary.reduce_on_plateau
    scheduler_parameters = {
        config_dictionary.min: 1e-7,
        config_dictionary.reduce_factor: params["reduce_factor"],
        config_dictionary.patience: params["patience"],
        config_dictionary.threshold: params["threshold"],
        config_dictionary.cooldown: 2,
    }

    # Configure regularizers and their weights.
    ideal_surface_regularizer = IdealSurfaceRegularizer(
        weight=params["ideal_surface_loss_weight"], reduction_dimensions=(1, 2, 3)
    )

    regularizers = [
        ideal_surface_regularizer,
    ]

    # Set optimizer parameters.
    optimization_configuration = {
        config_dictionary.initial_learning_rate: params["initial_learning_rate"],
        config_dictionary.tolerance: 0.00005,
        config_dictionary.max_epoch: 4500,
        config_dictionary.log_step: 0,
        config_dictionary.early_stopping_delta: 1e-4,
        config_dictionary.early_stopping_patience: 5000,
        config_dictionary.scheduler: scheduler,
        config_dictionary.scheduler_parameters: scheduler_parameters,
        config_dictionary.regularizers: regularizers,
    }

    # Create the surface reconstructor.
    surface_reconstructor = SurfaceReconstructor(
        ddp_setup=ddp_setup,
        scenario=scenario,
        data=data,
        optimization_configuration=optimization_configuration,
        number_of_surface_points=number_of_surface_points_per_facet,
        bitmap_resolution=torch.tensor([256, 256], device=device),
        device=device,
    )

    # Define loss.
    loss_definition = loss_functions.KLDivergenceLoss()

    # Reconstruct surfaces.
    loss = surface_reconstructor.reconstruct_surfaces(
        loss_definition=loss_definition,
        device=device,
    )

    return loss[torch.isfinite(loss)].sum().item()


if __name__ == "__main__":
    """
    Perform the hyperparameter search and save the results.

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
    reconstruction_parameter_ranges : dict[str, int | float]
        The reconstruction parameters.
    """
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()

    # Set default location for configuration file.
    script_dir = pathlib.Path(__file__).resolve().parent
    default_config_path = script_dir / "hpo_config.yaml"

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
    heliostat_for_reconstruction_default = config.get(
        "heliostat_for_reconstruction", {"AA39": [244862, 270398, 246213, 258959]}
    )
    scenarios_dir_default = config.get("scenarios_dir", "./scenarios")
    results_dir_default = config.get("results_dir", "./results")
    propulate_logs_dir_default = config.get("propulate_logs_dir", "./logs")
    reconstruction_parameter_ranges_default = config.get(
        "reconstruction_parameter_ranges",
        {
            "number_of_surface_points": [30, 110],
            "number_of_control_points": [4, 20],
            "number_of_rays": [50, 200],
            "nurbs_degree": [2, 3],
            "ideal_surface_loss_weight": [0.0, 2.0],
            "initial_learning_rate": [1e-7, 1e-3],
            "reduce_factor": [0.05, 0.5],
            "patience": [5, 50],
            "threshold": [1e-6, 1e-3],
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
        "--heliostat_for_reconstruction",
        type=str,
        help="The heliostat and its calibration numbers to be reconstructed.",
        nargs="+",
        default=heliostat_for_reconstruction_default,
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
        "--reconstruction_parameter_ranges",
        type=eval,
        help="Parameters used for the reconstruction.",
        default=reconstruction_parameter_ranges_default,
    )

    # Re-parse the full set of arguments.
    args = parser.parse_args(args=unknown)

    device = get_device(torch.device(args.device))
    data_dir = pathlib.Path(args.data_dir)
    propulate_logs_dir = pathlib.Path(args.propulate_logs_dir)
    results_dir = pathlib.Path(args.results_dir)

    # Define scenario path.
    ideal_scenario_file = (
        pathlib.Path(args.scenarios_dir) / "surface_reconstruction_ideal.h5"
    )
    if not ideal_scenario_file.exists():
        raise FileNotFoundError(
            f"The reconstruction scenario located at {ideal_scenario_file} could not be found! Please run the ``surface_reconstruction_generate_scenario.py`` to generate this scenario, or adjust the file path and try again."
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

    viable_heliostats_data = pathlib.Path(args.results_dir) / "surface_reconstruction_viable_heliostats.json"
    if not viable_heliostats_data.exists():
        raise FileNotFoundError(
            f"The viable heliostat list located at {viable_heliostats_data} could not be not found! Please run the ``surface_reconstruction_viable_heliostat_list.py`` script to generate this list, or adjust the file path and try again."
        )

    # Load viable heliostats data.
    with open(viable_heliostats_data, "r") as f:
        viable_heliostats = json.load(f)

    heliostat_data_mapping: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]] = [
        (
            item["name"],
            [pathlib.Path(p) for p in item["calibrations"]],
            [pathlib.Path(p) for p in item["flux_images"]],
        )
        for item in viable_heliostats
    ]

    reconstruction_parameter_ranges = {}
    for key, value in args.reconstruction_parameter_ranges.items():
        tuple_range = tuple(float(x) if isinstance(x, str) else x for x in value)
        reconstruction_parameter_ranges[key] = tuple_range

    # Set up evolutionary operator.
    num_generations = 500
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
        surface_reconstructor_for_hpo,
        scenario_path=ideal_scenario_file,
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
        top_n=10,
        debug=2,
    )

    hpo_result_file = propulate_logs_dir / "island_0_ckpt.pickle"
    optimized_parameters_file = results_dir / "hpo_results.json"

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
        if isinstance(value, str) and "E" in value.upper():
            parameters_dict[key] = float(value)
        else:
            parameters_dict[key] = value

    if not results_dir.parent.is_dir():
        results_dir.parent.mkdir(parents=True, exist_ok=True)

    with open(optimized_parameters_file, "w") as output_file:
        json.dump(parameters_dict, output_file, indent=2)
