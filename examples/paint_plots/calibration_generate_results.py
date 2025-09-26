import argparse
import json
import pathlib
import warnings
from copy import deepcopy
from typing import Any

import h5py
import paint.util.paint_mappings as paint_mappings
import torch
import yaml

from artist.core import KinematicCalibrator
from artist.core.loss_functions import AngleLoss
from artist.scenario import Scenario
from artist.util import config_dictionary, set_logger_config
from artist.util.environment_setup import get_device, setup_distributed_environment

set_logger_config()
torch.manual_seed(7)
torch.cuda.manual_seed(7)


def generate_calibration_results(
    scenario_path: pathlib.Path,
    heliostat_data_mapping: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    """
    Perform calibration in ``ARTIST`` and save results.

    This function performs the calibration in ``ARTIST`` and saves the results. Calibration is compared when using the
    focal spot centroids extracted from HELIOS and the focal spot centroids extracted from UTIS. The results are saved
    for plotting later.

    Parameters
    ----------
    scenario_path : pathlib.Path
        Path to calibration scenario.
    heliostat_data_mapping : list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
        Data mapping for each heliostat, containing a list of tuples with the heliostat name, the path to the calibration
        properties file, and the path to the flux images.
    device : torch.device | None
        Device used for optimization and tensor allocations.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping from heliostat name to per-centroid loss arrays and, later, positions.
    """
    device = get_device(device=device)

    results_dict: dict = {}

    data: dict[str, str | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]] = {
        config_dictionary.data_source: config_dictionary.paint,
        config_dictionary.heliostat_data_mapping: heliostat_data_mapping,
    }

    number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
        scenario_path=scenario_path
    )

    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file,
            number_of_surface_points_per_facet=torch.tensor([5, 5], device=device),
            device=device,
        )

    with setup_distributed_environment(
        number_of_heliostat_groups=number_of_heliostat_groups,
        device=device,
    ) as ddp_setup:
        # Select calibration via motor positions.
        kinematic_calibration_method = (
            config_dictionary.kinematic_calibration_motor_positions
        )

        # Configure the learning rate scheduler.
        scheduler = config_dictionary.exponential
        scheduler_parameters = {
            config_dictionary.gamma: 0.999,
            config_dictionary.min: 1e-6,
            config_dictionary.max: 1e-2,
            config_dictionary.step_size_up: 500,
            config_dictionary.reduce_factor: 0.3,
            config_dictionary.patience: 10,
            config_dictionary.threshold: 1e-3,
            config_dictionary.cooldown: 10,
        }

        # Set optimization parameters.
        optimization_configuration = {
            config_dictionary.initial_learning_rate: 0.04,
            config_dictionary.tolerance: 0,
            config_dictionary.max_epoch: 3000,
            config_dictionary.log_step: 100,
            config_dictionary.early_stopping_delta: 1e-6,
            config_dictionary.early_stopping_patience: 4000,
            config_dictionary.scheduler: scheduler,
            config_dictionary.scheduler_parameters: scheduler_parameters,
        }

        for centroid in [paint_mappings.UTIS_KEY, paint_mappings.HELIOS_KEY]:
            # Copy scenario so results are comparable across runs.
            current_scenario = deepcopy(scenario)
            loss_definition = AngleLoss()

            kinematic_calibrator = KinematicCalibrator(
                ddp_setup=ddp_setup,
                scenario=current_scenario,
                data=data,
                optimization_configuration=optimization_configuration,
                calibration_method=kinematic_calibration_method,
                centroid_extraction_method=centroid,
            )

            per_heliostat_losses = kinematic_calibrator.calibrate(
                loss_definition=loss_definition, device=device
            )

            for heliostat_group in scenario.heliostat_field.heliostat_groups:
                for index, name in enumerate(heliostat_group.names):
                    results_dict.setdefault(name, {})

                    # Save loss value per centroid.
                    results_dict[name][centroid] = (
                        per_heliostat_losses[index].detach().item()
                    )

    # Include heliostat position.
    for group in scenario.heliostat_field.heliostat_groups:
        for name, position in zip(group.names, group.positions):
            results_dict[name]["Position"] = position.clone().detach().cpu().tolist()
    return results_dict


if __name__ == "__main__":
    """
    Generate calibration results and save them.

    This script performs calibration in ``ARTIST``, generating the results and saving them to be later loaded for the
    plots.

    Parameters
    ----------
    config : str
        Path to the configuration file.
    data_dir : str
        Path to the data directory.
    device : str
        Device to use for the computation.
    results_dir : str
        Path to the directory for the results.
    scenarios_dir : str
        Path to the directory for saving the generated scenarios.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML configuration file.",
        default="examples/paint_plots/paint_plot_config.yaml",
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
    device_default = config.get("device", "cuda")
    results_dir_default = config.get("results_dir", "./results")
    scenarios_dir_default = config.get("scenarios_dir", "./scenarios")

    parser.add_argument(
        "--device",
        type=str,
        help="Device to use.",
        default=device_default,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Path to the results containing the viable heliostats list.",
        default=results_dir_default,
    )
    parser.add_argument(
        "--scenarios_dir",
        type=str,
        help="Path to save the generated scenario.",
        default=scenarios_dir_default,
    )

    # Re-parse the full set of arguments.
    args = parser.parse_args(args=unknown)

    device = get_device(torch.device(args.device))

    viable_heliostats_data = pathlib.Path(args.results_dir) / "viable_heliostats.json"
    if not viable_heliostats_data.exists():
        raise FileNotFoundError(
            f"The viable heliostat list located at {viable_heliostats_data} could not be not found! Please run the ``calibration_generate_viable_heliostats_list.py`` script to generate this list, or adjust the file path and try again."
        )

    # Define scenario path.
    scenario_path = pathlib.Path(args.scenarios_dir) / "calibration.h5"
    if not scenario_path.exists():
        raise FileNotFoundError(
            f"The calibration scenario located at {scenario_path} could not be found! Please run the ``calibration_scenario.py`` to generate this scenario, or adjust the file path and try again."
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

    calibration_results = generate_calibration_results(
        scenario_path=scenario_path,
        heliostat_data_mapping=heliostat_data_mapping,
        device=device,
    )

    results_path = pathlib.Path(args.results_dir) / "calibration_results.pt"
    if not results_path.parent.is_dir():
        results_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(calibration_results, results_path)
    print(f"Calibration results saved to {results_path}")
