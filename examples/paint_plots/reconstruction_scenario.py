import argparse
import json
import pathlib
import warnings

import torch
import yaml

from artist.data_parser import paint_scenario_parser
from artist.scenario import (
    H5ScenarioGenerator,
    LightSourceConfig,
    LightSourceListConfig,
)
from artist.util import config_dictionary, set_logger_config
from artist.util.environment_setup import get_device

set_logger_config()


def generate_reconstruction_scenario(
    scenario_path: pathlib.Path,
    tower_file: pathlib.Path,
    heliostat_files_list: list[tuple[str, pathlib.Path]],
    device: torch.device | None = None,
) -> None:
    """
    Generate a scenario for the kinematic reconstruction plots.

    Parameters
    ----------
    scenario_path : pathlib.Path
        File path where the generated HDF5 scenario will be stored.
    tower_file : pathlib.Path
        Path to the file containing the tower measurements data.
    heliostat_files_list : list[tuple[str, pathlib.Path]]
        A list of heliostat files containing a tuple of heliostat name and path to the properties file.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.
    """
    device = get_device(device=device)

    # Generate power plant configuration and target area list.
    power_plant_config, target_area_list_config = (
        paint_scenario_parser.extract_paint_tower_measurements(
            tower_measurements_path=tower_file, device=device
        )
    )

    # Set up light source configuration.
    light_source1_config = LightSourceConfig(
        light_source_key="sun_1",
        light_source_type=config_dictionary.sun_key,
        number_of_rays=10,
        distribution_type=config_dictionary.light_source_distribution_is_normal,
        mean=0.0,
        covariance=4.3681e-06,
    )
    light_source_list = [light_source1_config]
    light_source_list_config = LightSourceListConfig(
        light_source_list=light_source_list
    )

    # Generate heliostat list configuration.
    heliostat_list_config, prototype_config = (
        paint_scenario_parser.extract_paint_heliostats_ideal_surface(
            paths=heliostat_files_list,
            power_plant_position=power_plant_config.power_plant_position,
            device=device,
        )
    )

    # Generate scenario.
    scenario_generator = H5ScenarioGenerator(
        file_path=scenario_path,
        power_plant_config=power_plant_config,
        target_area_list_config=target_area_list_config,
        light_source_list_config=light_source_list_config,
        prototype_config=prototype_config,
        heliostat_list_config=heliostat_list_config,
    )
    scenario_generator.generate_scenario()


if __name__ == "__main__":
    """
    Generate a scenario for the kinematic reconstruction plots.

    This script generates a scenario based on the viable heliostats list previously generated.

    Parameters
    ----------
    config : str
        Path to the configuration file.
    data_dir : str
        Path to the data directory.
    device : str
        Device to use for the computation.
    tower_file_name : str
        Name of the file containing the tower measurements.
    results_dir : str
        Path to the directory containing the viable heliostat list required for scenario generation.
    scenarios_dir : str
        Path to the directory for saving the generated scenarios.
    """
    # Set default location for configuration file.
    script_dir = pathlib.Path(__file__).resolve().parent
    default_config_path = script_dir / "paint_plot_config.yaml"

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
    tower_file_name_default = config.get(
        "tower_file_name", "WRI1030197-tower-measurements.json"
    )
    results_dir_default = config.get("results_dir", "./examples/paint_plots/results")
    scenarios_dir_default = config.get(
        "scenarios_dir", "./examples/paint_plots/scenarios"
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
        "--tower_file_name",
        type=str,
        help="File name containing the tower data.",
        default=tower_file_name_default,
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

    data_dir = pathlib.Path(args.data_dir)
    tower_file = data_dir / args.tower_file_name

    viable_heliostats_data = pathlib.Path(args.results_dir) / "viable_heliostats.json"
    if not viable_heliostats_data.exists():
        raise FileNotFoundError(
            f"The viable heliostat list located at {viable_heliostats_data} could not be not found! Please run the ``reconstruction_generate_viable_heliostats_list.py`` script to generate this list, or adjust the file path and try again."
        )

    # Define scenario path.
    scenario_path = pathlib.Path(args.scenarios_dir) / "reconstruction.h5"
    if not scenario_path.parent.exists():
        scenario_path.parent.mkdir(parents=True, exist_ok=True)

    # Load viable heliostats data.
    with open(viable_heliostats_data, "r") as f:
        viable_heliostats = json.load(f)

    heliostat_properties_list: list[tuple[str, pathlib.Path]] = [
        (
            item["name"],
            pathlib.Path(item["properties"]),
        )
        for item in viable_heliostats
    ]

    if scenario_path.exists():
        print(
            f"Scenario found at {scenario_path}... continue without generating scenario."
        )
    else:
        print(f"Scenario not found. Generating a new one at {scenario_path}...")
        generate_reconstruction_scenario(
            scenario_path=scenario_path,
            tower_file=tower_file,
            heliostat_files_list=heliostat_properties_list,
            device=device,
        )
