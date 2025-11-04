import argparse
import json
import pathlib
import warnings
from pathlib import Path

import paint.util.paint_mappings as paint_mappings
import torch
import yaml

from artist.data_parser import paint_scenario_parser
from artist.scenario.configuration_classes import (
    LightSourceConfig,
    LightSourceListConfig,
)
from artist.scenario.h5_scenario_generator import H5ScenarioGenerator
from artist.util import config_dictionary, set_logger_config
from artist.util.environment_setup import get_device

set_logger_config()


def find_latest_deflectometry_file(heliostat_name: str, data_directory: Path) -> Path:
    """
    Find the latest deflectometry HDF5 file for a given heliostat.

    Parameters
    ----------
    heliostat_name : str
        Heliostat name being considered.
    data_directory : Path
        Data directory containing ``PAINT`` data.

    Returns
    -------
    pathlib.Path
        Path to the latest deflectometry file.
    """
    search_path = (
        pathlib.Path(data_directory)
        / heliostat_name
        / paint_mappings.SAVE_DEFLECTOMETRY
    )
    pattern = f"{heliostat_name}-filled*.h5"
    files = sorted(search_path.glob(pattern))
    if not files:
        return None
    return files[-1]


def generate_ideal_scenario(
    scenario_path: pathlib.Path,
    tower_file: pathlib.Path,
    heliostat_files_list: list[tuple[str, pathlib.Path]],
    device: torch.device | None = None,
) -> None:
    """
    Generate an ideal HDF5 scenario for the field optimizations.

    Parameters
    ----------
    scenario_path : pathlib.Path
        Path to save the generated HDF5 scenario.
    tower_file_path : pathlib.Path
        Path to the tower measurements file.
    heliostat_files_list : list[tuple[str, pathlib.Path]]
        List of heliostat names and their property files to include in the scenario.
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


def generate_fitted_scenario(
    data_directory: pathlib.Path,
    scenario_path: pathlib.Path,
    tower_file_path: pathlib.Path,
    heliostat_names: list[str],
    device: torch.device | None = None,
) -> None:
    """
    Generate a deflectometry HDF5 scenario for the evaluation of the field optimizations.
    
    Parameters
    ----------
    data_directory : pathlib.Path
        Path to the data directory.
    scenario_path : pathlib.Path
        Path to where the scenarios will be saved.
    tower_file_path : pathlib.Path
        Path to the tower data file.
    heliostat_names : list[str]
        List of heliostat names to include in the scenario.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.
    """
    device = get_device(device=device)

    # Include the power plant configuration.
    power_plant_config, target_area_list_config = (
        paint_scenario_parser.extract_paint_tower_measurements(
            tower_measurements_path=tower_file_path,
            device=device,
        )
    )

    # Include the light source configuration.
    light_source1_config = LightSourceConfig(
        light_source_key="sun_1",
        light_source_type=config_dictionary.sun_key,
        number_of_rays=10,
        distribution_type=config_dictionary.light_source_distribution_is_normal,
        mean=0.0,
        covariance=4.3681e-06,
    )

    # Create a list of light source configs.
    light_source_list = [light_source1_config]

    # Include the configuration for the list of light sources.
    light_source_list_config = LightSourceListConfig(
        light_source_list=light_source_list
    )

    heliostat_files_list = [
        (
            heliostat_name,
            pathlib.Path(
                f"{data_directory}/{heliostat_name}/{paint_mappings.SAVE_PROPERTIES}/{heliostat_name}-{paint_mappings.HELIOSTAT_PROPERTIES_KEY}.json"
            ),
            deflectometry_file,
        )
        for heliostat_name in heliostat_names
        if (deflectometry_file := find_latest_deflectometry_file(heliostat_name, data_directory)) is not None
    ]

    # Fit the NURBS.
    nurbs_fit_optimizer = torch.optim.Adam(
        [torch.empty(1, requires_grad=True)], lr=1e-3
    )
    nurbs_fit_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        nurbs_fit_optimizer,
        mode="min",
        factor=0.2,
        patience=50,
        threshold=1e-7,
        threshold_mode="abs",
    )

    # Create the list of heliostats.
    heliostat_list_config, prototype_config = (
        paint_scenario_parser.extract_paint_heliostats_fitted_surface(
            paths=heliostat_files_list,
            power_plant_position=power_plant_config.power_plant_position,
            number_of_nurbs_control_points=torch.tensor([20, 20], device=device),
            deflectometry_step_size=100,
            nurbs_fit_method=config_dictionary.fit_nurbs_from_normals,
            nurbs_fit_tolerance=1e-10,
            nurbs_fit_max_epoch=400,
            nurbs_fit_optimizer=nurbs_fit_optimizer,
            nurbs_fit_scheduler=nurbs_fit_scheduler,
            device=device,
        )
    )

    # Generate the scenario given the defined parameters.
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
    Generate scenarios for the field optimizations.

    This will generate ideal scenarios for the baseline case and for the full-field. Additionally it will generate 
    a fitted scenario with deflectometry data, containing only the heliostats from the baseline case which also have 
    available deflectometry measurements. The deflectometry scenario is used for the evaluation plots. 

    Parameters
    ----------
    config : str
        Path to the configuration file.
    device : str
        Device to use for the computation.
    data_dir : str
        Path to the data directory.
    tower_file_name : str
        Name of the file containing the tower measurements.
    results_dir : str
        Path to the results directory containing the viable heliostats list.
    scenarios_dir : str
        Path to the directory for saving the generated scenarios.
    heliostat_list_baseline : list[str]
        List of all heliostat names included in the baseline measurement.
    """
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
    device_default = config.get("device", "cuda")
    data_dir_default = config.get("data_dir", "./examples/field_optimizations/field_data")
    tower_file_name_default = config.get(
        "tower_file_name", "WRI1030197-tower-measurements.json"
    )
    results_dir_default = config.get("results_dir", "./examples/field_optimizations/results")
    scenarios_dir_default = config.get("scenarios_dir", "./examples/field_optimizations/scenarios")
    heliostat_list_baseline_default = config.get("heliostat_list_baseline", None)

    parser.add_argument(
        "--device",
        type=str,
        help="Device to use.",
        default=device_default,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to the data directory.",
        default=data_dir_default,
    )
    parser.add_argument(
        "--tower_file_name",
        type=str,
        help="Name of the file containing the tower measurements.",
        default=tower_file_name_default,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Path to the results directory containing the viable heliostats list.",
        default=results_dir_default,
    )
    parser.add_argument(
        "--scenarios_dir",
        type=str,
        help="Path to the directory for saving the generated scenarios.",
        default=scenarios_dir_default,
    )
    parser.add_argument(
        "--heliostat_list_baseline",
        type=list[str],
        help="List of all heliostat names included in the baseline measurement.",
        default=heliostat_list_baseline_default,
    )

    # Re-parse the full set of arguments.
    args = parser.parse_args(args=unknown)

    device = get_device(torch.device(args.device))
    data_dir = pathlib.Path(args.data_dir)
    tower_file = data_dir / args.tower_file_name

    for case in ["baseline", "full_field"]:
        viable_heliostats_data = pathlib.Path(args.results_dir) / case / "viable_heliostats.json"
        if not viable_heliostats_data.exists():
            raise FileNotFoundError(
                f"The viable heliostat list located at {viable_heliostats_data} could not be not found! Please run the ``generate_viable_heliostats_list.py`` script to generate this list, or adjust the file path and try again."
            )

        # Define scenario path.
        scenario_path = pathlib.Path(args.scenarios_dir) / f"ideal_{case}_scenario.h5"
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
            generate_ideal_scenario(
                scenario_path=scenario_path,
                tower_file=tower_file,
                heliostat_files_list=heliostat_properties_list,
                device=device,
            )

    deflectometry_scenario_path = (
        pathlib.Path(args.scenarios_dir) / "deflectometry_scenario_for_comparison.h5"
    )
    if deflectometry_scenario_path.exists():
        print(
            f"Scenario found at {deflectometry_scenario_path}... continue without generating scenario."
        )
    else:
        print(
            f"Scenario not found. Generating a new one at {deflectometry_scenario_path}..."
        )
        generate_fitted_scenario(
            data_directory=data_dir,
            scenario_path=deflectometry_scenario_path,
            tower_file_path=tower_file,
            heliostat_names=args.heliostat_list_baseline,
            device=device,
        )
