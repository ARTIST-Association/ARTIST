import argparse
import pathlib
import warnings
from pathlib import Path

import paint.util.paint_mappings as paint_mappings
import torch
import yaml

from artist.data_loader import paint_loader
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

    Raises
    ------
    FileNotFoundError
        If no matching file is found.
    """
    search_path = (
        pathlib.Path(data_directory)
        / heliostat_name
        / paint_mappings.SAVE_DEFLECTOMETRY
    )
    pattern = f"{heliostat_name}-filled*.h5"
    files = sorted(search_path.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No deflectometry file found for {heliostat_name} in {search_path}."
        )
    return files[-1]


def generate_flux_prediction_scenario(
    data_directory: Path,
    scenario_path: Path,
    tower_file_path: Path,
    heliostat_names: list[str],
    device: torch.device | None = None,
    use_deflectometry: bool = True,
) -> None:
    """
    Generate an HDF5 scenario for the flux prediction plots using ``PAINT`` data.

    Parameters
    ----------
    data_directory : pathlib.Path
        Directory where the ``PAINT`` data is stored.
    scenario_path : pathlib.Path
        Path to save the generated HDF5 scenario.
    tower_file_path : pathlib.Path
        Path to the tower measurements file.
    heliostat_names : list[str]
        Names of the heliostats to include in the scenario.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ARTIST will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.
    use_deflectometry : bool, optional
        Whether to use deflectometry data for surface fitting (default is ``True``).
    """
    device = get_device(device=device)

    # Make sure the parent folder is available for saving the scenario.
    if not scenario_path.exists():
        scenario_path.parent.mkdir(parents=True, exist_ok=True)

    # Include the power plant configuration.
    power_plant_config, target_area_list_config = (
        paint_loader.extract_paint_tower_measurements(
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

    heliostat_files_list: (
        list[tuple[str, pathlib.Path]] | list[tuple[str, pathlib.Path, pathlib.Path]]
    ) = []

    if use_deflectometry:
        heliostat_files_list = [
            (
                heliostat_name,
                pathlib.Path(
                    f"{data_directory}/{heliostat_name}/{paint_mappings.SAVE_PROPERTIES}/{heliostat_name}-{paint_mappings.HELIOSTAT_PROPERTIES_KEY}.json"
                ),
                find_latest_deflectometry_file(heliostat_name, data_directory),
            )
            for heliostat_name in heliostat_names
        ]

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

        heliostat_list_config, prototype_config = (
            paint_loader.extract_paint_heliostats_fitted_surface(
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
    else:
        heliostat_files_list = [
            (
                heliostat_name,
                pathlib.Path(
                    f"{data_directory}/{heliostat_name}/{paint_mappings.SAVE_PROPERTIES}/{heliostat_name}-{paint_mappings.HELIOSTAT_PROPERTIES_KEY}.json"
                ),
            )
            for heliostat_name in heliostat_names
        ]
        heliostat_list_config, prototype_config = (
            paint_loader.extract_paint_heliostats_ideal_surface(
                paths=heliostat_files_list,
                power_plant_position=power_plant_config.power_plant_position,
                number_of_nurbs_control_points=torch.tensor([20, 20], device=device),
                device=device,
            )
        )

    # Generate the scenario given the defined parameters.
    scenario_generator = H5ScenarioGenerator(
        file_path=scenario_path,  # pass Path
        power_plant_config=power_plant_config,
        target_area_list_config=target_area_list_config,
        light_source_list_config=light_source_list_config,
        prototype_config=prototype_config,
        heliostat_list_config=heliostat_list_config,
    )
    scenario_generator.generate_scenario()


if __name__ == "__main__":
    """
    Generate two scenario for the flux prediction plots.

    One of these scenarios uses ideal surfaces whilst one includes surfaces fitted with deflectometry data.
    If a configuration file is provided the values will be loaded from this file. It is also possible to override
    the configuration file using command line arguments. If no command line arguments and no configuration file
    is provided, default values will be used which may fail.

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
    heliostats : dict[str, int]
        The heliostats and associated calibration measurement required in the scenario.
    scenarios_dir : str
        Path to the directory for saving the generated scenarios.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML configuration file.",
        default="./paint_plot_config.yaml",
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
    metadata_dir_default = config.get("metadata_dir", "./metadata")
    metadata_file_name_default = config.get(
        "metadata_file_name", "calibration_metadata_all_heliostats.csv"
    )
    minimum_number_of_measurements_default = config.get(
        "minimum_number_of_measurements", 80
    )
    device_default = config.get("device", "cuda")
    tower_file_name_default = config.get(
        "tower_file_name", "WRI1030197-tower-measurements.json"
    )
    heliostats_default = config.get(
        "heliostats_for_raytracing", {"AA39": 149576, "AY26": 247613, "BC34": 82084}
    )
    scenarios_dir_default = config.get("scenarios_dir", "./scenarios")

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
        "--heliostats",
        type=str,
        help="Heliostats and calibration measurement required in the scenario.",
        nargs="+",
        default=heliostats_default,
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

    # Generate two scenarios: deflectometry and ideal (no deflectometry).
    deflectometry_scenario_file = (
        pathlib.Path(args.scenarios_dir) / "flux_prediction_deflectometry.h5"
    )
    ideal_scenario_file = pathlib.Path(args.scenarios_dir) / "flux_prediction_ideal.h5"

    for scenario_path, use_deflectometry in [
        (deflectometry_scenario_file, True),
        (ideal_scenario_file, False),
    ]:
        if scenario_path.exists():
            print(
                f"Scenario found at {scenario_path}... continue without generating scenario."
            )
        else:
            print(
                f"Scenario not found. Generating a new one at {scenario_path} (use_deflectometry={use_deflectometry})..."
            )
            generate_flux_prediction_scenario(
                data_directory=data_dir,
                scenario_path=scenario_path,
                tower_file_path=tower_file,
                heliostat_names=list(args.heliostats.keys()),
                device=device,
                use_deflectometry=use_deflectometry,
            )
