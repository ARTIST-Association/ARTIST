import pathlib
from typing import Optional

import torch

from artist.data_loader import paint_loader
from artist.scenario.configuration_classes import (
    LightSourceConfig,
    LightSourceListConfig,
)
from artist.scenario.h5_scenario_generator import H5ScenarioGenerator
from artist.util import config_dictionary
from artist.util.environment_setup import get_device
from examples.paint_plots.helpers import join_safe, load_config, load_heliostat_data


def _generate_paint_scenario(
    scenario_path: str | pathlib.Path,
    tower_file: str | pathlib.Path,
    heliostat_files_list: list[tuple[str, pathlib.Path]],
    device: Optional[torch.device],
) -> None:
    """Generate a scenario file based on tower and heliostat data.

    Parameters
    ----------
    scenario_path : str | pathlib.Path
        File path where the generated HDF5 scenario will be stored. The parent directory must exist.
    tower_file : str | pathlib.Path
        Path to the tower measurements file (JSON/HDF5 as expected by `paint_loader.extract_paint_tower_measurements`).
    heliostat_files_list : list[tuple[str, pathlib.Path]]
        List of tuples `(heliostat_name, properties_path)` for all heliostats to include.
    device : str, default="cpu"
        Device identifier used during scenario generation (e.g., "cpu" or "cuda").

    Raises
    ------
    FileNotFoundError
        If the parent directory of `scenario_path` does not exist.
    """
    scenario_path = pathlib.Path(scenario_path)
    scenario_path.parent.mkdir(parents=True, exist_ok=True)

    power_plant_config, target_area_list_config = (
        paint_loader.extract_paint_tower_measurements(
            tower_measurements_path=pathlib.Path(tower_file), device=device
        )
    )

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

    heliostat_list_config, prototype_config = paint_loader.extract_paint_heliostats(
        paths=heliostat_files_list,
        power_plant_position=power_plant_config.power_plant_position,
        device=device,
    )

    scenario_generator = H5ScenarioGenerator(
        file_path=pathlib.Path(scenario_path),
        power_plant_config=power_plant_config,
        target_area_list_config=target_area_list_config,
        light_source_list_config=light_source_list_config,
        prototype_config=prototype_config,
        heliostat_list_config=heliostat_list_config,
    )
    scenario_generator.generate_scenario()


def create_scenario(
    scenario_path: str | pathlib.Path,
    tower_file: str | pathlib.Path,
    heliostat_properties_list: list[tuple[str, pathlib.Path]],
    device: Optional[torch.device],
) -> None:
    """Ensure scenario exists, create if missing, and return loaded Scenario and its HDF5 path.

    Parameters
    ----------
    scenario_path : str | pathlib.Path
        Desired path of the scenario file (.h5). If the suffix is missing, ".h5" is appended.
    tower_file : str | pathlib.Path
        Tower measurements file used when generating the scenario.
    heliostat_properties_list : list[tuple[str, pathlib.Path]]
        Heliostat names and corresponding properties file paths.
    device : torch.device | None
        Device for loading the scenario and setting tensor devices.

    Returns
    -------
    tuple
        (scenario: Scenario, scenario_h5_path: pathlib.Path)
    """
    scenario_path = pathlib.Path(scenario_path)
    if scenario_path.suffix != ".h5":
        scenario_path = scenario_path.with_suffix(".h5")

    if not scenario_path.exists():
        print(f"Scenario file not found at {scenario_path}. Generating scenario...")
        _generate_paint_scenario(
            scenario_path=scenario_path,
            tower_file=tower_file,
            heliostat_files_list=heliostat_properties_list,
            device=device,
        )
    else:
        print(f"Scenario already exists at {scenario_path}. Skipping generation.")


if __name__ == "__main__":
    config = load_config()
    device = torch.device(config["device"])

    device = get_device(device=device)

    paint_directory = pathlib.Path(config["paint_repository_base_path"])
    tower_file = join_safe(paint_directory, config["paint_tower_file"])

    paint_plot_base_path = pathlib.Path(config["base_path"])
    scenario_path = join_safe(paint_plot_base_path, config["calibration_scenario_path"])

    heliostat_list_file = join_safe(paint_plot_base_path, config["heliostat_list_path"])

    heliostat_data_mapping, heliostat_properties_list = load_heliostat_data(
        paint_directory, heliostat_list_file
    )

    create_scenario(
        scenario_path=scenario_path,
        tower_file=tower_file,
        heliostat_properties_list=heliostat_properties_list,
        device=device,
    )
