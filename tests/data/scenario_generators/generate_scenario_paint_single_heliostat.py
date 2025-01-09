import pathlib

import torch

from artist import ARTIST_ROOT
from artist.util import config_dictionary, paint_loader, set_logger_config
from artist.util.configuration_classes import (
    LightSourceConfig,
    LightSourceListConfig,
    PrototypeConfig,
)
from artist.util.scenario_generator import ScenarioGenerator

# Set up logger
set_logger_config()

torch.manual_seed(7)
torch.cuda.manual_seed(7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The following parameter is the name of the scenario.
file_path = pathlib.Path(ARTIST_ROOT) / "tests/data/scenarios/test_scenario_paint_single_heliostat"

if not pathlib.Path(file_path).parent.is_dir():
    raise FileNotFoundError(
        f"The folder ``{pathlib.Path(file_path).parent}`` selected to save the scenario does not exist."
        "Please create the folder or adjust the file path before running again!"
    )

tower_file = pathlib.Path(ARTIST_ROOT) / "tests/data/paint_multiple_heliostats/tower-measurements.json"

# Include the power plant configuration.
power_plant_config, target_area_list_config = (
    paint_loader.extract_paint_tower_measurements(
        tower_measurements_path=tower_file, 
        device=device
    )
)

# Include the light source configuration.
light_source1_config = LightSourceConfig(
    light_source_key="sun_1",
    light_source_type=config_dictionary.sun_key,
    number_of_rays=1,
    distribution_type=config_dictionary.light_source_distribution_is_normal,
    mean=0.0,
    covariance=4.3681e-06,
)

# Create a list of light source configs - in this case only one.
light_source_list = [light_source1_config]

# Include the configuration for the list of light sources.
light_source_list_config = LightSourceListConfig(light_source_list=light_source_list)

# Include the configuration for the list of heliostats.
heliostats = ["AA39"]
heliostat_files_list = []
for heliostat in heliostats:
    heliostat_files = (heliostat,
                       pathlib.Path(ARTIST_ROOT) / f"tests/data/paint_multiple_heliostats/{heliostat}/heliostat-properties.json",
                       pathlib.Path(ARTIST_ROOT) / f"tests/data/paint_multiple_heliostats/{heliostat}/deflectometry.h5")
    heliostat_files_list.append(heliostat_files)

target_area = [target_area for target_area in target_area_list_config.target_area_list if target_area.target_area_key == config_dictionary.target_area_reveicer]

heliostat_list_config = (
    paint_loader.extract_paint_heliostats(
        heliostat_and_deflectometry_paths=heliostat_files_list, 
        power_plant_position=power_plant_config.power_plant_position,
        aim_point=target_area[0].center,
        device=device
    )
)

# Include the configuration for a prototype. (Will be extracted from the first heliostat in the list.)
prototype_config = PrototypeConfig(
    surface_prototype=heliostat_list_config.heliostat_list[0].surface,
    kinematic_prototype=heliostat_list_config.heliostat_list[0].kinematic,
    actuators_prototype=heliostat_list_config.heliostat_list[0].actuators,
)


if __name__ == "__main__":
    """Generate the scenario given the defined parameters."""
    scenario_generator = ScenarioGenerator(
        file_path=file_path,
        power_plant_config=power_plant_config,
        target_area_list_config=target_area_list_config,
        light_source_list_config=light_source_list_config,
        prototype_config=prototype_config,
        heliostat_list_config=heliostat_list_config,
    )
    scenario_generator.generate_scenario()
