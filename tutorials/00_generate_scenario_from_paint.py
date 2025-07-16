import pathlib

import torch

from artist.data_loader import paint_loader
from artist.scenario.configuration_classes import (
    LightSourceConfig,
    LightSourceListConfig,
)
from artist.scenario.scenario_generator import ScenarioGenerator
from artist.util import config_dictionary, set_logger_config
from artist.util.environment_setup import get_device

# Set up logger
set_logger_config()

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set the device.
device = get_device()

# Specify the path to your scenario file.
scenario_path = pathlib.Path(
    "/workVERLEIHNIX/mb/ARTIST/tutorials/data/scenarios/test_scenario_paint_multiple_heliostat_groups_ideal"
)

# Specify the path to your tower-measurements.json file.
tower_file = pathlib.Path(
    "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/tower-measurements.json"
)

# Specify the following data for each heliostat that you want to include in the scenario:
# A tuple of: (heliostat-name, heliostat-properties.json, deflectometry.h5)
heliostat_files_list = [
    (
        "AA31",
        pathlib.Path(
            "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA31/heliostat-properties.json"
        ),
        # pathlib.Path(
        #     "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA31/deflectometry.h5"
        # ),
    ),
    (
        "AA35",
        pathlib.Path(
            "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA35/heliostat-properties.json"
        ),
        # pathlib.Path(
        #     "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA35/deflectometry.h5"
        # ),
    ),
    (
        "AA39",
        pathlib.Path(
            "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/heliostat-properties.json"
        ),
        # pathlib.Path(
        #     "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/deflectometry.h5"
        # ),
    ),
    (
        "AB38",
        pathlib.Path(
            "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AB38/heliostat-properties.json"
        ),
        # pathlib.Path(
        #     "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AB38/deflectometry.h5"
        # ),
    ),
    (
        "AA28",
        pathlib.Path(
            "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA28/heliostat-properties.json"
        ),
        # pathlib.Path(
        #     "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA28/deflectometry.h5"
        # ),
    ),
    (
        "AC43",
        pathlib.Path(
            "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AC43/heliostat-properties.json"
        ),
        # pathlib.Path(
        #     "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AC43/deflectometry.h5"
        # ),
    ),
    # (
    # "name2",
    # pathlib.Path(
    #     "please/insert/the/path/to/the/paint/data/here/heliostat-properties.json"
    # ),
    # pathlib.Path(
    #     "please/insert/the/path/to/the/paint/data/here/deflectometry.h5"
    # ),
    # ),
    # ... Include as many as you want, but at least one!
]

# This checks to make sure the path you defined is valid and a scenario HDF5 can be saved there.
if not pathlib.Path(scenario_path).parent.is_dir():
    raise FileNotFoundError(
        f"The folder ``{pathlib.Path(scenario_path).parent}`` selected to save the scenario does not exist. "
        "Please create the folder or adjust the file path before running again!"
    )

# Include the power plant configuration.
power_plant_config, target_area_list_config = (
    paint_loader.extract_paint_tower_measurements(
        tower_measurements_path=tower_file, device=device
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

# Create a list of light source configs - in this case only one.
light_source_list = [light_source1_config]

# Include the configuration for the list of light sources.
light_source_list_config = LightSourceListConfig(light_source_list=light_source_list)

target_area = [
    target_area
    for target_area in target_area_list_config.target_area_list
    if target_area.target_area_key == config_dictionary.target_area_receiver
]

heliostat_list_config, prototype_config = paint_loader.extract_paint_heliostats(
    paths=heliostat_files_list,
    power_plant_position=power_plant_config.power_plant_position,
    aim_point=target_area[0].center,
    device=device,
)


if __name__ == "__main__":
    """Generate the scenario given the defined parameters."""
    scenario_generator = ScenarioGenerator(
        file_path=scenario_path,
        power_plant_config=power_plant_config,
        target_area_list_config=target_area_list_config,
        light_source_list_config=light_source_list_config,
        prototype_config=prototype_config,
        heliostat_list_config=heliostat_list_config,
    )
    scenario_generator.generate_scenario()
