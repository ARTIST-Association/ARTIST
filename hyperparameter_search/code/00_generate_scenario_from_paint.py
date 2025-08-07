import pathlib

import torch

from artist.data_loader import paint_loader
from artist.scenario.configuration_classes import (
    LightSourceConfig,
    LightSourceListConfig,
)
from artist.scenario.h5_scenario_generator import H5ScenarioGenerator
from artist.util import config_dictionary, set_logger_config
from artist.util.environment_setup import get_device

# Set up logger
set_logger_config()

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set the device.
device = get_device()

control_points = 100
number_of_heliostats = 2

# Specify the path to your scenario file.
scenario_path = pathlib.Path(f"/workVERLEIHNIX/mb/ARTIST/hyperparameter_search/ideal_h_{number_of_heliostats}_cp_{control_points}")

# Specify the path to your tower-measurements.json file.
tower_file = pathlib.Path(
    "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/tower-measurements.json"
)

# Specify the following data for each heliostat that you want to include in the scenario:
# A tuple of: (heliostat-name, heliostat-properties.json, deflectometry.h5)
# or to create ideal helisotat surfaces, skip the deflectometry files and specify
# a tuple of: (heliostat-name, heliostat-properties.json)

heliostat_files_list = [
    (
        "AA31",
        pathlib.Path(
            "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA31/heliostat-properties.json"
        ),
        #pathlib.Path("/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/deflectometry.h5"),
    ),
    (
        "AA39",
        pathlib.Path(
            "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/heliostat-properties.json"
        ),
        #pathlib.Path("/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/deflectometry.h5"),
    ),
    # (
    #     "AD37",
    #     pathlib.Path(
    #         "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AD37/heliostat-properties.json"
    #     ),
    #     #pathlib.Path("/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/deflectometry.h5"),
    # ),
    # (
    #     "AU46",
    #     pathlib.Path(
    #         "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AU46/heliostat-properties.json"
    #     ),
    #     #pathlib.Path("/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/deflectometry.h5"),
    # ),
    # (
    #     "AC43",
    #     pathlib.Path(
    #         "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AC43/heliostat-properties.json"
    #     ),
    #     pathlib.Path("/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AC43/deflectometry.h5"),
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

number_of_nurbs_control_points = torch.tensor([control_points, control_points], device=device)
nurbs_fit_method = config_dictionary.fit_nurbs_from_normals
nurbs_deflectometry_step_size = 100
nurbs_fit_tolerance = 1e-10
nurbs_fit_max_epoch = 400

# Please leave the optimizable parameters empty, they will automatically be added for the surface fit.
nurbs_fit_optimizer = torch.optim.Adam([torch.empty(1, requires_grad=True)], lr=1e-3)
nurbs_fit_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    nurbs_fit_optimizer,
    mode="min",
    factor=0.2,
    patience=50,
    threshold=1e-7,
    threshold_mode="abs",
)

heliostat_list_config, prototype_config = paint_loader.extract_paint_heliostats(
    paths=heliostat_files_list,
    power_plant_position=power_plant_config.power_plant_position,
    number_of_nurbs_control_points=number_of_nurbs_control_points,
    deflectometry_step_size=nurbs_deflectometry_step_size,
    nurbs_fit_method=nurbs_fit_method,
    nurbs_fit_tolerance=nurbs_fit_tolerance,
    nurbs_fit_max_epoch=nurbs_fit_max_epoch,
    nurbs_fit_optimizer=nurbs_fit_optimizer,
    nurbs_fit_scheduler=nurbs_fit_scheduler,
    device=device,
)

if __name__ == "__main__":
    """Generate the scenario given the defined parameters."""
    scenario_generator = H5ScenarioGenerator(
        file_path=scenario_path,
        power_plant_config=power_plant_config,
        target_area_list_config=target_area_list_config,
        light_source_list_config=light_source_list_config,
        prototype_config=prototype_config,
        heliostat_list_config=heliostat_list_config,
    )
    scenario_generator.generate_scenario()
