import pathlib

import torch

from artist.data_parser import stral_scenario_parser
from artist.scenario.configuration_classes import (
    ActuatorConfig,
    ActuatorPrototypeConfig,
    HeliostatConfig,
    HeliostatListConfig,
    KinematicPrototypeConfig,
    LightSourceConfig,
    LightSourceListConfig,
    PowerPlantConfig,
    PrototypeConfig,
    SurfacePrototypeConfig,
    TargetAreaConfig,
    TargetAreaListConfig,
)
from artist.scenario.h5_scenario_generator import H5ScenarioGenerator
from artist.scenario.surface_generator import SurfaceGenerator
from artist.util import config_dictionary, set_logger_config
from artist.util.environment_setup import get_device

# Set up logger.
set_logger_config()

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set the device.
device = get_device()

# Specify the path to your scenario file.
scenario_path = pathlib.Path("please/insert/the/path/to/the/scenario/here/name")

# Specify the path to your stral_data.binp file.
stral_file_path = pathlib.Path(
    "please/insert/the/path/to/the/stral/data/here/test_stral_data.binp"
)

# This checks to make sure the path you defined is valid and a scenario HDF5 can be saved there.
if not pathlib.Path(scenario_path).parent.is_dir():
    raise FileNotFoundError(
        f"The folder ``{pathlib.Path(scenario_path).parent}`` selected to save the scenario does not exist. "
        "Please create the folder or adjust the file path before running again!"
    )

# Include the power plant configuration.
power_plant_config = PowerPlantConfig(
    power_plant_position=torch.tensor([0.0, 0.0, 0.0], device=device)
)

# Include a single tower area (receiver).
receiver_config = TargetAreaConfig(
    target_area_key="receiver",
    geometry=config_dictionary.target_area_type_planar,
    center=torch.tensor([0.0, -50.0, 0.0, 1.0], device=device),
    normal_vector=torch.tensor([0.0, 1.0, 0.0, 0.0], device=device),
    plane_e=8.629666667,
    plane_u=7.0,
)

# Create list of target area configs - in this case only one.
target_area_config_list = [receiver_config]

# Include the tower area configurations.
target_area_list_config = TargetAreaListConfig(target_area_config_list)

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

(
    facet_translation_vectors,
    canting,
    surface_points_with_facets_list,
    surface_normals_with_facets_list,
) = stral_scenario_parser.extract_stral_deflectometry_data(
    stral_file_path=stral_file_path, device=device
)

# Generate surface configuration from STRAL data.
surface_generator = SurfaceGenerator(device=device)

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

# Use this surface config for fitted deflectometry surfaces.
surface_config = surface_generator.generate_fitted_surface_config(
    heliostat_name="heliostat_1",
    facet_translation_vectors=facet_translation_vectors,
    canting=canting,
    surface_points_with_facets_list=surface_points_with_facets_list,
    surface_normals_with_facets_list=surface_normals_with_facets_list,
    optimizer=nurbs_fit_optimizer,
    scheduler=nurbs_fit_scheduler,
    device=device,
)

# Use this surface configuration for ideal surfaces.
# surface_config = surface_generator.generate_ideal_surface_config(
#     facet_translation_vectors=facet_translation_vectors,
#     canting=canting,
#     device=device,
# )

surface_prototype_config = SurfacePrototypeConfig(facet_list=surface_config.facet_list)

# Include the kinematic prototype configuration.
kinematic_prototype_config = KinematicPrototypeConfig(
    type=config_dictionary.rigid_body_key,
    initial_orientation=torch.tensor([0.0, 0.0, 1.0, 0.0]),
)

# The minimum and maximum motor positions provided here are approximations of the actuators in the heliostat field in Juelich.
min_max_motor_positions_actuator_1 = [0.0, 60000.0]
min_max_motor_positions_actuator_2 = [0.0, 80000.0]

# Include an ideal actuator.
actuator1_prototype = ActuatorConfig(
    key="actuator_1",
    type=config_dictionary.ideal_actuator_key,
    clockwise_axis_movement=False,
    min_max_motor_positions=min_max_motor_positions_actuator_1,
)

# Include an ideal actuator.
actuator2_prototype = ActuatorConfig(
    key="actuator_2",
    type=config_dictionary.ideal_actuator_key,
    clockwise_axis_movement=True,
    min_max_motor_positions=min_max_motor_positions_actuator_2,
)

# Create a list of actuators.
actuator_prototype_list = [actuator1_prototype, actuator2_prototype]

# Include the actuator prototype config.
actuator_prototype_config = ActuatorPrototypeConfig(
    actuator_list=actuator_prototype_list
)

# Include the final prototype config.
prototype_config = PrototypeConfig(
    surface_prototype=surface_prototype_config,
    kinematic_prototype=kinematic_prototype_config,
    actuators_prototype=actuator_prototype_config,
)


# Include the configuration for a heliostat.
heliostat1 = HeliostatConfig(
    name="heliostat_1",
    id=1,
    position=torch.tensor([0.0, 5.0, 0.0, 1.0], device=device),
)

# Create a list of all the heliostats - in this case, only one.
heliostat_list = [heliostat1]

# Create the configuration for all heliostats.
heliostats_list_config = HeliostatListConfig(heliostat_list=heliostat_list)

if __name__ == "__main__":
    """Generate the scenario given the defined parameters."""
    scenario_generator = H5ScenarioGenerator(
        file_path=scenario_path,
        power_plant_config=power_plant_config,
        target_area_list_config=target_area_list_config,
        light_source_list_config=light_source_list_config,
        prototype_config=prototype_config,
        heliostat_list_config=heliostats_list_config,
    )
    scenario_generator.generate_scenario()
