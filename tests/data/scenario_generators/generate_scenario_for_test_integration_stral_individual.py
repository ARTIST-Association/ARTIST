import pathlib

import torch

from artist import ARTIST_ROOT
from artist.util import config_dictionary
from artist.util.configuration_classes import (
    ActuatorConfig,
    ActuatorListConfig,
    ActuatorPrototypeConfig,
    HeliostatConfig,
    HeliostatListConfig,
    KinematicConfig,
    KinematicOffsets,
    KinematicPrototypeConfig,
    LightSourceConfig,
    LightSourceListConfig,
    PowerPlantConfig,
    PrototypeConfig,
    ReceiverConfig,
    ReceiverListConfig,
    SurfaceConfig,
    SurfacePrototypeConfig,
)
from artist.util.scenario_generator import ScenarioGenerator
from artist.util.stral_to_surface_converter import StralToSurfaceConverter

torch.manual_seed(7)
torch.cuda.manual_seed(7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The following parameter is the name of the scenario.
file_path = file_path = (
    pathlib.Path(ARTIST_ROOT) / "tests/data/test_individual_measurements_scenario"
)

if not pathlib.Path(file_path).parent.is_dir():
    raise FileNotFoundError(
        f"The folder ``{pathlib.Path(file_path).parent}`` selected to save the scenario does not exist. "
        "Please create the folder or adjust the file path before running again!"
    )

# Include the power plant configuration.
power_plant_config = PowerPlantConfig(
    power_plant_position=torch.tensor([0.0, 0.0, 0.0], device=device)
)

# Include the receiver configuration.
receiver1_config = ReceiverConfig(
    receiver_key="receiver1",
    receiver_type=config_dictionary.receiver_type_planar,
    position_center=torch.tensor([0.0, -50.0, 0.0, 1.0], device=device),
    normal_vector=torch.tensor([0.0, 1.0, 0.0, 0.0], device=device),
    plane_e=8.629666667,
    plane_u=7.0,
    resolution_e=256,
    resolution_u=256,
)

# Create list of receiver configs - in this case only one.
receiver_list = [receiver1_config]

# Include the configuration for the list of receivers.
receiver_list_config = ReceiverListConfig(receiver_list=receiver_list)

# Include the light source configuration.
light_source1_config = LightSourceConfig(
    light_source_key="sun1",
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

# Generate surface configuration from STRAL data.
stral_converter = StralToSurfaceConverter(
    stral_file_path=pathlib.Path(ARTIST_ROOT) / "tests/data/stral_test_data",
    surface_header_name="=5f2I2f",
    facet_header_name="=i9fI",
    points_on_facet_struct_name="=7f",
    step_size=100,
)
list_of_facets = stral_converter.generate_surface_config_from_stral(
    number_eval_points_e=100,
    number_eval_points_n=100,
    conversion_method=config_dictionary.convert_nurbs_from_normals,
    number_control_points_e=20,
    number_control_points_n=20,
    degree_e=3,
    degree_n=3,
    tolerance=3e-5,
    max_epoch=10000,
    initial_learning_rate=1e-3,
    device=device,
)

surface_prototype_config = SurfacePrototypeConfig(facets_list=list_of_facets)

# Include the initial orientation offsets for the kinematic.
kinematic_prototype_offsets = KinematicOffsets(
    kinematic_initial_orientation_offset_e=torch.tensor(
        torch.tensor(torch.pi / 2, device=device), device=device
    )
)

# Include the kinematic prototype configuration.
kinematic_prototype_config = KinematicPrototypeConfig(
    kinematic_type=config_dictionary.rigid_body_key,
    kinematic_initial_orientation_offsets=kinematic_prototype_offsets,
)

# Include an ideal actuator.
actuator1_prototype = ActuatorConfig(
    actuator_key="actuator1",
    actuator_type=config_dictionary.ideal_actuator_key,
    actuator_clockwise=False,
)

# Include a linear actuator.
actuator2_prototype = ActuatorConfig(
    actuator_key="actuator2",
    actuator_type=config_dictionary.ideal_actuator_key,
    actuator_clockwise=True,
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
    actuator_prototype=actuator_prototype_config,
)

# Include the heliostat surface config. In this case, it is identical to the prototype.
heliostat1_surface_config = SurfaceConfig(facets_list=list_of_facets)

# Include kinematic configuration for the heliostat.
heliostat1_kinematic_offsets = KinematicOffsets(
    kinematic_initial_orientation_offset_e=torch.tensor(
        torch.tensor(torch.pi / 2, device=device), device=device
    )
)

heliostat1_kinematic_config = KinematicConfig(
    kinematic_type=config_dictionary.rigid_body_key,
    kinematic_initial_orientation_offsets=heliostat1_kinematic_offsets,
)

# Include actuators for the heliostat.
# Include an ideal actuator.
actuator1_heliostat1 = ActuatorConfig(
    actuator_key="actuator1",
    actuator_type=config_dictionary.ideal_actuator_key,
    actuator_clockwise=False,
)

# Include a linear actuator.
actuator2_heliostat1 = ActuatorConfig(
    actuator_key="actuator2",
    actuator_type=config_dictionary.ideal_actuator_key,
    actuator_clockwise=True,
)
actuator_heliostat1_list = [actuator1_heliostat1, actuator2_heliostat1]
heliostat1_actuator_config = ActuatorListConfig(actuator_list=actuator_heliostat1_list)

# Include the configuration for a heliostat.
heliostat_1 = HeliostatConfig(
    heliostat_key="heliostat1",
    heliostat_id=1,
    heliostat_position=torch.tensor([0.0, 5.0, 0.0, 1.0], device=device),
    heliostat_aim_point=torch.tensor([0.0, -50.0, 0.0, 1.0], device=device),
    heliostat_surface=heliostat1_surface_config,
    heliostat_kinematic=heliostat1_kinematic_config,
    heliostat_actuator=heliostat1_actuator_config,
)

# Create a list of all the heliostats -- in this case, only one.
heliostat_list = [heliostat_1]

# Create the configuration for all heliostats.
heliostats_list_config = HeliostatListConfig(heliostat_list=heliostat_list)

if __name__ == "__main__":
    """Generate the scenario given the defined parameters."""
    # Create a scenario object.
    scenario_object = ScenarioGenerator(
        file_path=file_path,
        power_plant_config=power_plant_config,
        receiver_list_config=receiver_list_config,
        light_source_list_config=light_source_list_config,
        prototype_config=prototype_config,
        heliostat_list_config=heliostats_list_config,
    )

    # Generate the scenario.
    scenario_object.generate_scenario()
