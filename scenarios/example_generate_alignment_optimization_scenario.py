import math
from pathlib import Path

import json
from artist.util.paint_to_surface_converter import PAINTToSurfaceConverter
import torch

from artist import ARTIST_ROOT
from artist.util import config_dictionary
from artist.util.configuration_classes import (
    ActuatorConfig,
    ActuatorParameters,
    ActuatorPrototypeConfig,
    HeliostatConfig,
    HeliostatListConfig,
    KinematicDeviations,
    KinematicOffsets,
    KinematicPrototypeConfig,
    LightSourceConfig,
    LightSourceListConfig,
    PrototypeConfig,
    ReceiverConfig,
    ReceiverListConfig,
    SurfacePrototypeConfig,
)
from artist.util.scenario_generator import ScenarioGenerator
from artist.util.stral_to_surface_converter import StralToSurfaceConverter
from artist.util import utils

# The following parameter is the name of the scenario.
file_path = "scenarios/test_alignment_optimization"

if not Path(file_path).parent.is_dir():
    raise FileNotFoundError(
        f"The folder ``{Path(file_path).parent}`` selected to save the scenario does not exist. "
        "Please create the folder or adjust the file path before running again!"
    )

# TODO eigentlich sollten hier paths zu PAINT sein?
tower_file = f"{ARTIST_ROOT}/measurement_data/juelich-tower-measurements.json"
calibration_file = f"{ARTIST_ROOT}/measurement_data/AA39/Calibration/86500-calibration-properties.json"
heliostat_file = f"{ARTIST_ROOT}/measurement_data/AA39/Properties/AA39-heliostat_properties.json"
deflectometry_file = f"{ARTIST_ROOT}/measurement_data/AA39/Deflectometry/AA39-filled-2023-09-18Z08_49_09Z-deflectometry.h5"

with open(calibration_file, 'r') as file:
    calibration_dict = json.load(file)
    target_name = calibration_dict["target_name"]

with open(tower_file, 'r') as file:
    tower_dict = json.load(file)
    target_type = tower_dict[target_name][config_dictionary.receiver_type]
    power_plant_coordinates = tower_dict["power_plant_properties"]["coordinates"]
    position_center_lat_lon = tower_dict[target_name]["coordinates"]["center"]
    position_center_3d = utils.calculate_position_in_m_from_lat_lon(position_center_lat_lon, power_plant_coordinates)
    position_center = utils.convert_3d_points_to_4d_format(position_center_3d)
    normal_vector = utils.convert_3d_direction_to_4d_format(torch.tensor(tower_dict[target_name]["normal_vector"]))

# Include the receiver configuration.
receiver1_config = ReceiverConfig(
    receiver_key="receiver1",
    receiver_type=target_type,
    position_center=position_center,
    normal_vector=normal_vector,
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
paint_converter = PAINTToSurfaceConverter(
    deflectometry_file_path=deflectometry_file,
    heliostat_file_path=heliostat_file,
    step_size=100,
)

facet_prototype_list = paint_converter.generate_surface_config_from_paint(
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
)

surface_prototype_config = SurfacePrototypeConfig(facets_list=facet_prototype_list)

# Include kinematic deviations.
kinematic_prototype_deviations = KinematicDeviations(
    first_joint_translation_e=torch.tensor(0.0),
    first_joint_translation_n=torch.tensor(0.0),
    first_joint_translation_u=torch.tensor(0.0),
    first_joint_tilt_e=torch.tensor(0.0),
    first_joint_tilt_n=torch.tensor(0.0),
    first_joint_tilt_u=torch.tensor(0.0),
    second_joint_translation_e=torch.tensor(0.0),
    second_joint_translation_n=torch.tensor(0.0),
    second_joint_translation_u=torch.tensor(0.0),
    second_joint_tilt_e=torch.tensor(0.0),
    second_joint_tilt_n=torch.tensor(0.0),
    second_joint_tilt_u=torch.tensor(0.0),
    concentrator_translation_e=torch.tensor(0.0),
    concentrator_translation_n=torch.tensor(0.0),
    concentrator_translation_u=torch.tensor(0.0),
    concentrator_tilt_e=torch.tensor(0.0),
    concentrator_tilt_n=torch.tensor(0.0),
    concentrator_tilt_u=torch.tensor(0.0),
)

# Include the initial orientation offsets for the kinematic.
kinematic_prototype_offsets = KinematicOffsets(
    kinematic_initial_orientation_offset_e=torch.tensor(math.pi / 2)
)

# Include the kinematic prototype configuration.
kinematic_prototype_config = KinematicPrototypeConfig(
    kinematic_type=config_dictionary.rigid_body_key,
    kinematic_initial_orientation_offsets=kinematic_prototype_offsets,
    kinematic_deviations=kinematic_prototype_deviations
)

# Include actuator parameters for both actuators.
actuator1_parameters = ActuatorParameters(
    increment=torch.tensor(0.0),
    initial_stroke_length=torch.tensor(0.0),
    offset=torch.tensor(0.0),
    radius=torch.tensor(0.0),
    phi_0=torch.tensor(0.0),
)

actuator2_parameters = ActuatorParameters(
    increment=torch.tensor(0.0),
    initial_stroke_length=torch.tensor(0.0),
    offset=torch.tensor(0.0),
    radius=torch.tensor(0.0),
    phi_0=torch.tensor(0.0),
)

# Include a linear actuator.
actuator1_prototype = ActuatorConfig(
    actuator_key="actuator1",
    actuator_type=config_dictionary.linear_actuator_key,
    actuator_clockwise=False,
    actuator_parameters=actuator1_parameters,
)

# Include a linear actuator.
actuator2_prototype = ActuatorConfig(
    actuator_key="actuator2",
    actuator_type=config_dictionary.linear_actuator_key,
    actuator_clockwise=True,
    actuator_parameters=actuator2_parameters,
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

# Note, we do not include individual heliostat parameters in this scenario.

# Include the configuration for a heliostat.
heliostat1 = HeliostatConfig(
    heliostat_key="heliostat1",
    heliostat_id=1,
    heliostat_position=torch.tensor([0.0, 5.0, 0.0, 1.0]),
    heliostat_aim_point=torch.tensor([0.0, -50.0, 0.0, 1.0]),
)

# Create a list of all the heliostats - in this case, only one.
heliostat_list = [heliostat1]

# Create the configuration for all heliostats.
heliostats_list_config = HeliostatListConfig(heliostat_list=heliostat_list)


if __name__ == "__main__":
    """Generate the scenario given the defined parameters."""

    # Create a scenario object.
    scenario_object = ScenarioGenerator(
        file_path=file_path,
        receiver_list_config=receiver_list_config,
        light_source_list_config=light_source_list_config,
        prototype_config=prototype_config,
        heliostat_list_config=heliostats_list_config,
    )

    # Generate the scenario.
    scenario_object.generate_scenario()