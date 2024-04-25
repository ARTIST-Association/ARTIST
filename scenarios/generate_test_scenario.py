import math

import h5py
import torch

from artist.util import config_dictionary
from artist.util.scenario_generator import (
    ActuatorDeviations,
    HeliostatListConfig,
    KinematicDeviations,
    KinematicOffsets,
    LightSourceConfig,
    ReceiverConfig,
    ScenarioGenerator,
    SingleHeliostatConfig,
)

# Include the receiver configuration.
receiver_config = ReceiverConfig(
    receiver_center=torch.tensor([0.0, -50.0, 0.0, 1.0]),
    plane_normal=torch.tensor([0.0, 1.0, 0.0, 0.0]),
    plane_x=8.629666667,
    plane_y=7.0,
    resolution_x=256,
    resolution_y=256,
)

# Include the light source configuration.
light_source_config = LightSourceConfig(
    sun_number_of_rays=10,
    sun_distribution_type=config_dictionary.sun_distribution_is_normal,
    sun_mean=0.0,
    sun_covariance=4.3681e-06,
)

# Include the kinematic deviations.
kinematic_deviations = KinematicDeviations(
    first_joint_translation_e=0.0,
    first_joint_translation_n=0.0,
    first_joint_translation_u=0.0,
    first_joint_tilt_e=0.0,
    first_joint_tilt_n=0.0,
    first_joint_tilt_u=0.0,
    second_joint_translation_e=0.0,
    second_joint_translation_n=0.0,
    second_joint_translation_u=0.0,
    second_joint_tilt_e=0.0,
    second_joint_tilt_n=0.0,
    second_joint_tilt_u=0.0,
    concentrator_translation_e=0.0,
    concentrator_translation_n=0.0,
    concentrator_translation_u=0.0,
    concentrator_tilt_e=0.0,
    concentrator_tilt_n=0.0,
    concentrator_tilt_u=0.0,
)

# Include the initial orientation offsets for the kinematic.
kinematic_offsets = KinematicOffsets(
    kinematic_initial_orientation_offset_e=math.pi / 2,
    kinematic_initial_orientation_offset_n=0.0,
    kinematic_initial_orientation_offset_u=0.0,
)

# Include the deviations for the actuator
actuator_deviations = ActuatorDeviations(
    first_joint_increment=0.0,
    first_joint_initial_stroke_length=0.0,
    first_joint_actuator_offset=0.0,
    first_joint_radius=0.0,
    first_joint_phi_0=0.0,
    second_joint_increment=0.0,
    second_joint_initial_stroke_length=0.0,
    second_joint_actuator_offset=0.0,
    second_joint_radius=0.0,
    second_joint_phi_0=0.0,
)

# Include the configuration for the first heliostat
heliostat_1 = SingleHeliostatConfig(
    heliostat_name="Single_Heliostat",
    heliostat_id=0,
    alignment_type=config_dictionary.rigid_body_key,
    actuator_type=config_dictionary.ideal_actuator_key,
    heliostat_position=torch.tensor([0.0, 5.0, 0.0, 1.0]),
    heliostat_aim_point=torch.tensor([0.0, -50.0, 0.0, 1.0]),
    facets_type=config_dictionary.point_cloud_facet_key,
    has_individual_surface_points=False,
    has_individual_surface_normals=False,
    heliostat_individual_surface_points=False,
    heliostat_individual_surface_normals=False,
    kinematic_deviations=kinematic_deviations,
    kinematic_offsets=kinematic_offsets,
    actuator_deviations=actuator_deviations,
)


# Create a list of all the heliostats -- in this case only one
all_heliostats = [heliostat_1]

# Load general surface points measurement
general_surface_points = torch.tensor(
    h5py.File(
        "../measurement_data/test_data.h5",
        "r",
    )[config_dictionary.load_points_key][()]
)

# Load general surface normals measurement
general_surface_normals = torch.tensor(
    h5py.File(
        "../measurement_data/test_data.h5",
        "r",
    )[config_dictionary.load_normals_key][()]
)

# Create the configuration for all heliostats
heliostats_list_config = HeliostatListConfig(
    general_surface_points=general_surface_points,
    general_surface_normals=general_surface_normals,
    heliostat_list=all_heliostats,
)


# The following parameter is the name of the scenario.
file_path = "./test_scenario"

if __name__ == "__main__":
    """Generate the scenario given the defined parameters."""

    # Create a scenario object
    scenario_object = ScenarioGenerator(
        file_path=file_path,
        receiver_config=receiver_config,
        light_source_config=light_source_config,
        heliostat_list_config=heliostats_list_config,
    )

    # Generate the scenario
    scenario_object.generate_scenario()
