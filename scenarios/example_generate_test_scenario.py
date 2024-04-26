import math

import torch

from artist.util import config_dictionary
from artist.util.scenario_generator import (
    ActuatorPrototypeConfig,
    FacetConfig,
    HeliostatConfig,
    HeliostatListConfig,
    KinematicOffsets,
    KinematicPrototypeConfig,
    LightSourceConfig,
    LightSourceListConfig,
    PrototypeConfig,
    ReceiverConfig,
    ReceiverListConfig,
    ScenarioGenerator,
    SurfacePrototypeConfig,
)

# Include the receiver configuration.
receiver1_config = ReceiverConfig(
    receiver_key="receiver1",
    receiver_type=config_dictionary.receiver_type_planar,
    position_center=torch.tensor([0.0, -50.0, 0.0, 1.0]),
    normal_vector=torch.tensor([0.0, 1.0, 0.0, 0.0]),
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

# Include the configuration fo the list of light sources.
light_source_list_config = LightSourceListConfig(light_source_list=light_source_list)


# Include four facets for the surface prototype
prototype_facet1_config = FacetConfig(
    facet_key="facet1",
    control_points_e=torch.Tensor([0.0, 0.0, 0.0, 0.0]),
    control_points_u=torch.Tensor([0.0, 0.0, 0.0, 0.0]),
    knots_e=torch.Tensor([0.0, 0.0, 0.0, 0.0]),
    knots_u=torch.Tensor([0.0, 0.0, 0.0, 0]),
    width=25.0,
    height=25.0,
    position=torch.tensor([0.0, 0.0, 0.0, 1.0]),
    canting_e=torch.Tensor([0.0, 0.0, 0.0, 0.0]),
    canting_u=torch.Tensor([0.0, 0.0, 0.0, 0.0]),
)
prototype_facet2_config = FacetConfig(
    facet_key="facet2",
    control_points_e=torch.Tensor([0.0, 0.0, 0.0, 0.0]),
    control_points_u=torch.Tensor([0.0, 0.0, 0.0, 0.0]),
    knots_e=torch.Tensor([0.0, 0.0, 0.0, 0.0]),
    knots_u=torch.Tensor([0.0, 0.0, 0.0, 0]),
    width=25.0,
    height=25.0,
    position=torch.tensor([0.0, 0.0, 0.0, 1.0]),
    canting_e=torch.Tensor([0.0, 0.0, 0.0, 0.0]),
    canting_u=torch.Tensor([0.0, 0.0, 0.0, 0.0]),
)
prototype_facet3_config = FacetConfig(
    facet_key="facet3",
    control_points_e=torch.Tensor([0.0, 0.0, 0.0, 0.0]),
    control_points_u=torch.Tensor([0.0, 0.0, 0.0, 0.0]),
    knots_e=torch.Tensor([0.0, 0.0, 0.0, 0.0]),
    knots_u=torch.Tensor([0.0, 0.0, 0.0, 0]),
    width=25.0,
    height=25.0,
    position=torch.tensor([0.0, 0.0, 0.0, 1.0]),
    canting_e=torch.Tensor([0.0, 0.0, 0.0, 0.0]),
    canting_u=torch.Tensor([0.0, 0.0, 0.0, 0.0]),
)
prototype_facet4_config = FacetConfig(
    facet_key="facet4",
    control_points_e=torch.Tensor([0.0, 0.0, 0.0, 0.0]),
    control_points_u=torch.Tensor([0.0, 0.0, 0.0, 0.0]),
    knots_e=torch.Tensor([0.0, 0.0, 0.0, 0.0]),
    knots_u=torch.Tensor([0.0, 0.0, 0.0, 0]),
    width=25.0,
    height=25.0,
    position=torch.tensor([0.0, 0.0, 0.0, 1.0]),
    canting_e=torch.Tensor([0.0, 0.0, 0.0, 0.0]),
    canting_u=torch.Tensor([0.0, 0.0, 0.0, 0.0]),
)

# Create a list of prototype facets
prototype_facets_list = [
    prototype_facet1_config,
    prototype_facet2_config,
    prototype_facet3_config,
    prototype_facet4_config,
]

# Include the facet prototype config.
surface_prototype_config = SurfacePrototypeConfig(facets_list=prototype_facets_list)

# Include the kinematic deviations.
# kinematic_prototype_deviations = KinematicDeviations() #Here there are no kinematic deviations!

# Include the initial orientation offsets for the kinematic.
kinematic_prototype_offsets = KinematicOffsets(
    kinematic_initial_orientation_offset_e=math.pi / 2
)

# Include the kinematic prototype configuration
kinematic_prototype_config = KinematicPrototypeConfig(
    kinematic_type=config_dictionary.rigid_body_key,
    kinematic_initial_orientation_offsets=kinematic_prototype_offsets,
)

# Include the deviations for the actuator prototype.
# actuator_prototype_parameters = ActuatorParameters() # Here commented due to ideal actuator without parameters

# Include the Actuator prototype config.
actuator_prototype_config = ActuatorPrototypeConfig(
    actuator_type=config_dictionary.ideal_actuator_key
)

# Include the final prototype config.
prototype_config = PrototypeConfig(
    surface_prototype=surface_prototype_config,
    kinematic_prototype=kinematic_prototype_config,
    actuator_prototype=actuator_prototype_config,
)

# If heliostats had individual surface, kinematic, or actuator configs, these must be defined here. Otherwise, only
# the parameters from the prototype are used.

# Include the configuration for a heliostat
heliostat_1 = HeliostatConfig(
    heliostat_key="heliostat1",
    heliostat_id=1,
    heliostat_position=torch.tensor([0.0, 5.0, 0.0, 1.0]),
    heliostat_aim_point=torch.tensor([0.0, -50.0, 0.0, 1.0]),
)

# Create a list of all the heliostats -- in this case, only one.
heliostat_list = [heliostat_1]

# Create the configuration for all heliostats.
heliostats_list_config = HeliostatListConfig(heliostat_list=heliostat_list)


# The following parameter is the name of the scenario.
file_path = "./test_scenario"

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
