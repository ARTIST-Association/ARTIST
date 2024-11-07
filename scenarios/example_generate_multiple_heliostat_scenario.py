import math
from pathlib import Path

import torch

from artist.util import config_dictionary
from artist.util.configuration_classes import (
    ActuatorConfig,
    ActuatorParameters,
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
    SurfacePrototypeConfig,
)
from artist.util.scenario_generator import ScenarioGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The following parameter is the name of the scenario.
file_path = "multiple_heliostat_scenario"

if not Path(file_path).parent.is_dir():
    raise FileNotFoundError(
        f"The folder ``{Path(file_path).parent}`` selected to save the scenario does not exist. "
        "Please create the folder or adjust the file path before running again!"
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


# Include four facets for the surface prototype.
prototype_facet1_config = FacetConfig(
    facet_key="facet1",
    control_points=torch.tensor([0.0, 0.0, 0.0, 0.0], device=device),
    degree_e=2,
    degree_n=2,
    number_eval_points_e=10,
    number_eval_points_n=10,
    width=25.0,
    height=25.0,
    translation_vector=torch.tensor([0.0, 0.0, 0.0, 1.0], device=device),
    canting_e=torch.tensor([0.0, 0.0, 0.0, 0.0], device=device),
    canting_n=torch.tensor([0.0, 0.0, 0.0, 0.0], device=device),
)
prototype_facet2_config = FacetConfig(
    facet_key="facet2",
    control_points=torch.tensor([0.0, 0.0, 0.0, 0.0], device=device),
    degree_e=2,
    degree_n=2,
    number_eval_points_e=10,
    number_eval_points_n=10,
    width=25.0,
    height=25.0,
    translation_vector=torch.tensor([0.0, 0.0, 0.0, 1.0], device=device),
    canting_e=torch.tensor([0.0, 0.0, 0.0, 0.0], device=device),
    canting_n=torch.tensor([0.0, 0.0, 0.0, 0.0], device=device),
)
prototype_facet3_config = FacetConfig(
    facet_key="facet3",
    control_points=torch.tensor([0.0, 0.0, 0.0, 0.0], device=device),
    degree_e=2,
    degree_n=2,
    number_eval_points_e=10,
    number_eval_points_n=10,
    width=25.0,
    height=25.0,
    translation_vector=torch.tensor([0.0, 0.0, 0.0, 1.0], device=device),
    canting_e=torch.tensor([0.0, 0.0, 0.0, 0.0], device=device),
    canting_n=torch.tensor([0.0, 0.0, 0.0, 0.0], device=device),
)
prototype_facet4_config = FacetConfig(
    facet_key="facet4",
    control_points=torch.tensor([0.0, 0.0, 0.0, 0.0], device=device),
    degree_e=2,
    degree_n=2,
    number_eval_points_e=10,
    number_eval_points_n=10,
    width=25.0,
    height=25.0,
    translation_vector=torch.tensor([0.0, 0.0, 0.0, 1.0], device=device),
    canting_e=torch.tensor([0.0, 0.0, 0.0, 0.0], device=device),
    canting_n=torch.tensor([0.0, 0.0, 0.0, 0.0], device=device),
)

# Create a list of prototype facets.
prototype_facets_list = [
    prototype_facet1_config,
    prototype_facet2_config,
    prototype_facet3_config,
    prototype_facet4_config,
]

# Include the facet prototype config.
surface_prototype_config = SurfacePrototypeConfig(facets_list=prototype_facets_list)

# Note, we do not include kinematic deviations in this scenario!

# Include the initial orientation offsets for the kinematic.
kinematic_prototype_offsets = KinematicOffsets(
    kinematic_initial_orientation_offset_e=torch.tensor(math.pi / 2, device=device)
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

# Include parameters for a linear actuator.
actuator2_prototype_parameters = ActuatorParameters(
    increment=torch.tensor(0.0, device=device),
    initial_stroke_length=torch.tensor(0.0, device=device),
    offset=torch.tensor(0.0, device=device),
    radius=torch.tensor(0.0, device=device),
    phi_0=torch.tensor(0.0, device=device),
)

# Include a linear actuator.
actuator2_prototype = ActuatorConfig(
    actuator_key="actuator2",
    actuator_type=config_dictionary.linear_actuator_key,
    actuator_clockwise=True,
    actuator_parameters=actuator2_prototype_parameters,
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

# Include the configuration for three heliostats.
heliostat1 = HeliostatConfig(
    heliostat_key="heliostat1",
    heliostat_id=1,
    heliostat_position=torch.tensor([-50.0, 5.0, 0.0, 1.0], device=device),
    heliostat_aim_point=torch.tensor([0.0, -50.0, 0.0, 1.0], device=device),
)
heliostat2 = HeliostatConfig(
    heliostat_key="heliostat2",
    heliostat_id=2,
    heliostat_position=torch.tensor([25.0, 0.0, 0.0, 1.0], device=device),
    heliostat_aim_point=torch.tensor([0.0, -50.0, 0.0, 1.0], device=device),
)
heliostat3 = HeliostatConfig(
    heliostat_key="heliostat3",
    heliostat_id=3,
    heliostat_position=torch.tensor([50.0, 5.0, 0.0, 1.0], device=device),
    heliostat_aim_point=torch.tensor([0.0, -50.0, 0.0, 1.0], device=device),
)

# Create a list of all the heliostats.
heliostat_list = [heliostat1, heliostat2, heliostat3]

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
