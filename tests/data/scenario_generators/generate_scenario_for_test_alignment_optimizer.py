import json
import pathlib

import torch

from artist import ARTIST_ROOT
from artist.util import config_dictionary, utils
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
    PowerPlantConfig,
    PrototypeConfig,
    ReceiverConfig,
    ReceiverListConfig,
    SurfacePrototypeConfig,
)
from artist.util.paint_to_surface_converter import PAINTToSurfaceConverter
from artist.util.scenario_generator import ScenarioGenerator

torch.manual_seed(7)
torch.cuda.manual_seed(7)

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# The following parameter is the name of the scenario.
file_path = pathlib.Path(ARTIST_ROOT) / "tests/data/test_scenario_paint"

if not pathlib.Path(file_path).parent.is_dir():
    raise FileNotFoundError(
        f"The folder ``{pathlib.Path(file_path).parent}`` selected to save the scenario does not exist."
        "Please create the folder or adjust the file path before running again!"
    )

tower_file = pathlib.Path(ARTIST_ROOT) / "tests/data/tower.json"
calibration_file = pathlib.Path(ARTIST_ROOT) / "tests/data/calibration_properties.json"
heliostat_file = pathlib.Path(ARTIST_ROOT) / "tests/data/heliostat_properties.json"
deflectometry_file = pathlib.Path(ARTIST_ROOT) / "tests/data/deflectometry.h5"

with open(calibration_file, "r") as file:
    calibration_dict = json.load(file)
    target_name = calibration_dict["target_name"]

with open(tower_file, "r") as file:
    tower_dict = json.load(file)
    target_type = tower_dict[target_name][config_dictionary.receiver_type]
    power_plant_position = torch.tensor(
        tower_dict["power_plant_properties"]["coordinates"],
        dtype=torch.float64,
        device=device,
    )
    target_center_lat_lon = torch.tensor(
        tower_dict[target_name]["coordinates"]["center"],
        dtype=torch.float64,
        device=device,
    )
    target_center_3d = utils.convert_wgs84_coordinates_to_local_enu(
        target_center_lat_lon, power_plant_position, device=device
    )
    target_center = utils.convert_3d_points_to_4d_format(
        target_center_3d, device=device
    )
    normal_vector = utils.convert_3d_direction_to_4d_format(
        torch.tensor(tower_dict[target_name]["normal_vector"], device=device),
        device=device,
    )
    upper_left = utils.convert_wgs84_coordinates_to_local_enu(
        torch.tensor(
            tower_dict[target_name]["coordinates"]["upper_left"],
            dtype=torch.float64,
            device=device,
        ),
        power_plant_position,
        device=device,
    )
    lower_left = utils.convert_wgs84_coordinates_to_local_enu(
        torch.tensor(
            tower_dict[target_name]["coordinates"]["lower_left"],
            dtype=torch.float64,
            device=device,
        ),
        power_plant_position,
        device=device,
    )
    upper_right = utils.convert_wgs84_coordinates_to_local_enu(
        torch.tensor(
            tower_dict[target_name]["coordinates"]["upper_right"],
            dtype=torch.float64,
            device=device,
        ),
        power_plant_position,
        device=device,
    )
    lower_right = utils.convert_wgs84_coordinates_to_local_enu(
        torch.tensor(
            tower_dict[target_name]["coordinates"]["lower_right"],
            dtype=torch.float64,
            device=device,
        ),
        power_plant_position,
        device=device,
    )
    plane_e = (
        torch.abs(upper_right[0] - upper_left[0])
        + torch.abs(lower_right[0] - lower_left[0])
    ) / 2
    plane_u = (
        torch.abs(upper_left[2] - lower_left[2])
        + torch.abs(upper_right[2] - lower_right[2])
    ) / 2

with open(heliostat_file, "r") as file:
    heliostat_dict = json.load(file)
    heliostat_position_3d = utils.convert_wgs84_coordinates_to_local_enu(
        torch.tensor(
            heliostat_dict["heliostat_position"], dtype=torch.float64, device=device
        ),
        power_plant_position,
        device=device,
    )
    heliostat_position = utils.convert_3d_points_to_4d_format(
        heliostat_position_3d, device=device
    )


# Include the power plant configuration.
power_plant_config = PowerPlantConfig(power_plant_position=power_plant_position)

# Include the receiver configuration.
receiver1_config = ReceiverConfig(
    receiver_key="receiver1",
    receiver_type=target_type,
    position_center=target_center,
    normal_vector=normal_vector,
    plane_e=plane_e,
    plane_u=plane_u,
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
    number_of_rays=1,
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
    device=device,
)

surface_prototype_config = SurfacePrototypeConfig(facets_list=facet_prototype_list)

# Include kinematic deviations.
kinematic_prototype_deviations = KinematicDeviations(
    first_joint_translation_e=torch.tensor(0.0, device=device),
    first_joint_translation_n=torch.tensor(0.0, device=device),
    first_joint_translation_u=torch.tensor(0.0, device=device),
    first_joint_tilt_e=torch.tensor(0.0, device=device),
    first_joint_tilt_n=torch.tensor(0.0, device=device),
    first_joint_tilt_u=torch.tensor(0.0, device=device),
    second_joint_translation_e=torch.tensor(0.0, device=device),
    second_joint_translation_n=torch.tensor(0.0, device=device),
    second_joint_translation_u=torch.tensor(0.315, device=device),
    second_joint_tilt_e=torch.tensor(0.0, device=device),
    second_joint_tilt_n=torch.tensor(0.0, device=device),
    second_joint_tilt_u=torch.tensor(0.0, device=device),
    concentrator_translation_e=torch.tensor(0.0, device=device),
    concentrator_translation_n=torch.tensor(-0.17755, device=device),
    concentrator_translation_u=torch.tensor(-0.4045, device=device),
    concentrator_tilt_e=torch.tensor(0.0, device=device),
    concentrator_tilt_n=torch.tensor(0.0, device=device),
    concentrator_tilt_u=torch.tensor(0.0, device=device),
)

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
    kinematic_deviations=kinematic_prototype_deviations,
)

# Include actuator parameters for both actuators.
actuator1_parameters = ActuatorParameters(
    increment=torch.tensor(154166.666, device=device),
    initial_stroke_length=torch.tensor(0.075, device=device),
    offset=torch.tensor(0.34061, device=device),
    radius=torch.tensor(0.3204, device=device),
    phi_0=torch.tensor(-1.570796, device=device),
)

actuator2_parameters = ActuatorParameters(
    increment=torch.tensor(154166.666, device=device),
    initial_stroke_length=torch.tensor(0.075, device=device),
    offset=torch.tensor(0.3479, device=device),
    radius=torch.tensor(0.309, device=device),
    phi_0=torch.tensor(0.959931, device=device),
)

# Include an ideal actuator.
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
    heliostat_position=heliostat_position,
    heliostat_aim_point=target_center,
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
        power_plant_config=power_plant_config,
        receiver_list_config=receiver_list_config,
        light_source_list_config=light_source_list_config,
        prototype_config=prototype_config,
        heliostat_list_config=heliostats_list_config,
    )

    # Generate the scenario.
    scenario_object.generate_scenario()
