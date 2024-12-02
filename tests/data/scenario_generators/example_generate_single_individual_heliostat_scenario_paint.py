import json
import pathlib

import torch

from artist import ARTIST_ROOT
from artist.util import config_dictionary, set_logger_config, utils
from artist.util.configuration_classes import (
    ActuatorConfig,
    ActuatorListConfig,
    ActuatorParameters,
    ActuatorPrototypeConfig,
    HeliostatConfig,
    HeliostatListConfig,
    KinematicConfig,
    KinematicDeviations,
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
    TowerAreaConfig,
    TowerAreaListConfig,
)
from artist.util.paint_to_surface_converter import PAINTToSurfaceConverter
from artist.util.scenario_generator2 import ScenarioGenerator

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set up logger
set_logger_config()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# The following parameter is the name of the scenario.
file_path = pathlib.Path(ARTIST_ROOT) / "tests/data/test_scenario_single_heliostat_paint"

if not pathlib.Path(file_path).parent.is_dir():
    raise FileNotFoundError(
        f"The folder ``{pathlib.Path(file_path).parent}`` selected to save the scenario does not exist."
        "Please create the folder or adjust the file path before running again!"
    )

heliostat_name = "AA39"
tower_file = pathlib.Path(ARTIST_ROOT) / "tests/data/download_test/tower.json"
heliostat_file = pathlib.Path(ARTIST_ROOT) / "tests/data/download_test/AA39/heliostat_properties.json"
deflectometry_file = pathlib.Path(ARTIST_ROOT) / "tests/data/download_test/AA39/deflectometry.h5"

# Extract all tower areas (calibration targets and receivers)
with open(tower_file, "r") as file:
    tower_dict = json.load(file)
    power_plant_position = torch.tensor(
        tower_dict[config_dictionary.power_plant_properties_key][config_dictionary.coordinates],
        dtype=torch.float64,
        device=device,
    )
    tower_areas = list(tower_dict.keys())[1:]

    tower_areas_configs_list = []
    
    for tower_area_key in tower_areas:
        area_type = tower_dict[tower_area_key][config_dictionary.paint_tower_area_type]

        center_lat_lon = torch.tensor(
            tower_dict[tower_area_key][config_dictionary.coordinates]["center"],
            dtype=torch.float64,
            device=device,
        )
        center_3d = utils.convert_wgs84_coordinates_to_local_enu(
            center_lat_lon, power_plant_position, device=device
        )
        center = utils.convert_3d_points_to_4d_format(
            center_3d, device=device
        )
        normal_vector = utils.convert_3d_direction_to_4d_format(
            torch.tensor(tower_dict[tower_area_key][config_dictionary.tower_area_normal_vector], device=device),
            device=device,
        )
        prefix= ""
        if tower_area_key == "receiver":
            prefix = "receiver_outer_"
        upper_left = utils.convert_wgs84_coordinates_to_local_enu(
            torch.tensor(
                tower_dict[tower_area_key][config_dictionary.coordinates][f"{prefix}upper_left"],
                dtype=torch.float64,
                device=device,
            ),
            power_plant_position,
            device=device,
        )
        lower_left = utils.convert_wgs84_coordinates_to_local_enu(
            torch.tensor(
                tower_dict[tower_area_key][config_dictionary.coordinates][f"{prefix}lower_left"],
                dtype=torch.float64,
                device=device,
            ),
            power_plant_position,
            device=device,
        )
        upper_right = utils.convert_wgs84_coordinates_to_local_enu(
            torch.tensor(
                tower_dict[tower_area_key][config_dictionary.coordinates][f"{prefix}upper_right"],
                dtype=torch.float64,
                device=device,
            ),
            power_plant_position,
            device=device,
        )
        lower_right = utils.convert_wgs84_coordinates_to_local_enu(
            torch.tensor(
                tower_dict[tower_area_key][config_dictionary.coordinates][f"{prefix}lower_right"],
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

        tower_area_config = TowerAreaConfig(tower_area_key=tower_area_key,
                                              area_type=area_type,
                                              center=center,
                                              normal_vector=normal_vector,
                                              plane_e=plane_e,
                                              plane_u=plane_u)

        tower_areas_configs_list.append(tower_area_config)

# Include the power plant configuration.
power_plant_config = PowerPlantConfig(
    power_plant_position=power_plant_position
)

# Include the tower area configurations.
tower_area_list_config = TowerAreaListConfig(tower_areas_configs_list)

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

# Include heliostat
with open(heliostat_file, "r") as file:
    heliostat_dict = json.load(file)
    heliostat_position_3d = utils.convert_wgs84_coordinates_to_local_enu(
        torch.tensor(
            heliostat_dict[config_dictionary.heliostat_position], dtype=torch.float64, device=device
        ),
        power_plant_position,
        device=device,
    )
    heliostat_position = utils.convert_3d_points_to_4d_format(
        heliostat_position_3d, device=device
    )

# Generate surface configuration from STRAL data.
paint_converter = PAINTToSurfaceConverter(
    deflectometry_file_path=deflectometry_file,
    heliostat_file_path=heliostat_file,
    step_size=100,
)

facets_list = paint_converter.generate_surface_config_from_paint(
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

# Generate the surface prototype configuration.
surface_prototype_config = SurfacePrototypeConfig(facets_list=facets_list)

# Include kinematic deviations.
kinematic_prototype_deviations = KinematicDeviations(
    first_joint_translation_e=torch.tensor(heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][config_dictionary.first_joint_translation_e], device=device),
    first_joint_translation_n=torch.tensor(heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][config_dictionary.first_joint_translation_n], device=device),
    first_joint_translation_u=torch.tensor(heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][config_dictionary.first_joint_translation_u], device=device),
    first_joint_tilt_e=torch.tensor(0.0, device=device),
    first_joint_tilt_n=torch.tensor(0.0, device=device),
    first_joint_tilt_u=torch.tensor(0.0, device=device),
    second_joint_translation_e=torch.tensor(heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][config_dictionary.second_joint_translation_e], device=device),
    second_joint_translation_n=torch.tensor(heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][config_dictionary.second_joint_translation_n], device=device),
    second_joint_translation_u=torch.tensor(heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][config_dictionary.second_joint_translation_u], device=device),
    second_joint_tilt_e=torch.tensor(0.0, device=device),
    second_joint_tilt_n=torch.tensor(0.0, device=device),
    second_joint_tilt_u=torch.tensor(0.0, device=device),
    concentrator_translation_e=torch.tensor(heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][config_dictionary.concentrator_translation_e], device=device),
    concentrator_translation_n=torch.tensor(heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][config_dictionary.concentrator_translation_n], device=device),
    concentrator_translation_u=torch.tensor(heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][config_dictionary.concentrator_translation_u], device=device),
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

# Include actuator parameters for actuator 1.
index = 1
# actuator1_parameters = ActuatorParameters(
#     increment=torch.tensor(heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][f"{config_dictionary.paint_increment}_{index}"], device=device),
#     initial_stroke_length=torch.tensor(heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][f"{config_dictionary.paint_initial_stroke_length}_{index}"], device=device),
#     offset=torch.tensor(heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][f"{config_dictionary.paint_offset}_{index}"], device=device),
#     pivot_radius=torch.tensor(heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][f"{config_dictionary.paint_pivot_radius}_{index}"], device=device),
#     initial_angle=torch.tensor(heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][f"{config_dictionary.paint_initial_angle}_{index}"], device=device),
# )
# Include an actuator 1.
actuator1_parameters = ActuatorParameters(
    increment=torch.tensor(154166.666, device=device),
    initial_stroke_length=torch.tensor(0.075, device=device),
    offset=torch.tensor(0.34061, device=device),
    pivot_radius=torch.tensor(0.3204, device=device),
    initial_angle=torch.tensor(-1.570796, device=device),
)
actuator1_prototype = ActuatorConfig(
    actuator_key=f"{config_dictionary.actuator_key}_{index}",
    actuator_type=heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][f"{config_dictionary.paint_actuator_type}_{index}"].lower(),
    actuator_clockwise=heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][f"{config_dictionary.paint_clockwise}_{index}"],
    actuator_parameters=actuator1_parameters,
)

index = 2
actuator2_parameters = ActuatorParameters(
    increment=torch.tensor(154166.666, device=device),
    initial_stroke_length=torch.tensor(0.075, device=device),
    offset=torch.tensor(0.3479, device=device),
    pivot_radius=torch.tensor(0.309, device=device),
    initial_angle=torch.tensor(0.959931, device=device),
)
# actuator2_parameters = ActuatorParameters(
#     increment=torch.tensor(heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][f"{config_dictionary.paint_increment}_{index}"], device=device),
#     initial_stroke_length=torch.tensor(heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][f"{config_dictionary.paint_initial_stroke_length}_{index}"], device=device),
#     offset=torch.tensor(heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][f"{config_dictionary.paint_offset}_{index}"], device=device),
#     pivot_radius=torch.tensor(heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][f"{config_dictionary.paint_pivot_radius}_{index}"], device=device),
#     initial_angle=torch.tensor(heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][f"{config_dictionary.paint_initial_angle}_{index}"], device=device),
# )
# Include an actuator 2.
actuator2_prototype = ActuatorConfig(
    actuator_key=f"{config_dictionary.actuator_key}_{index}",
    actuator_type=heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][f"{config_dictionary.paint_actuator_type}_{index}"].lower(),
    actuator_clockwise=heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][f"{config_dictionary.paint_clockwise}_{index}"],
    actuator_parameters=actuator2_parameters,
)

# Create a list of actuators.
actuator_prototype_list = [actuator1_prototype, actuator2_prototype]

# Include the actuator prototype config.
actuators_prototype_config = ActuatorPrototypeConfig(
    actuator_list=actuator_prototype_list
)

# Include the final prototype config.
prototype_config = PrototypeConfig(
    surface_prototype=surface_prototype_config,
    kinematic_prototype=kinematic_prototype_config,
    actuators_prototype=actuators_prototype_config,
)

# Include individual heliostat parameters (-> protoype technically not needed)
# In this example the individual heliostat is the same as the prototype
heliostat_surface_config = SurfaceConfig(facets_list=facets_list)
heliostat_kinematic_config = KinematicConfig(
    kinematic_type=config_dictionary.rigid_body_key,
    kinematic_initial_orientation_offsets=kinematic_prototype_offsets,
    kinematic_deviations=kinematic_prototype_deviations
)
heliostat_actuators_config = ActuatorListConfig(actuator_list=actuator_prototype_list)


# Choose default aimpoint for heliostat
heliostats_aimpoint_area = "receiver"
heliostats_aimpoint = next(tower_area for tower_area in tower_areas_configs_list if tower_area.tower_area_key == heliostats_aimpoint_area).center

# Include the configuration for a heliostat.
heliostat = HeliostatConfig(
    heliostat_key=heliostat_name,
    heliostat_id=1,
    heliostat_position=heliostat_position,
    heliostat_aim_point=heliostats_aimpoint,
    heliostat_surface = heliostat_surface_config,
    heliostat_kinematic = heliostat_kinematic_config,
    heliostat_actuators = heliostat_actuators_config
)

# Create a list of all the heliostats - in this case, only one.
heliostat_list = [heliostat]

# Create the configuration for all heliostats.
heliostats_list_config = HeliostatListConfig(heliostat_list=heliostat_list)

if __name__ == "__main__":
    """Generate the scenario given the defined parameters."""
    # Create a scenario object.
    scenario_object = ScenarioGenerator(
        file_path=file_path,
        power_plant_config=power_plant_config,
        tower_area_list_config=tower_area_list_config,
        light_source_list_config=light_source_list_config,
        prototype_config=prototype_config,
        heliostat_list_config=heliostats_list_config,
    )

    # Generate the scenario.
    scenario_object.generate_scenario()
