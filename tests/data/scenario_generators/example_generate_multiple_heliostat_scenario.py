import json
import pathlib

import torch

from artist import ARTIST_ROOT
from artist.util import config_dictionary, utils
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
    PowerPlantConfig,
    PrototypeConfig,
    ReceiverConfig,
    ReceiverListConfig,
    SurfacePrototypeConfig,
    TargetAreaConfig,
)
from artist.util.scenario_generator import ScenarioGenerator

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# The following parameter is the name of the scenario.
file_path = file_path = (
    pathlib.Path(ARTIST_ROOT) / "tests/data/four_heliostat_scenario"
)

if not pathlib.Path(file_path).parent.is_dir():
    raise FileNotFoundError(
        f"The folder ``{pathlib.Path(file_path).parent}`` selected to save the scenario does not exist. "
        "Please create the folder or adjust the file path before running again!"
    )

tower_file = pathlib.Path(ARTIST_ROOT) / "tests/data/download_test/tower.json"

with open(tower_file, "r") as file:
    tower_dict = json.load(file)
    power_plant_position = torch.tensor(
        tower_dict["power_plant_properties"]["coordinates"],
        dtype=torch.float64,
        device=device,
    )
    target_areas = list(tower_dict.keys())[1:]

    target_areas_configs_list = []
    
    for target_area_key in target_areas:
        target_area_type = tower_dict[target_area_key]["type"]

        center_lat_lon = torch.tensor(
            tower_dict[target_area_key]["coordinates"]["center"],
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
            torch.tensor(tower_dict[target_area_key]["normal_vector"], device=device),
            device=device,
        )
        upper_left = utils.convert_wgs84_coordinates_to_local_enu(
            torch.tensor(
                tower_dict[target_area_key]["coordinates"]["upper_left"],
                dtype=torch.float64,
                device=device,
            ),
            power_plant_position,
            device=device,
        )
        lower_left = utils.convert_wgs84_coordinates_to_local_enu(
            torch.tensor(
                tower_dict[target_area_key]["coordinates"]["lower_left"],
                dtype=torch.float64,
                device=device,
            ),
            power_plant_position,
            device=device,
        )
        upper_right = utils.convert_wgs84_coordinates_to_local_enu(
            torch.tensor(
                tower_dict[target_area_key]["coordinates"]["upper_right"],
                dtype=torch.float64,
                device=device,
            ),
            power_plant_position,
            device=device,
        )
        lower_right = utils.convert_wgs84_coordinates_to_local_enu(
            torch.tensor(
                tower_dict[target_area_key]["coordinates"]["lower_right"],
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

        target_area_config = TargetAreaConfig(target_area_key=target_area_key,
                                              target_area_type=target_area_type,
                                              center=center,
                                              normal_vector=normal_vector,
                                              plane_e=plane_e,
                                              plane_u=plane_u)

        target_areas_configs_list.append(target_area_config)

# Include the power plant configuration.
power_plant_config = PowerPlantConfig(
    power_plant_position=torch.tensor([0.0, 0.0, 0.0], device=device),
    target_areas_configs_list=target_areas_configs_list
)

heliostats = ["AA31", "AA35", "AA39", "AB38"]

for heliostat in heliostats:
    heliostat_file = pathlib.Path(ARTIST_ROOT) / f"tests/data/download_test/{heliostat}/heliostat_properties.json"
    deflectometry_file = pathlib.Path(ARTIST_ROOT) / f"tests/data/download_test/{heliostat}/deflectometry.h5"

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
        power_plant_config=power_plant_config,
        receiver_list_config=receiver_list_config,
        light_source_list_config=light_source_list_config,
        prototype_config=prototype_config,
        heliostat_list_config=heliostats_list_config,
    )

    # Generate the scenario.
    scenario_object.generate_scenario()
