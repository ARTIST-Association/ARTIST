import json
import pathlib

from artist.util.surface_converter import SurfaceConverter
import torch

from artist import ARTIST_ROOT
from artist.util import config_dictionary, paint_loader, utils
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


calibration_target_name  = paint_loader.read_paint_calibration_properties(calibration_file)

power_plant_position, target_type, target_center, normal_vector, plane_e, plane_u = paint_loader(tower_file, calibration_target_name, device)

heliostat_position, kinematic_deviations = paint_loader.read_paint_heliostat_properties(heliostat_file, power_plant_position, device)

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
surface_converter = SurfaceConverter(
    deflectometry_file_path=deflectometry_file,
    heliostat_file_path=heliostat_file,
    step_size=100,
    max_epoch=400,
)

facet_prototype_list = surface_converter.generate_surface_config(device=device)

surface_prototype_config = SurfacePrototypeConfig(facets_list=facet_prototype_list)

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
    kinematic_deviations=kinematic_deviations,
)


# Include actuator parameters for actuator 1.
index = 1
actuator1_parameters = ActuatorParameters(
    increment=torch.tensor(
        heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][
            f"{config_dictionary.paint_increment}_{index}"
        ],
        device=device,
    ),
    initial_stroke_length=torch.tensor(
        heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][
            f"{config_dictionary.paint_initial_stroke_length}_{index}"
        ],
        device=device,
    ),
    offset=torch.tensor(
        heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][
            f"{config_dictionary.paint_offset}_{index}"
        ],
        device=device,
    ),
    pivot_radius=torch.tensor(
        heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][
            f"{config_dictionary.paint_pivot_radius}_{index}"
        ],
        device=device,
    ),
    initial_angle=torch.tensor(
        heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][
            f"{config_dictionary.paint_initial_angle}_{index}"
        ],
        device=device,
    ),
)
# Include an actuator 1.
actuator1_prototype = ActuatorConfig(
    actuator_key=f"{config_dictionary.actuator_key}_{index}",
    actuator_type=heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][
        f"{config_dictionary.paint_actuator_type}_{index}"
    ].lower(),
    actuator_clockwise=heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][
        f"{config_dictionary.paint_clockwise}_{index}"
    ],
    actuator_parameters=actuator1_parameters,
)
# Include actuator parameters for actuator 2.
index = 2
actuator2_parameters = ActuatorParameters(
    increment=torch.tensor(
        heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][
            f"{config_dictionary.paint_increment}_{index}"
        ],
        device=device,
    ),
    initial_stroke_length=torch.tensor(
        heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][
            f"{config_dictionary.paint_initial_stroke_length}_{index}"
        ],
        device=device,
    ),
    offset=torch.tensor(
        heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][
            f"{config_dictionary.paint_offset}_{index}"
        ],
        device=device,
    ),
    pivot_radius=torch.tensor(
        heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][
            f"{config_dictionary.paint_pivot_radius}_{index}"
        ],
        device=device,
    ),
    initial_angle=torch.tensor(
        heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][
            f"{config_dictionary.paint_initial_angle}_{index}"
        ],
        device=device,
    ),
)
# Include an actuator 2.
actuator2_prototype = ActuatorConfig(
    actuator_key=f"{config_dictionary.actuator_key}_{index}",
    actuator_type=heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][
        f"{config_dictionary.paint_actuator_type}_{index}"
    ].lower(),
    actuator_clockwise=heliostat_dict[config_dictionary.paint_heliostat_kinematic_key][
        f"{config_dictionary.paint_clockwise}_{index}"
    ],
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
