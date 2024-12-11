import pathlib

import torch

from artist import ARTIST_ROOT
from artist.util import config_dictionary, paint_loader, set_logger_config
from artist.util.configuration_classes import (
    ActuatorPrototypeConfig,
    HeliostatConfig,
    HeliostatListConfig,
    KinematicPrototypeConfig,
    LightSourceConfig,
    LightSourceListConfig,
    PowerPlantConfig,
    PrototypeConfig,
    ReceiverConfig,
    ReceiverListConfig,
    SurfacePrototypeConfig,
)
from artist.util.scenario_generator import ScenarioGenerator
from artist.util.surface_converter import SurfaceConverter

# Set up logger
set_logger_config()

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


calibration_target_name = paint_loader.extract_paint_calibration_target_name(
    calibration_file
)

power_plant_position, target_type, target_center, normal_vector, plane_e, plane_u = (
    paint_loader.extract_paint_tower_measurements(
        tower_file, calibration_target_name, device
    )
)

heliostat_position, kinematic_config, actuators_list_config = (
    paint_loader.extract_paint_heliostat_properties(
        heliostat_file, power_plant_position, device
    )
)

# Include the power plant configuration.
power_plant_config = PowerPlantConfig(power_plant_position=power_plant_position)

# Include the receiver configuration.
receiver1_config = ReceiverConfig(
    receiver_key="receiver_1",
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


# Generate surface configuration from STRAL data.
surface_converter = SurfaceConverter(
    deflectometry_file_path=deflectometry_file,
    heliostat_file_path=heliostat_file,
    step_size=100,
    max_epoch=400,
)

facet_prototype_list = surface_converter.generate_surface_config(device=device)

surface_prototype_config = SurfacePrototypeConfig(facets_list=facet_prototype_list)

# Include the kinematic prototype configuration.
kinematic_prototype_config = KinematicPrototypeConfig(
    type=config_dictionary.rigid_body_key,
    initial_orientation=kinematic_config.initial_orientation,
    deviations=kinematic_config.deviations,
)

# Include the actuator prototype config.
actuator_prototype_config = ActuatorPrototypeConfig(
    actuator_list=actuators_list_config.actuator_list
)

# Include the final prototype config.
prototype_config = PrototypeConfig(
    surface_prototype=surface_prototype_config,
    kinematic_prototype=kinematic_prototype_config,
    actuator_prototype=actuator_prototype_config,
)

# Include the configuration for a heliostat.
heliostat1 = HeliostatConfig(
    heliostat_key="heliostat_1",
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
    scenario_generator = ScenarioGenerator(
        file_path=file_path,
        power_plant_config=power_plant_config,
        receiver_list_config=receiver_list_config,
        light_source_list_config=light_source_list_config,
        prototype_config=prototype_config,
        heliostat_list_config=heliostats_list_config,
    )
    scenario_generator.generate_scenario()
