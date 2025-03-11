import pathlib
from pathlib import Path

import h5py
import torch

from artist import ARTIST_ROOT
from artist.scenario import Scenario
from artist.util import config_dictionary, paint_loader, utils
from artist.util import set_logger_config as artist_logger
from artist.util.alignment_optimizer import AlignmentOptimizer
from artist.util.configuration_classes import (
    ActuatorPrototypeConfig,
    ActuatorConfig,
    ActuatorListConfig,
    HeliostatConfig,
    HeliostatListConfig,
    KinematicPrototypeConfig,
    KinematicConfig,
    LightSourceConfig,
    LightSourceListConfig,
    TargetAreaConfig,
    TargetAreaListConfig,
    PowerPlantConfig,
    PrototypeConfig,
    SurfacePrototypeConfig,
)
from artist.util.scenario_generator import ScenarioGenerator
from artist.util.surface_converter import SurfaceConverter

from paint.data.stac_client import StacClient
from paint.util import set_logger_config as paint_logger

def generate_heliostat(name, id, position, aim_point, actuators):
    """
    Generates a heliostat configuration with the given parameters.
    
    Parameters:
    - name (str): Name of the heliostat.
    - id (int): ID of the heliostat.
    - position (torch.Tensor): Position of the heliostat.
    - aim_point (torch.Tensor): Aim point of the heliostat.
    - actuators (list): List of ActuatorConfig objects.

    Returns:
    - HeliostatConfig object.
    """
    actuators_list_config = ActuatorListConfig(actuator_list=actuators)
    kinematic_config = KinematicConfig(
        type="RigidBody",
        initial_orientation=torch.tensor([0.0, -1.0, 0.0]),
        deviations=None,
    )
    
    return HeliostatConfig(
        name=name,
        id=id,
        position=position,
        aim_point=aim_point,
    ), actuators_list_config, kinematic_config

##### Download files from paint #######
paint_logger()
output_dir = Path("tutorials/data/test_scenario_surface_optimization_AA39")
if not output_dir.is_dir():
    client = StacClient(output_dir=output_dir)
    client.get_heliostat_data(
        heliostats=["AA39"],
        collections=["deflectometry", "properties"],
        filtered_calibration_keys=['flux_centered_image']
    )

heliostat_file = output_dir / "AA39" / "Properties" / "AA39-heliostat-properties.json"
deflectometry_file = output_dir / "AA39" / "Deflectometry" / "AA39-filled-2023-09-18Z08-49-09Z-deflectometry.h5"

######### Setup Scenario File ############
artist_logger()

torch.manual_seed(7)
torch.cuda.manual_seed(7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Power plant and target configuration
power_plant_config = PowerPlantConfig(power_plant_position=torch.tensor([0.0, 0.0, 0.0]))
receiver_config = TargetAreaConfig(
    target_area_key="receiver_1",
    geometry="target",
    center=torch.tensor([0.0, -25.0, 0.0]),
    normal_vector=torch.tensor([0.0, 1.0, 0.0]),
    plane_e=10.0,
    plane_u=10.0,
)
receiver_list_config = TargetAreaListConfig(target_area_list=[receiver_config])

light_source_config = LightSourceConfig(
    light_source_key="sun_1",
    light_source_type=config_dictionary.sun_key,
    number_of_rays=100,
    distribution_type=config_dictionary.light_source_distribution_is_normal,
    mean=0.0,
    covariance=4.3681e-06,
)
light_source_list_config = LightSourceListConfig(light_source_list=[light_source_config])

# Generate surface configuration from STRAL data
surface_converter = SurfaceConverter(max_epoch=400)
facet_list = surface_converter.generate_surface_config_from_paint(
    deflectometry_file_path=deflectometry_file,
    heliostat_file_path=heliostat_file,
    device=device,
)
surface_prototype_config = SurfacePrototypeConfig(facet_list=facet_list)

# Kinematic prototype configuration
kinematic_prototype_config = KinematicPrototypeConfig(
    type=config_dictionary.rigid_body_key,
    initial_orientation=torch.tensor([0.0, 0.0, 1.0, 0.0], device=device),
)

# Actuator prototypes
actuator1_prototype = ActuatorConfig(key="actuator_1", type=config_dictionary.ideal_actuator_key, clockwise_axis_movement=False)
actuator2_prototype = ActuatorConfig(key="actuator_2", type=config_dictionary.ideal_actuator_key, clockwise_axis_movement=True)
actuator_prototype_config = ActuatorPrototypeConfig(actuator_list=[actuator1_prototype, actuator2_prototype])

# Prototype configuration
prototype_config = PrototypeConfig(
    surface_prototype=surface_prototype_config,
    kinematic_prototype=kinematic_prototype_config,
    actuators_prototype=actuator_prototype_config,
)

# Create a heliostat
actuator_list = [
    ActuatorConfig(key=f"axis_{i}", type="ideal", clockwise_axis_movement=True, parameters=None) for i in range(2)
]

heliostat, actuators_list_config, kinematic_config = generate_heliostat(
    name="heliostat_1",
    id=1,
    position=torch.tensor([0.0, 0.0, 0.0, 1.0], device=device),
    aim_point=torch.tensor([0.0, -25, 0.0, 1.0], device=device),
    actuators=actuator_list,
)

heliostat_list_config = HeliostatListConfig(heliostat_list=[heliostat])
