import json
import logging
from time import perf_counter
import h5py
import pytest
import torch
import colorlog
import sys

from artist import ARTIST_ROOT
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.scenario import Scenario
from artist.util import utils

import matplotlib.pyplot as plt

log = logging.getLogger("TEST2")

log_formatter = colorlog.ColoredFormatter(
    fmt="[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s]"
    "[%(log_color)s%(levelname)s%(reset)s] - %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
    secondary_log_colors={},
)
handler = logging.StreamHandler(stream=sys.stdout)
log.addHandler(handler)
log.setLevel(logging.INFO)

def test_motor_position() -> None:

    torch.manual_seed(7)

    # Load the scenario.
    with h5py.File(f"{ARTIST_ROOT}/scenarios/test_alignment_optimization_with_deviations.h5", "r") as config_h5:
        scenario = Scenario.load_scenario_from_hdf5(scenario_file=config_h5)

    # Load the calibration data.
    calibration_item_stac_file = f"{ARTIST_ROOT}/measurement_data/AA39/Calibration/86500-calibration-item-stac.json"
    with open(calibration_item_stac_file, 'r') as file:
        calibration_dict = json.load(file)
        sun_azimuth = torch.tensor(calibration_dict["view:sun_azimuth"])
        sun_elevation = torch.tensor(calibration_dict["view:sun_elevation"])
        incident_ray_direction = utils.convert_3d_direction_to_4d_format(utils.azimuth_elevation_to_enu(sun_azimuth, sun_elevation, degree=False))
        incident_ray_direction.requires_grad_()

    calibration_properties_file = f"{ARTIST_ROOT}/measurement_data/AA39/Calibration/86500-calibration-properties.json"
    with open(calibration_properties_file, 'r') as file:
        calibration_dict = json.load(file)
        center_calibration_image = utils.calculate_position_in_m_from_lat_lon(torch.tensor(calibration_dict["focal_spot"]["UTIS"]), scenario.power_plant_position)
        center_calibration_image = utils.convert_3d_points_to_4d_format(center_calibration_image)
        center_calibration_image.requires_grad_()
        motor_positions = torch.tensor([calibration_dict["motor_position"]["Axis1MotorPosition"], calibration_dict["motor_position"]["Axis2MotorPosition"]])

    
    scenario.heliostats.heliostat_list[0].set_aligned_surface(
        incident_ray_direction=incident_ray_direction
    )

    # Create raytracer
    raytracer = HeliostatRayTracer(
        scenario=scenario
    )

    final_bitmap = raytracer.trace_rays(incident_ray_direction=incident_ray_direction)
    final_bitmap = raytracer.normalize_bitmap(final_bitmap)

    x = "_set_motor_pos"

    plt.imshow(final_bitmap.T.detach().numpy())
    plt.savefig(f"final{x}.png")
