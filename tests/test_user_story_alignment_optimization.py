import json
import logging
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

log = logging.getLogger("TEST")

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
"""A formatter for the logger for the ``PAINT`` to surface converter."""
handler = logging.StreamHandler(stream=sys.stdout)
"""A handler for the logger for the ``PAINT`` to surface converter."""
log.addHandler(handler)
log.setLevel(logging.INFO)

def test_alignment_optimization() -> None:

    # TODO
    tolerance: float = 1e-5
    initial_learning_rate: float = 3
    max_epoch: int = 2500

    torch.manual_seed(7)

    # Load the scenario.
    with h5py.File(f"{ARTIST_ROOT}/scenarios/test_alignment_optimization_with_deviations.h5", "r") as config_h5:
        scenario = Scenario.load_scenario_from_hdf5(scenario_file=config_h5)
        target_center = scenario.receivers.receiver_list[0].position_center.requires_grad_()
        plane_e = scenario.receivers.receiver_list[0].plane_e
        plane_u = scenario.receivers.receiver_list[0].plane_u

    # Load the calibration data.
    calibration_item_stac_file = f"{ARTIST_ROOT}/measurement_data/AA39/Calibration/86500-calibration-item-stac.json"
    with open(calibration_item_stac_file, 'r') as file:
        calibration_dict = json.load(file)
        sun_azimuth = torch.tensor(calibration_dict["view:sun_azimuth"])
        sun_elevation = torch.tensor(calibration_dict["view:sun_elevation"])
        incident_ray_direction = utils.convert_3d_direction_to_4d_format(utils.azimuth_elevation_to_enu(sun_azimuth, sun_elevation, degree=False))

    calibration_properties_file = f"{ARTIST_ROOT}/measurement_data/AA39/Calibration/86500-calibration-properties.json"
    with open(calibration_properties_file, 'r') as file:
        calibration_dict = json.load(file)
        center_calibration_image = utils.calculate_position_in_m_from_lat_lon(torch.tensor(calibration_dict["focal_spot"]["UTIS"], requires_grad=True), scenario.power_plant_position)
        center_calibration_image = utils.convert_3d_points_to_4d_format(center_calibration_image)
        center_calibration_image.requires_grad_()
    
    parameters_list = [scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_e,
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_n,
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_u,
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_e,
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_n,
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_u,
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_e,
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_n,
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_u,
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_e,
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_n,
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_u,
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_e,
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_n,
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_u,
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_e,
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_n,
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_u,]

    #optimizer = torch.optim.Adam([scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_e], lr=initial_learning_rate)
    optimizer = torch.optim.Adam(parameters_list, lr=initial_learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.2,
        patience=50,
        threshold=1e-7,
        threshold_mode="abs",
    )
    loss = torch.inf
    epoch = 0
    while loss > tolerance and epoch <= max_epoch:
        
        # Align heliostat.
        scenario.heliostats.heliostat_list[0].set_aligned_surface(
            incident_ray_direction=incident_ray_direction
        )

        # Create raytracer
        raytracer = HeliostatRayTracer(
            scenario=scenario
        )

        final_bitmap = raytracer.trace_rays(incident_ray_direction=incident_ray_direction)
        final_bitmap = raytracer.normalize_bitmap(final_bitmap)

        # plt.imshow(final_bitmap.detach().numpy())
        # plt.savefig("bitmap.png")

        # TODO
        # get_center_of_mass() zu receiver.py verschieben?
        # center_ideal_scipy = utils.get_center_of_mass_scipy(final_bitmap, target_center=target_center, plane_e=plane_e, plane_u=plane_u).requires_grad_(True)
        center_ideal_torch = utils.get_center_of_mass(final_bitmap, target_center=target_center, plane_e=plane_e, plane_u=plane_u)
        
        optimizer.zero_grad()

        loss = (center_ideal_torch - center_calibration_image).abs().mean()
        loss.backward()
        
        optimizer.step()
        scheduler.step(loss.abs().mean())

        log.info(
            f"Epoch: {epoch}, Loss: {loss.abs().mean().item()}, LR: {optimizer.param_groups[0]['lr']}, Center Ideal: {center_ideal_torch}",
        )

        epoch += 1