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
    #device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    device = "cpu"
    log.info(
            device,
    )
    torch.set_default_device(device)
    # if torch.cuda.is_available(): 
    #     torch.set_default_tensor_type(torch.cuda.FloatTensor)
    


    torch.manual_seed(7)

    # Load the scenario.
    with h5py.File(f"{ARTIST_ROOT}/scenarios/test_alignment_optimization_with_deviations.h5", "r") as config_h5:
        scenario = Scenario.load_scenario_from_hdf5(scenario_file=config_h5)
        target_center = scenario.receivers.receiver_list[0].position_center
        plane_e = scenario.receivers.receiver_list[0].plane_e
        plane_u = scenario.receivers.receiver_list[0].plane_u

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

    # Create raytracer
    raytracer = HeliostatRayTracer(
        scenario=scenario
    )


    first_joint_translation_e_offset = torch.tensor([0.0], requires_grad=True)
    first_joint_translation_n_offset = torch.tensor([0.0], requires_grad=True)
    first_joint_translation_u_offset = torch.tensor([0.0], requires_grad=True)
    first_joint_tilt_e_offset = torch.tensor([0.0], requires_grad=True)
    first_joint_tilt_n_offset = torch.tensor([0.0], requires_grad=True)
    first_joint_tilt_u_offset = torch.tensor([0.0], requires_grad=True)
    second_joint_translation_e_offset = torch.tensor([0.0], requires_grad=True)
    second_joint_translation_n_offset = torch.tensor([0.0], requires_grad=True)
    second_joint_translation_u_offset = torch.tensor([0.0], requires_grad=True)
    second_joint_tilt_e_offset = torch.tensor([0.0], requires_grad=True)
    second_joint_tilt_n_offset = torch.tensor([0.0], requires_grad=True)
    second_joint_tilt_u_offset = torch.tensor([0.0], requires_grad=True)
    concentrator_translation_e_offset = torch.tensor([0.0], requires_grad=True)
    concentrator_translation_n_offset = torch.tensor([0.0], requires_grad=True)
    concentrator_translation_u_offset = torch.tensor([0.0], requires_grad=True)
    concentrator_tilt_e_offset = torch.tensor([0.0], requires_grad=True)
    concentrator_tilt_n_offset = torch.tensor([0.0], requires_grad=True)
    concentrator_tilt_u_offset = torch.tensor([0.0], requires_grad=True)

    increment_1_offset = torch.tensor([0.0], requires_grad=True)
    initial_stroke_length_1_offset = torch.tensor([0.0], requires_grad=True)
    offset_1_offset = torch.tensor([0.0], requires_grad=True)
    radius_1_offset = torch.tensor([0.0], requires_grad=True)
    phi_0_1_offset = torch.tensor([0.0], requires_grad=True)

    increment_2_offset = torch.tensor([0.0], requires_grad=True)
    initial_stroke_length_2_offset = torch.tensor([0.0], requires_grad=True)
    offset_2_offset = torch.tensor([0.0], requires_grad=True)
    radius_2_offset = torch.tensor([0.0], requires_grad=True)
    phi_0_2_offset = torch.tensor([0.0], requires_grad=True)

    parameters_list = [first_joint_translation_e_offset,
                       first_joint_translation_n_offset,
                       first_joint_translation_u_offset,
                       first_joint_tilt_e_offset,
                       first_joint_tilt_n_offset,
                       first_joint_tilt_u_offset,
                       second_joint_translation_e_offset,
                       second_joint_translation_n_offset,
                       second_joint_translation_u_offset,
                       second_joint_tilt_e_offset,
                       second_joint_tilt_n_offset,
                       second_joint_tilt_u_offset,
                       concentrator_translation_e_offset,
                       concentrator_translation_n_offset,
                       concentrator_translation_u_offset,
                       concentrator_tilt_e_offset,                       
                       concentrator_tilt_n_offset,
                       concentrator_tilt_u_offset,
                       increment_1_offset,
                       initial_stroke_length_1_offset,
                       offset_1_offset,
                       radius_1_offset,
                       phi_0_1_offset,
                       increment_2_offset,
                       initial_stroke_length_2_offset,
                       offset_2_offset,
                       radius_2_offset,
                       phi_0_2_offset]
    # TODO
    tolerance: float = 1e-10
    initial_learning_rate: float = 0.001
    max_epoch: int = 150

    optimizer = torch.optim.Adam(parameters_list, lr=initial_learning_rate)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.2,
        patience=5,
        threshold=500,
        threshold_mode="abs",
    )
    loss = torch.inf
    epoch = 0

    while loss > tolerance and epoch <= max_epoch:
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_e = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_e + first_joint_translation_e_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_n = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_n + first_joint_translation_n_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_u = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_u + first_joint_translation_u_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_e = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_e + first_joint_tilt_e_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_n = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_n + first_joint_tilt_n_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_u = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_u + first_joint_tilt_u_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_e = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_e + second_joint_translation_e_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_n = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_n + second_joint_translation_n_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_u = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_u + second_joint_translation_u_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_e = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_e + second_joint_tilt_e_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_n = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_n + second_joint_tilt_n_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_u = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_u + second_joint_tilt_u_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_e = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_e + concentrator_translation_e_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_n = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_n + concentrator_translation_n_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_u = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_u + concentrator_translation_u_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_e = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_e + concentrator_tilt_e_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_n = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_n + concentrator_tilt_n_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_u = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_u + concentrator_tilt_u_offset

        scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].increment = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].increment + increment_1_offset
        scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].initial_stroke_length = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].initial_stroke_length + initial_stroke_length_1_offset
        scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].offset = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].offset + offset_1_offset
        scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].radius = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].radius + radius_1_offset
        scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].phi_0 = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].phi_0 + phi_0_1_offset
        
        scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].increment = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].increment + increment_2_offset
        scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].initial_stroke_length = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].initial_stroke_length + initial_stroke_length_2_offset
        scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].offset = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].offset + offset_2_offset
        scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].radius = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].radius + radius_2_offset
        scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].phi_0 = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].phi_0 + phi_0_2_offset

        # Align heliostat.
        actuator_steps = scenario.heliostats.heliostat_list[0].set_aligned_surface(
            incident_ray_direction=incident_ray_direction
        )

        optimizer.zero_grad()

        loss = (actuator_steps - motor_positions).abs().sum()
        loss.backward()

        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_e = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_e - first_joint_translation_e_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_n = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_n - first_joint_translation_n_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_u = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_u - first_joint_translation_u_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_e = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_e - first_joint_tilt_e_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_n = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_n - first_joint_tilt_n_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_u = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_u - first_joint_tilt_u_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_e = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_e - second_joint_translation_e_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_n = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_n - second_joint_translation_n_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_u = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_u - second_joint_translation_u_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_e = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_e - second_joint_tilt_e_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_n = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_n - second_joint_tilt_n_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_u = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_u - second_joint_tilt_u_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_e = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_e - concentrator_translation_e_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_n = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_n - concentrator_translation_n_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_u = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_u - concentrator_translation_u_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_e = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_e - concentrator_tilt_e_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_n = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_n - concentrator_tilt_n_offset
        scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_u = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_u - concentrator_tilt_u_offset

        scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].increment = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].increment - increment_1_offset
        scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].initial_stroke_length = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].initial_stroke_length - initial_stroke_length_1_offset
        scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].offset = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].offset - offset_1_offset
        scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].radius = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].radius - radius_1_offset
        scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].phi_0 = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].phi_0 - phi_0_1_offset
        
        scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].increment = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].increment - increment_2_offset
        scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].initial_stroke_length = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].initial_stroke_length - initial_stroke_length_2_offset
        scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].offset = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].offset - offset_2_offset
        scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].radius = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].radius - radius_2_offset
        scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].phi_0 = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].phi_0 - phi_0_2_offset

        optimizer.step()
        scheduler.step(loss.abs().sum())

        log.info(
            f"Epoch: {epoch}, Loss: {loss.abs().sum().item()}, LR: {optimizer.param_groups[0]['lr']}, Actuator steps: {actuator_steps}",
        )


        if epoch in range(0, 10):
        #if (epoch + 1) % 20 == 0:
            final_bitmap = raytracer.trace_rays(incident_ray_direction=incident_ray_direction)
            final_bitmap = raytracer.normalize_bitmap(final_bitmap)

            plt.imshow(final_bitmap.T.detach().numpy())
            plt.savefig(f"final{epoch}.png")

        epoch += 1




















    # while loss > tolerance and epoch <= max_epoch:
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_e = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_e + first_joint_translation_e_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_n = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_n + first_joint_translation_n_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_u = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_u + first_joint_translation_u_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_e = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_e + first_joint_tilt_e_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_n = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_n + first_joint_tilt_n_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_u = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_u + first_joint_tilt_u_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_e = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_e + second_joint_translation_e_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_n = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_n + second_joint_translation_n_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_u = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_u + second_joint_translation_u_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_e = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_e + second_joint_tilt_e_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_n = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_n + second_joint_tilt_n_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_u = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_u + second_joint_tilt_u_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_e = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_e + concentrator_translation_e_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_n = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_n + concentrator_translation_n_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_u = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_u + concentrator_translation_u_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_e = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_e + concentrator_tilt_e_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_n = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_n + concentrator_tilt_n_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_u = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_u + concentrator_tilt_u_offset

    #     scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].increment = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].increment + increment_1_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].initial_stroke_length = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].initial_stroke_length + initial_stroke_length_1_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].offset = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].offset + offset_1_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].radius = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].radius + radius_1_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].phi_0 = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].phi_0 + phi_0_1_offset
        
    #     scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].increment = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].increment + increment_2_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].initial_stroke_length = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].initial_stroke_length + initial_stroke_length_2_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].offset = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].offset + offset_2_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].radius = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].radius + radius_2_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].phi_0 = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].phi_0 + phi_0_2_offset

    #     # Align heliostat.
    #     actuator_steps = scenario.heliostats.heliostat_list[0].set_aligned_surface(
    #         incident_ray_direction=incident_ray_direction
    #     )

    #     # Create raytracer
    #     raytracer = HeliostatRayTracer(
    #         scenario=scenario
    #     )

    #     final_bitmap = raytracer.trace_rays(incident_ray_direction=incident_ray_direction)
    #     final_bitmap = raytracer.normalize_bitmap(final_bitmap)

    #     # TODO
    #     # get_center_of_mass() zu receiver.py verschieben?
    #     center_ideal_torch = utils.get_center_of_mass(final_bitmap, target_center=target_center, plane_e=plane_e, plane_u=plane_u)
        
    #     optimizer.zero_grad()

    #     loss = (center_ideal_torch - center_calibration_image).abs().sum()
    #     loss.backward()

    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_e = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_e - first_joint_translation_e_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_n = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_n - first_joint_translation_n_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_u = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_u - first_joint_translation_u_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_e = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_e - first_joint_tilt_e_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_n = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_n - first_joint_tilt_n_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_u = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_u - first_joint_tilt_u_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_e = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_e - second_joint_translation_e_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_n = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_n - second_joint_translation_n_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_u = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_u - second_joint_translation_u_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_e = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_e - second_joint_tilt_e_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_n = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_n - second_joint_tilt_n_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_u = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_u - second_joint_tilt_u_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_e = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_e - concentrator_translation_e_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_n = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_n - concentrator_translation_n_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_u = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_u - concentrator_translation_u_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_e = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_e - concentrator_tilt_e_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_n = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_n - concentrator_tilt_n_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_u = scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_u - concentrator_tilt_u_offset

    #     scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].increment = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].increment - increment_1_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].initial_stroke_length = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].initial_stroke_length - initial_stroke_length_1_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].offset = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].offset - offset_1_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].radius = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].radius - radius_1_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].phi_0 = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].phi_0 - phi_0_1_offset
        
    #     scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].increment = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].increment - increment_2_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].initial_stroke_length = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].initial_stroke_length - initial_stroke_length_2_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].offset = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].offset - offset_2_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].radius = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].radius - radius_2_offset
    #     scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].phi_0 = scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].phi_0 - phi_0_2_offset

    #     optimizer.step()
    #     scheduler.step(loss.abs().sum())

    #     log.info(
    #         f"Epoch: {epoch}, Loss: {loss.abs().sum().item()}, LR: {optimizer.param_groups[0]['lr']}, Center Ideal: {center_ideal_torch}",
    #     )


    #     # if epoch in range(0, 10):
    #     # #if (epoch + 1) % 20 == 0:
    #     #     final_bitmap = raytracer.trace_rays(incident_ray_direction=incident_ray_direction)
    #     #     final_bitmap = raytracer.normalize_bitmap(final_bitmap)

    #     #     plt.imshow(final_bitmap.T.detach().numpy())
    #     #     plt.savefig(f"final{epoch}.png")

    #     # epoch += 1
