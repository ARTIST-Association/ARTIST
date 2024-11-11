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
        incident_ray_direction = utils.convert_3d_direction_to_4d_format(utils.azimuth_elevation_to_enu(sun_azimuth, sun_elevation, degree=True))

    calibration_properties_file = f"{ARTIST_ROOT}/measurement_data/AA39/Calibration/86500-calibration-properties.json"
    with open(calibration_properties_file, 'r') as file:
        calibration_dict = json.load(file)
        center_calibration_image = utils.calculate_position_in_m_from_lat_lon(torch.tensor(calibration_dict["focal_spot"]["UTIS"]), scenario.power_plant_position)
        center_calibration_image = utils.convert_3d_points_to_4d_format(center_calibration_image)
        motor_positions = torch.tensor([calibration_dict["motor_position"]["Axis1MotorPosition"], calibration_dict["motor_position"]["Axis2MotorPosition"]])
    
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
        factor=0.1,
        patience=30,
        threshold=0.01,
        threshold_mode="abs",
    )# -> 6.854534149169922e-07
    
    loss = torch.inf
    epoch = 0

    normal_vector = (center_calibration_image - scenario.heliostats.heliostat_list[0].position) / torch.norm(center_calibration_image - scenario.heliostats.heliostat_list[0].position)

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


        orientation = scenario.heliostats.heliostat_list[0].set_aligned_surface(
            incident_ray_direction=incident_ray_direction
        )

        new_normal_vector = orientation[0:4, 2]

        optimizer.zero_grad()
        
        loss = (new_normal_vector - normal_vector).abs().mean()
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
        scheduler.step(loss.abs().mean())

        log.info(
            f"Epoch: {epoch}, Loss: {loss.abs().mean().item()}, LR: {optimizer.param_groups[0]['lr']}, normal: {new_normal_vector}",
        )

        epoch += 1

    log.info(
        f"parameters: {parameters_list}",
    )

    # Create raytracer
    raytracer = HeliostatRayTracer(
        scenario=scenario
    )
    
    final_bitmap = raytracer.trace_rays(incident_ray_direction=incident_ray_direction)
    final_bitmap = raytracer.normalize_bitmap(final_bitmap)

    plt.imshow(final_bitmap.T.detach().numpy())
    plt.savefig(f"final{epoch}.png")
