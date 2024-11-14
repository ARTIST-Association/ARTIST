import json
import logging
import sys
import time
from typing import Union

from artist import ARTIST_ROOT
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
import colorlog
import h5py
import torch

from artist.scenario import Scenario
from artist.util import utils


class AlignmentOptimizer:
    """
    TODO Docstrings
    """

    def __init__(
        self,
        scenario_path: str,
        calibration_properties_path: str,
        log_level: int = logging.INFO,
    ) -> None:
        """
        TODO Docstrings
        """
        log = logging.getLogger("alignment-optimizer")
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
        handler.setFormatter(log_formatter)
        log.addHandler(handler)
        log.setLevel(log_level)
        self.log = log

        self.scenario_path = scenario_path
        self.calibration_properties_path = calibration_properties_path

    def optimize_kinematic_parameters_with_motor_positions(
        self,
        tolerance: float = 1e-10,
        max_epoch: int = 150,
        initial_learning_rate: float = 0.001,
        scheduler_factor: float = 0.1,
        scheduler_patience: int = 30,
        scheduler_threshold: float = 0.1,
        device: Union[torch.device, str] = "cuda",
    ) -> list[torch.Tensor]:
        """
        TODO Docstrings
        """
        # Load the scenario.
        with h5py.File(self.scenario_path, "r") as config_h5:
            scenario = Scenario.load_scenario_from_hdf5(scenario_file=config_h5, device=device)

        # Load the calibration data (heliostat)
        with open(self.calibration_properties_path, "r") as file:
            calibration_dict = json.load(file)
            center_calibration_image = utils.calculate_position_in_m_from_lat_lon(torch.tensor(calibration_dict["focal_spot"]["UTIS"], device=device), scenario.power_plant_position, device=device)
            center_calibration_image = utils.convert_3d_points_to_4d_format(center_calibration_image, device=device)
            motor_positions = torch.tensor([calibration_dict["motor_position"]["Axis1MotorPosition"], calibration_dict["motor_position"]["Axis2MotorPosition"]], device=device)

        # Set up optimizer
        parameters_list = [scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_e.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_n.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_u.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_e.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_n.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_u.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_e.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_n.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_u.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_e.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_n.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_u.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_e.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_n.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_u.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_e.requires_grad_(),                       
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_n.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_u.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].increment.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].initial_stroke_length.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].offset.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].radius.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].phi_0.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].increment.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].initial_stroke_length.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].offset.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].radius.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].phi_0.requires_grad_()]

        optimizer = torch.optim.Adam(parameters_list, lr=initial_learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            threshold=scheduler_threshold,
            threshold_mode="abs",
        )  # -> 6.854534149169922e-07

        loss = torch.inf
        epoch = 0

        normal_vector = (
            center_calibration_image - scenario.heliostats.heliostat_list[0].position
        )
        normal = normal_vector / torch.norm(normal_vector)

        while loss > tolerance and epoch <= max_epoch:
            orientation = scenario.heliostats.heliostat_list[
                0
            ].get_orientation_from_motor_positions(
                motor_positions=motor_positions, device=device
            )

            new_normal = orientation[0:4, 2]

            optimizer.zero_grad()

            loss = (new_normal - normal).abs().mean()
            loss.backward()

            optimizer.step()
            scheduler.step(loss)

            self.log.info(
                f"Epoch: {epoch}, Loss: {loss.item()}, LR: {optimizer.param_groups[0]['lr']}, normal: {new_normal}",
            )

            epoch += 1

        self.log.info(
            f"parameters: {parameters_list}",
        )

        return parameters_list


    def optimize_kinematic_parameters_with_raytracing(
        self,
        tolerance: float = 1e-10,
        max_epoch: int = 150,
        initial_learning_rate: float = 0.001,
        scheduler_factor: float = 0.1,
        scheduler_patience: int = 3,
        scheduler_threshold: float = 0.1,
        device: Union[torch.device, str] = "cuda",
    ) -> list[torch.Tensor]:
        """
        TODO Docstrings
        """
        # Load the scenario.
        with h5py.File(self.scenario_path, "r") as config_h5:
            scenario = Scenario.load_scenario_from_hdf5(scenario_file=config_h5, device=device)
            target_center = scenario.receivers.receiver_list[0].position_center.requires_grad_()
            plane_e = scenario.receivers.receiver_list[0].plane_e
            plane_u = scenario.receivers.receiver_list[0].plane_u
        
        # Load the calibration data.
        calibration_item_stac_file = f"{ARTIST_ROOT}/measurement_data/download_test/AA39/Calibration/86500-calibration-item-stac.json"
        with open(calibration_item_stac_file, 'r') as file:
            calibration_dict = json.load(file)
            sun_azimuth = torch.tensor(calibration_dict["view:sun_azimuth"], device=device)
            sun_elevation = torch.tensor(calibration_dict["view:sun_elevation"], device=device)
            incident_ray_direction = utils.convert_3d_direction_to_4d_format(utils.azimuth_elevation_to_enu(sun_azimuth, sun_elevation, degree=False), device=device)

        # Load the calibration data (heliostat)
        with open(self.calibration_properties_path, "r") as file:
            calibration_dict = json.load(file)
            center_calibration_image = utils.calculate_position_in_m_from_lat_lon(torch.tensor(calibration_dict["focal_spot"]["UTIS"], device=device), scenario.power_plant_position, device=device)
            center_calibration_image = utils.convert_3d_points_to_4d_format(center_calibration_image, device=device).requires_grad_()

        # Set up optimizer
        parameters_list = [scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_e.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_n.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_u.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_e.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_n.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_u.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_e.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_n.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_u.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_e.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_n.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_u.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_e.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_n.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_u.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_e.requires_grad_(),                       
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_n.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_u.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].increment.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].initial_stroke_length.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].offset.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].radius.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].phi_0.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].increment.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].initial_stroke_length.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].offset.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].radius.requires_grad_(),
                       scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].phi_0.requires_grad_()]

        optimizer = torch.optim.Adam(parameters_list, lr=initial_learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            threshold=scheduler_threshold,
            threshold_mode="abs",
        )

        loss = torch.inf
        epoch = 0

        while loss > tolerance and epoch <= max_epoch:
            # Align heliostat
            start_time = time.time()
            scenario.heliostats.heliostat_list[0].set_aligned_surface(
                    incident_ray_direction=incident_ray_direction, device=device
            )
            end_time = time.time()
            elapsed = end_time - start_time
            self.log.critical(f"align heliostat: {elapsed}")

            # Create raytracer
            start_time = time.time()
            raytracer = HeliostatRayTracer(
                scenario=scenario
            )
            end_time = time.time()
            elapsed = end_time - start_time
            self.log.critical(f"create raytracer: {elapsed}")

            start_time = time.time()
            final_bitmap = raytracer.trace_rays(incident_ray_direction=incident_ray_direction, device=device)
            end_time = time.time()
            elapsed = end_time - start_time
            self.log.critical(f"trace rays: {elapsed}")

            start_time = time.time()
            final_bitmap = raytracer.normalize_bitmap(final_bitmap)
            end_time = time.time()
            elapsed = end_time - start_time
            self.log.critical(f"normalize bitmap: {elapsed}")

            # TODO
            # get_center_of_mass() zu receiver.py verschieben?
            start_time = time.time()
            center = utils.get_center_of_mass(final_bitmap, target_center=target_center, plane_e=plane_e, plane_u=plane_u, device=device)
            end_time = time.time()
            elapsed = end_time - start_time
            self.log.critical(f"center of mass: {elapsed}")

            optimizer.zero_grad()

            loss = (center - center_calibration_image).abs().mean()
            loss.backward()

            optimizer.step()
            scheduler.step(loss)

            self.log.info(
                f"Epoch: {epoch}, Loss: {loss.item()}, LR: {optimizer.param_groups[0]['lr']}, normal: {center}",
            )

            epoch += 1

        self.log.info(
            f"parameters: {parameters_list}",
        )

        return parameters_list