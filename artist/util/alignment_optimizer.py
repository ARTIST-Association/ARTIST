import json
import logging
import pathlib
import sys
from typing import Union

import colorlog
import h5py
import torch

from artist.raytracing import raytracing_utils
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.scenario import Scenario
from artist.util import utils


class AlignmentOptimizer:
    """
    Implements an alignment optimizer to find optimal kinematic parameters.

    Attributes
    ----------
    log : logging.Logger
        Logger used to display optimization progress.
    scenario_path : pathlib.Path
        The path to the scenario h5 file.
    calibration_properties_path : pathlib.Path
        The path to the calibration properties json file.

    Methods
    -------
    optimize_kinematic_parameters_with_motor_positions()
        Optimize the kinematic parameters using the motor positions.
    optimize_kinematic_parameters_with_raytracing()
        Optimize the kinematic parameters using raytracing (slower).
    """

    def __init__(
        self,
        scenario_path: pathlib.Path,
        calibration_properties_path: pathlib.Path,
        log_level: int = logging.INFO,
    ) -> None:
        """
        Initialize the alignment optimizer.

        The alignment optimizer optimizes all 28 Parameters of the rigid body kinematics.
        These parameters include the 18 kinematic deviations parameters as well as 5 actuator
        parameters for each actuator.

        Parameters
        ----------
        log : logging.Logger
            Logger used to display optimization progress.
        scenario_path : pathlib.Path
            The path to the scenario h5 file.
        calibration_properties_path : pathlib.Path
            The path to the calibration properties json file.
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
        tolerance: float = 1e-7,
        max_epoch: int = 150,
        initial_learning_rate: float = 0.001,
        scheduler_factor: float = 0.1,
        scheduler_patience: int = 20,
        scheduler_threshold: float = 0.1,
        device: Union[torch.device, str] = "cuda",
    ) -> tuple[list[torch.Tensor], Scenario]:
        """
        Optimize the kinematic parameters using the motor positions.

        This optimizer method optimizes the kinematic parameters by extracting the motor positions
        and incident ray direction from a specific calibration and using the scence's geometry.

        Parameters
        ----------
        tolerance : float
            The tolerance indicating when to stop optimizing (default: 1e-7).
        max_epoch : int
            Maximum number of epochs for the optimizer (default: 150).
        initial_learning_rate : float
            The initial learning rate of the optimizer (default: 0.001).
        scheduler_factor : float
            Factor by which the learning rate will be reduced (default: 0.1).
        scheduler_patience : int
            The number of allowed epochs with no improvement after which the learning rate will be reduced (default: 20).
        scheduler_threshold : float
            The scheduler threshold (default: 0.1).
        device : Union[torch.device, str]
            The device on which to initialize tensors (default: cuda).

        Returns
        -------
        list[torch.Tensor]
            The list of optimized parameters.
        Scenario
            The scenario with aligned heliostat and optimized kinematic parameters.
        """
        # Load the scenario.
        with h5py.File(self.scenario_path, "r") as config_h5:
            scenario = Scenario.load_scenario_from_hdf5(
                scenario_file=config_h5, device=device
            )

        # Load the calibration data
        with open(self.calibration_properties_path, "r") as file:
            calibration_dict = json.load(file)
            center_calibration_image = utils.convert_wgs84_coordinates_to_local_enu(
                torch.tensor(
                    calibration_dict["focal_spot"]["UTIS"],
                    dtype=torch.float64,
                    device=device,
                ),
                scenario.power_plant_position,
                device=device,
            )
            center_calibration_image = utils.convert_3d_points_to_4d_format(
                center_calibration_image, device=device
            )
            sun_azimuth = torch.tensor(calibration_dict["Sun_azimuth"], device=device)
            sun_elevation = torch.tensor(
                calibration_dict["Sun_elevation"], device=device
            )
            incident_ray_direction = utils.convert_3d_direction_to_4d_format(
                utils.azimuth_elevation_to_enu(sun_azimuth, sun_elevation, degree=True),
                device=device,
            )
            motor_positions = torch.tensor(
                [
                    calibration_dict["motor_position"]["Axis1MotorPosition"],
                    calibration_dict["motor_position"]["Axis2MotorPosition"],
                ],
                device=device,
            )

        # Set up optimizer
        parameters_list = [
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.first_joint_translation_e,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.first_joint_translation_n,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.first_joint_translation_u,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.first_joint_tilt_e,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.first_joint_tilt_n,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.first_joint_tilt_u,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.second_joint_translation_e,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.second_joint_translation_n,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.second_joint_translation_u,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.second_joint_tilt_e,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.second_joint_tilt_n,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.second_joint_tilt_u,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.concentrator_translation_e,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.concentrator_translation_n,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.concentrator_translation_u,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.concentrator_tilt_e,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.concentrator_tilt_n,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.concentrator_tilt_u,
            scenario.heliostats.heliostat_list[0]
            .kinematic.actuators.actuator_list[0]
            .increment,
            scenario.heliostats.heliostat_list[0]
            .kinematic.actuators.actuator_list[0]
            .initial_stroke_length,
            scenario.heliostats.heliostat_list[0]
            .kinematic.actuators.actuator_list[0]
            .offset,
            scenario.heliostats.heliostat_list[0]
            .kinematic.actuators.actuator_list[0]
            .radius,
            scenario.heliostats.heliostat_list[0]
            .kinematic.actuators.actuator_list[0]
            .phi_0,
            scenario.heliostats.heliostat_list[0]
            .kinematic.actuators.actuator_list[1]
            .increment,
            scenario.heliostats.heliostat_list[0]
            .kinematic.actuators.actuator_list[1]
            .initial_stroke_length,
            scenario.heliostats.heliostat_list[0]
            .kinematic.actuators.actuator_list[1]
            .offset,
            scenario.heliostats.heliostat_list[0]
            .kinematic.actuators.actuator_list[1]
            .radius,
            scenario.heliostats.heliostat_list[0]
            .kinematic.actuators.actuator_list[1]
            .phi_0,
        ]

        for parameter in parameters_list:
            if parameter is not None:
                parameter.requires_grad_()

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

        preferred_reflection_direction_calibration = (
            center_calibration_image - scenario.heliostats.heliostat_list[0].position
        )
        preferred_reflection_direction_calibration = (
            preferred_reflection_direction_calibration
            / torch.norm(preferred_reflection_direction_calibration)
        )

        while loss > tolerance and epoch <= max_epoch:
            orientation = scenario.heliostats.heliostat_list[
                0
            ].get_orientation_from_motor_positions(
                motor_positions=motor_positions, device=device
            )

            preferred_reflection_direction = raytracing_utils.reflect(
                -incident_ray_direction, orientation[0:4, 2]
            )

            optimizer.zero_grad()

            loss = (
                (
                    preferred_reflection_direction
                    - preferred_reflection_direction_calibration
                )
                .abs()
                .mean()
            )
            loss.backward()

            optimizer.step()
            scheduler.step(loss)

            self.log.info(
                f"Epoch: {epoch}, Loss: {loss.item()}, LR: {optimizer.param_groups[0]['lr']}, normal: {preferred_reflection_direction}",
            )

            epoch += 1

        self.log.info(
            f"parameters: {parameters_list}",
        )

        # Align heliostat, reason: scenario will be ready to use for raytracing.
        # can be removed if we decide to only return the optimized paramters.
        scenario.heliostats.heliostat_list[0].set_aligned_surface_with_motor_positions(
            motor_positions=motor_positions.to(device), device=device
        )

        return parameters_list, scenario

    def optimize_kinematic_parameters_with_raytracing(
        self,
        tolerance: float = 0.05,
        max_epoch: int = 20,
        initial_learning_rate: float = 0.001,
        scheduler_factor: float = 0.1,
        scheduler_patience: int = 7,
        scheduler_threshold: float = 0.5,
        device: Union[torch.device, str] = "cuda",
    ) -> tuple[list[torch.Tensor], Scenario]:
        """
        Optimize the kinematic parameters using raytracing.

        This optimizer method optimizes the kinematic parameters by extracting the focus point
        of a calibration image and using heliostat-tracing. This method is slower than the other
        optimization method found in the alignment optimizer.

        Parameters
        ----------
        tolerance : float
            The tolerance indicating when to stop optimizing (default: 0.05).
        max_epoch : int
            Maximum number of epochs for the optimizer (default: 20).
        initial_learning_rate : float
            The initial learning rate of the optimizer (default: 0.001).
        scheduler_factor : float
            Factor by which the learning rate will be reduced (default: 0.1).
        scheduler_patience : int
            The number of allowed epochs with no improvement after which the learning rate will be reduced (default: 7).
        scheduler_threshold : float
            The scheduler threshold (default: 0.5).
        device : Union[torch.device, str]
            The device on which to initialize tensors (default: cuda).

        Returns
        -------
        list[torch.Tensor]
            The list of optimized parameters.
        Scenario
            The scenario with aligned heliostat and optimized kinematic parameters.
        """
        # Load the scenario.
        with h5py.File(self.scenario_path, "r") as config_h5:
            scenario = Scenario.load_scenario_from_hdf5(
                scenario_file=config_h5, device=device
            )
            target_center = scenario.receivers.receiver_list[
                0
            ].position_center.requires_grad_()
            plane_e = scenario.receivers.receiver_list[0].plane_e
            plane_u = scenario.receivers.receiver_list[0].plane_u

        # Load the calibration data.
        with open(self.calibration_properties_path, "r") as file:
            calibration_dict = json.load(file)
            center_calibration_image = utils.convert_wgs84_coordinates_to_local_enu(
                torch.tensor(
                    calibration_dict["focal_spot"]["UTIS"],
                    dtype=torch.float64,
                    device=device,
                ),
                scenario.power_plant_position,
                device=device,
            )
            center_calibration_image = utils.convert_3d_points_to_4d_format(
                center_calibration_image, device=device
            ).requires_grad_()
            sun_azimuth = torch.tensor(calibration_dict["Sun_azimuth"], device=device)
            sun_elevation = torch.tensor(
                calibration_dict["Sun_elevation"], device=device
            )
            incident_ray_direction = utils.convert_3d_direction_to_4d_format(
                utils.azimuth_elevation_to_enu(sun_azimuth, sun_elevation, degree=True),
                device=device,
            )

        # Set up optimizer
        parameters_list = [
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.first_joint_translation_e,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.first_joint_translation_n,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.first_joint_translation_u,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.first_joint_tilt_e,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.first_joint_tilt_n,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.first_joint_tilt_u,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.second_joint_translation_e,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.second_joint_translation_n,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.second_joint_translation_u,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.second_joint_tilt_e,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.second_joint_tilt_n,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.second_joint_tilt_u,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.concentrator_translation_e,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.concentrator_translation_n,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.concentrator_translation_u,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.concentrator_tilt_e,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.concentrator_tilt_n,
            scenario.heliostats.heliostat_list[
                0
            ].kinematic.deviation_parameters.concentrator_tilt_u,
            scenario.heliostats.heliostat_list[0]
            .kinematic.actuators.actuator_list[0]
            .increment,
            scenario.heliostats.heliostat_list[0]
            .kinematic.actuators.actuator_list[0]
            .initial_stroke_length,
            scenario.heliostats.heliostat_list[0]
            .kinematic.actuators.actuator_list[0]
            .offset,
            scenario.heliostats.heliostat_list[0]
            .kinematic.actuators.actuator_list[0]
            .radius,
            scenario.heliostats.heliostat_list[0]
            .kinematic.actuators.actuator_list[0]
            .phi_0,
            scenario.heliostats.heliostat_list[0]
            .kinematic.actuators.actuator_list[1]
            .increment,
            scenario.heliostats.heliostat_list[0]
            .kinematic.actuators.actuator_list[1]
            .initial_stroke_length,
            scenario.heliostats.heliostat_list[0]
            .kinematic.actuators.actuator_list[1]
            .offset,
            scenario.heliostats.heliostat_list[0]
            .kinematic.actuators.actuator_list[1]
            .radius,
            scenario.heliostats.heliostat_list[0]
            .kinematic.actuators.actuator_list[1]
            .phi_0,
        ]

        for parameter in parameters_list:
            if parameter is not None:
                parameter.requires_grad_()

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
            scenario.heliostats.heliostat_list[
                0
            ].set_aligned_surface_with_incident_ray_direction(
                incident_ray_direction=incident_ray_direction, device=device
            )

            # Create raytracer
            raytracer = HeliostatRayTracer(scenario=scenario)

            final_bitmap = raytracer.trace_rays(
                incident_ray_direction=incident_ray_direction, device=device
            )
            final_bitmap = raytracer.normalize_bitmap(final_bitmap)

            center = utils.get_center_of_mass(
                torch.flip(final_bitmap.T, dims=(0, 1)),
                target_center=target_center,
                plane_e=plane_e,
                plane_u=plane_u,
                device=device,
            )

            optimizer.zero_grad()

            loss = (center - center_calibration_image).abs().mean()
            loss.backward()

            optimizer.step()
            scheduler.step(loss)

            self.log.info(
                f"Epoch: {epoch}, Loss: {loss.item()}, LR: {optimizer.param_groups[0]['lr']}, center of mass: {center}",
            )

            epoch += 1

        self.log.info(
            f"parameters: {parameters_list}",
        )

        return parameters_list, scenario
