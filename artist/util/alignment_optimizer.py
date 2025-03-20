import logging
from typing import Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

from artist.raytracing import raytracing_utils
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util import utils
from artist.util.scenario import Scenario

log = logging.getLogger(__name__)
"""A logger for the alignment optimizer."""


class KinematicOptimizer:
    """
    An optimizer used to find optimal kinematic parameters.

    The kinematic optimizer optimizes kinematic parameters.
    These parameters are specific to a certain kinematic type 
    and can for example include the 18 kinematic deviations parameters as well as five actuator
    parameters for each actuator for a rigid body kinematic.

    Attributes
    ----------
    scenario : Scenario
        The scenario.
    optimizer : Optimizer
        The optimizer.
    scheduler : Union[_LRScheduler, ReduceLROnPlateau]
        The learning rate scheduler.

    Methods
    -------
    optimize()
        Optimize the kinematic parameters.
    """

    def __init__(
        self,
        scenario: Scenario,
        optimizer: Optimizer,
        scheduler: Union[_LRScheduler, ReduceLROnPlateau],
    ) -> None:
        """
        Initialize the kinematic optimizer.

        Parameters
        ----------
        scenario : Scenario
            The scenario.
        optimizer : Optimizer
            The optimizer.
        scheduler : Union[_LRScheduler, ReduceLROnPlateau]
            The learning rate scheduler.
        """
        log.info("Create a kinematic optimizer.")
        self.scenario = scenario
        self.optimizer = optimizer
        self.scheduler = scheduler

    def optimize(
        self,
        tolerance: float,
        max_epoch: int,
        center_calibration_images: torch.Tensor,
        incident_ray_directions: torch.Tensor,
        calibration_target_names: Optional[str] = None,
        motor_positions: Optional[torch.Tensor] = None,
        num_log: int = 3,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """
        Optimize the kinematic parameters.

        Parameters
        ----------
        tolerance : float
            The optimizer tolerance.
        max_epoch : int
            The maximum number of optimization epochs.
        center_calibration_images : torch.Tensor
            The centers of the calibration flux densities.
        incident_ray_directions : torch.Tensor
            The incident ray directions specified in the calibration.
        calibration_target_names : Optional[str]
            The name of the calibration targets (default is None).
        motor_positions : Optional[torch.Tensor]
            The motor positions specified in the calibration files (default is None).
        num_log : int
            Number of log messages during training (default is 3).
        device : Union[torch.device, str] = "cuda"
            The device on which to initialize tensors (default is cuda).
        """
        log.info("Start the kinematic calibration.")
        device = torch.device(device)

        if motor_positions is not None:
            self._optimize_kinematic_parameters_with_motor_positions(
                tolerance=tolerance,
                max_epoch=max_epoch,
                center_calibration_images=center_calibration_images,
                incident_ray_directions=incident_ray_directions,
                all_motor_positions=motor_positions,
                num_log=num_log,
                device=device,
            )

        elif calibration_target_names is not None:
            self._optimize_kinematic_parameters_with_raytracing(
                tolerance=tolerance,
                max_epoch=max_epoch,
                calibration_target_names=calibration_target_names,
                center_calibration_images=center_calibration_images,
                incident_ray_directions=incident_ray_directions,
                num_log=num_log,
                device=device,
            )
        log.info("Kinematic parameters optimized.")

    def _optimize_kinematic_parameters_with_motor_positions(
        self,
        tolerance: float,
        max_epoch: int,
        center_calibration_images: torch.Tensor,
        incident_ray_directions: torch.Tensor,
        all_motor_positions: torch.Tensor,
        num_log: int = 3,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """
        Optimize the kinematic parameters using the motor positions.

        This optimizer method optimizes the kinematic parameters by extracting the motor positions
        and incident ray direction from a specific calibration and using the scene's geometry.

        Parameters
        ----------
        tolerance : float
            The optimzer tolerance.
        max_epoch : int
            The maximum number of optimization epochs.
        center_calibration_images : torch.Tensor
            The center of the calibration flux densities.
        incident_ray_directions : torch.Tensor
            The incident ray directions specified in the calibration data.
        all_motor_positions : torch.Tensor
            The motor positions specified in the calibration data.
        num_log : int
            Number of log messages during training (default is 3).
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).
        """
        log.info("Kinematic calibration with motor positions.")
        device = torch.device(device)
        loss = torch.inf
        epoch = 0

        preferred_reflection_direction_calibrations = torch.nn.functional.normalize(
            (
            center_calibration_images
            - self.scenario.heliostat_field.all_heliostat_positions
            ), p=2, dim=1
        )

        log_step = max_epoch // num_log
        while loss > tolerance and epoch <= max_epoch:
            total_loss = 0.0
            self.optimizer.zero_grad()

            for motor_positions, incident_ray_direction, preferred_reflection_direction_calibration in zip(all_motor_positions, incident_ray_directions, preferred_reflection_direction_calibrations):

                orientation = self.scenario.heliostat_field.get_orientations_from_motor_positions(
                    motor_positions=motor_positions,
                    device=device
                )

                preferred_reflection_direction = raytracing_utils.reflect(
                    incoming_ray_direction=incident_ray_direction, 
                    reflection_surface_normals=orientation[:, 0:4, 2]
                )

                loss = (
                (
                    preferred_reflection_direction
                    - preferred_reflection_direction_calibration
                )
                .abs()
                .mean()
                )

                loss.backward()
                total_loss += loss

            self.optimizer.step()
            self.scheduler.step(total_loss)

            if epoch % log_step == 0:
                log.info(
                    f"Epoch: {epoch}, Loss: {total_loss.item()}, LR: {self.optimizer.param_groups[0]['lr']}",
                )

            epoch += 1
    

    def _optimize_kinematic_parameters_with_raytracing(
        self,
        tolerance: float,
        max_epoch: int,
        calibration_target_names: list[str],
        center_calibration_images: torch.Tensor,
        incident_ray_directions: torch.Tensor,
        num_log: int = 3,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """
        Optimize the kinematic parameters using raytracing.

        This optimizer method optimizes the kinematic parameters by extracting the focus point
        of a calibration image and using heliostat-tracing. This method is slower than the other
        optimization method found in the kinematic optimizer.

        Parameters
        ----------
        tolerance : float
            The optimzer tolerance.
        max_epoch : int
            The maximum number of optimization epochs.
        calibration_target_names : list[str]
            The name of the calibration targets.
        center_calibration_images : torch.Tensor
            The centers of the calibration flux densities.
        incident_ray_directions : torch.Tensor
            The incident ray directions specified in the calibration data.
        num_log : int
            Number of log messages during training (default is 3).
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).
        """
        log.info("Kinematic optimization with raytracing.")
        device = torch.device(device)
        loss = torch.inf
        epoch = 0

        # Create a raytracer.
        raytracer = HeliostatRayTracer(
            scenario=self.scenario,
            batch_size=1
        )

        log_step = max_epoch // num_log
        while loss > tolerance and epoch <= max_epoch:
            total_loss = 0.0
            self.optimizer.zero_grad()

            for calibration_target_name, incident_ray_direction, center_calibration_image in zip(calibration_target_names, incident_ray_directions, center_calibration_images):

                calibration_target = self.scenario.get_target_area(calibration_target_name)
                
                self.scenario.heliostat_field.all_aim_points = calibration_target.center.expand(self.scenario.heliostat_field.number_of_heliostats, -1)

                # Align heliostat.
                self.scenario.heliostat_field.align_surfaces_with_incident_ray_direction(
                    incident_ray_direction=incident_ray_direction,
                    device=device)

                # Perform heliostat-based raytracing.
                final_bitmap = raytracer.trace_rays(
                    incident_ray_direction=incident_ray_direction,
                    target_area=calibration_target,
                    device=device
                )

                center = utils.get_center_of_mass(
                    bitmap=final_bitmap,
                    target_center=calibration_target.center,
                    plane_e=calibration_target.plane_e,
                    plane_u=calibration_target.plane_u,
                    device=device,
                )

                loss = (center - center_calibration_image).abs().mean()
                loss.backward()
                total_loss += loss

            self.optimizer.step()
            self.scheduler.step(total_loss)

            if epoch % log_step == 0:
                log.info(
                    f"Epoch: {epoch}, Loss: {total_loss.item()}, LR: {self.optimizer.param_groups[0]['lr']}",
                )

            epoch += 1
