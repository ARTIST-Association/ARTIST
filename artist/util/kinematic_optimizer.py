import logging
from typing import Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

from artist.field.heliostat_group import HeliostatGroup
from artist.raytracing import raytracing_utils
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util import utils
from artist.util.scenario import Scenario

log = logging.getLogger(__name__)
"""A logger for the kinematic optimizer."""


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
    calibration_group : HeliostatGroup
        The heliostat group to be calibrated.
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
        calibration_group: HeliostatGroup,
        optimizer: Optimizer,
        scheduler: Union[_LRScheduler, ReduceLROnPlateau],
    ) -> None:
        """
        Initialize the kinematic optimizer.

        Parameters
        ----------
        scenario : Scenario
            The scenario.
        calibration_group : HeliostatGroup
            The heliostat group to be calibrated.
        optimizer : Optimizer
            The optimizer.
        scheduler : Union[_LRScheduler, ReduceLROnPlateau]
            The learning rate scheduler.
        """
        log.info("Create a kinematic optimizer.")
        self.scenario = scenario
        self.calibration_group=calibration_group
        self.optimizer = optimizer
        self.scheduler = scheduler

    def optimize(
        self,
        tolerance: float,
        max_epoch: int,
        centers_calibration_images: torch.Tensor,
        incident_ray_directions: torch.Tensor,
        target_area_indices: Optional[torch.Tensor] = None,
        motor_positions: Optional[torch.Tensor] = None,
        num_log: int = 3,
        device: Union[torch.device, str] = "cuda",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Optimize the kinematic parameters.

        Parameters
        ----------
        tolerance : float
            The optimizer tolerance.
        max_epoch : int
            The maximum number of optimization epochs.
        centers_calibration_images : torch.Tensor
            The centers of the calibration flux densities.
        incident_ray_directions : torch.Tensor
            The incident ray directions specified in the calibrations.
        target_area_indices : Optional[torch.Tensor]
            The indices of the target area for each calibration (default is None).
        motor_positions : Optional[torch.Tensor]
            The motor positions specified in the calibration files (default is None).
        num_log : int
            Number of log messages during training (default is 3).
        device : Union[torch.device, str] = "cuda"
            The device on which to initialize tensors (default is cuda).
        
        Returns
        -------
        torch.Tensor
            The calibrated kinematic deviation parameters.
        torch.Tensor
            The calibrated actuator parameters.
        """
        log.info("Start the kinematic calibration.")
        device = torch.device(device)

        if motor_positions is not None:
            self._optimize_kinematic_parameters_with_motor_positions(
                tolerance=tolerance,
                max_epoch=max_epoch,
                centers_calibration_images=centers_calibration_images,
                incident_ray_directions=incident_ray_directions,
                motor_positions=motor_positions,
                num_log=num_log,
                device=device,
            )

        elif motor_positions is None:
            self._optimize_kinematic_parameters_with_raytracing(
                tolerance=tolerance,
                max_epoch=max_epoch,
                target_area_indices=target_area_indices,
                centers_calibration_images=centers_calibration_images,
                incident_ray_directions=incident_ray_directions,
                num_log=num_log,
                device=device,
            )
        log.info("Kinematic parameters optimized.")
    
        return self.calibration_group.kinematic_deviation_parameters, self.calibration_group.actuator_parameters

    def _optimize_kinematic_parameters_with_motor_positions(
        self,
        tolerance: float,
        max_epoch: int,
        centers_calibration_images: torch.Tensor,
        incident_ray_directions: torch.Tensor,
        motor_positions: torch.Tensor,
        num_log: int = 3,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """
        Optimize the kinematic parameters using the motor positions.

        This optimizer method optimizes the kinematic parameters by extracting the motor positions
        and incident ray directions from calibration data and using the scene's geometry.

        Parameters
        ----------
        tolerance : float
            The optimizer tolerance.
        max_epoch : int
            The maximum number of optimization epochs.
        centers_calibration_images : torch.Tensor
            The centers of the calibration flux densities.
        incident_ray_directions : torch.Tensor
            The incident ray directions specified in the calibration data.
        motor_positions : torch.Tensor
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

        active_heliostats_indices = torch.arange(motor_positions.shape[0], device=device)
        unique_heliostats = list(set(self.calibration_group.names))
        heliostat_duplicates_mapping = []
        for heliostat in unique_heliostats:
            heliostat_duplicates_mapping.append([i for i, s in enumerate(self.calibration_group.names) if s == heliostat])

        preferred_reflection_directions_calibration = torch.nn.functional.normalize(
            (
                centers_calibration_images
                - self.calibration_group.positions
            ),
            p=2,
            dim=1,
        )

        log_step = max_epoch // num_log
        while loss > tolerance and epoch <= max_epoch:
            self.optimizer.zero_grad()


            orientations = (
                self.calibration_group.get_orientations_from_motor_positions(
                    motor_positions=motor_positions, 
                    active_heliostats_indices=active_heliostats_indices,
                    device=device
                )
            )

            preferred_reflection_directions = raytracing_utils.reflect(
                incident_ray_directions=incident_ray_directions,
                reflection_surface_normals=orientations[:, 0:4, 2].unsqueeze(1),
            )

            loss = (
                (
                    preferred_reflection_directions.squeeze(1)
                    - preferred_reflection_directions_calibration
                )
                .abs()
                .mean()
            )

            loss.backward()

            # Since each heliostat has multiple calibration datapoints, each heliostat appears multiple times
            # in the calibration group. The kinematic and actuator parameters are duplicated per datapoint for
            # each heliostat. Since each calibration has equal power on the gradients, the gradients of the parameters
            # for each heliostat are averaged.
            with torch.no_grad():
                for param_group in self.optimizer.param_groups:
                    for parameter in param_group["params"]:
                        for group in heliostat_duplicates_mapping:
                            averaged_gradients = parameter.grad[group].mean(dim=0, keepdim=True)
                            parameter.grad[group] = averaged_gradients

            self.optimizer.step()
            #self.scheduler.step(loss)

            if epoch % log_step == 0:
                log.info(
                    f"Epoch: {epoch}, Loss: {loss}, LR: {self.optimizer.param_groups[0]['lr']}",
                )

            epoch += 1


    def _optimize_kinematic_parameters_with_raytracing(
        self,
        tolerance: float,
        max_epoch: int,
        target_area_indices: torch.Tensor,
        centers_calibration_images: torch.Tensor,
        incident_ray_directions: torch.Tensor,
        num_log: int = 3,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """
        Optimize the kinematic parameters using ray tracing.

        This optimizer method optimizes the kinematic parameters by extracting the focus points
        of calibration images and using heliostat-tracing.

        Parameters
        ----------
        tolerance : float
            The optimizer tolerance.
        max_epoch : int
            The maximum number of optimization epochs.
        target_area_indices : torch.Tensor
            The indices of the target area for each calibration.
        centers_calibration_images : torch.Tensor
            The centers of the calibration flux densities.
        incident_ray_directions : torch.Tensor
            The incident ray directions specified in the calibration data.
        num_log : int
            Number of log messages during training (default is 3).
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).
        """
        log.info("Kinematic optimization with ray tracing.")
        device = torch.device(device)
        
        loss = torch.inf
        epoch = 0

        active_heliostats_indices = torch.arange(target_area_indices.shape[0], device=device)
        unique_heliostats = list(set(self.calibration_group.names))
        heliostat_duplicates_mapping = []
        for heliostat in unique_heliostats:
            heliostat_duplicates_mapping.append([i for i, s in enumerate(self.calibration_group.names) if s == heliostat])

        # Create a ray tracer.
        ray_tracer = HeliostatRayTracer(
            scenario=self.scenario,
            heliostat_group=self.calibration_group,
            batch_size=target_area_indices.shape[0]
        )

        # Start the optimization.
        log_step = max_epoch // num_log
        while loss > tolerance and epoch <= max_epoch:
            self.optimizer.zero_grad()

            # Align heliostats.
            self.calibration_group.align_surfaces_with_incident_ray_directions(
                incident_ray_directions=incident_ray_directions,
                active_heliostats_indices=active_heliostats_indices,
                device=device
            )

            # Perform heliostat-based ray tracing.
            flux_distributions = ray_tracer.trace_rays(
                incident_ray_directions=incident_ray_directions,
                active_heliostats_indices=active_heliostats_indices,
                target_area_indices=target_area_indices,
                device=device,
            )

            centers = utils.get_center_of_mass(
                bitmaps=flux_distributions,
                target_centers=self.scenario.target_areas.centers[target_area_indices],
                target_widths=self.scenario.target_areas.dimensions[target_area_indices][:, 0],
                target_heights=self.scenario.target_areas.dimensions[target_area_indices][:, 1],
                device=device,
            )

            loss = (centers - centers_calibration_images).abs().mean()
            loss.backward()

            # Since each heliostat has multiple calibration datapoints, each heliostat appears multiple times
            # in the calibration group. The kinematic and actuator parameters are duplicated per datapoint for
            # each heliostat. Since each calibration has equal power on the gradients, the gradients of the parameters
            # for each heliostat are averaged.
            with torch.no_grad():
                for param_group in self.optimizer.param_groups:
                    for parameter in param_group["params"]:
                        for group in heliostat_duplicates_mapping:
                            averaged_gradients = parameter.grad[group].mean(dim=0, keepdim=True)
                            parameter.grad[group] = averaged_gradients

            self.optimizer.step()
            self.scheduler.step(loss)

            if epoch % log_step == 0:
                log.info(
                    f"Epoch: {epoch}, Loss: {loss}, LR: {self.optimizer.param_groups[0]['lr']}",
                )

            epoch += 1

