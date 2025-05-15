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
        center_calibration_images: torch.Tensor,
        incident_ray_directions: torch.Tensor,
        target_area_indices: Optional[torch.Tensor] = None,
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
        calibration_target_names : Optional[list[str]]
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

        elif motor_positions is None:
            self._optimize_kinematic_parameters_with_raytracing(
                tolerance=tolerance,
                max_epoch=max_epoch,
                target_area_indices=target_area_indices,
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
            The optimizer tolerance.
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
            ),
            p=2,
            dim=1,
        )

        log_step = max_epoch // num_log
        while loss > tolerance and epoch <= max_epoch:
            total_loss = 0.0
            self.optimizer.zero_grad()

            for (
                motor_positions,
                incident_ray_direction,
                preferred_reflection_direction_calibration,
            ) in zip(
                all_motor_positions,
                incident_ray_directions,
                preferred_reflection_direction_calibrations,
            ):
                orientation = (
                    self.scenario.heliostat_field.get_orientations_from_motor_positions(
                        motor_positions=motor_positions, device=device
                    )
                )

                preferred_reflection_direction = raytracing_utils.reflect(
                    incoming_ray_direction=incident_ray_direction,
                    reflection_surface_normals=orientation[:, 0:4, 2],
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
                    f"Epoch: {epoch}, Loss: {total_loss}, LR: {self.optimizer.param_groups[0]['lr']}",
                )

            epoch += 1

    def _optimize_kinematic_parameters_with_raytracing(
        self,
        tolerance: float,
        max_epoch: int,
        target_area_indices: torch.Tensor,
        center_calibration_images: torch.Tensor,
        incident_ray_directions: torch.Tensor,
        num_log: int = 3,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """
        Optimize the kinematic parameters using ray tracing.

        This optimizer method optimizes the kinematic parameters by extracting the focus point
        of a calibration image and using heliostat-tracing. This method is slower than the other
        optimization method found in the kinematic optimizer.

        Parameters
        ----------
        tolerance : float
            The optimizer tolerance.
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

            loss = (centers - center_calibration_images).abs().mean()
            loss.backward()

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

        return self.calibration_group.kinematic_deviation_parameters, self.calibration_group.actuator_parameters
