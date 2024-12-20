import logging
from typing import Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

from artist.raytracing import raytracing_utils
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.scenario import Scenario
from artist.util import utils

log = logging.getLogger(__name__)
"""A logger for the alignment optimizer."""


class AlignmentOptimizer:
    """
    Alignment optimizer to find optimal kinematic parameters.

    The alignment optimizer optimizes parameters of the rigid body kinematics.
    These parameters can include the 18 kinematic deviations parameters as well as five actuator
    parameters for each actuator.

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
        Initialize the alignment optimizer.

        Parameters
        ----------
        scenario : Scenario
            The scenario.
        optimizer : Optimizer
            The optimizer.
        scheduler : Union[_LRScheduler, ReduceLROnPlateau]
            The learning rate scheduler.
        """
        log.info("Create alignment optimizer.")
        self.scenario = scenario
        self.optimizer = optimizer
        self.scheduler = scheduler

    def optimize(
        self,
        tolerance: float,
        max_epoch: int,
        center_calibration_image: torch.Tensor,
        incident_ray_direction: torch.Tensor,
        motor_positions: Optional[torch.Tensor] = None,
        num_log: int = 3,
        device: Union[torch.device, str] = "cuda",
    ) -> tuple[list[torch.Tensor], Scenario]:
        """
        Optimize the kinematic parameters.

        Parameters
        ----------
        tolerance : float
            The optimizer tolerance.
        max_epoch : int
            The maximum number of optimization epochs.
        center_calibration_image : torch.Tensor
            The center of the calibration flux density.
        incident_ray_direction : torch.Tensor
            The incident ray direction specified in the calibration.
        motor_positions : Optional[torch.Tensor]
            The motor positions specified in the calibration (default is ``None``).
        num_log : int
            Number of log messages during training (default is 3).
        device : Union[torch.device, str] = "cuda"
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        list[torch.Tensor]
            The list of optimized kinematic parameters.
        Scenario
            The scenario with aligned heliostat and optimized kinematic parameters.
        """
        log.info("Start alignment optimization.")
        device = torch.device(device)

        if motor_positions is not None:
            optimized_parameters, self.scenario = (
                self._optimize_kinematic_parameters_with_motor_positions(
                    tolerance=tolerance,
                    max_epoch=max_epoch,
                    center_calibration_image=center_calibration_image,
                    incident_ray_direction=incident_ray_direction,
                    motor_positions=motor_positions,
                    num_log=num_log,
                    device=device,
                )
            )
        else:
            optimized_parameters, self.scenario = (
                self._optimize_kinematic_parameters_with_raytracing(
                    tolerance=tolerance,
                    max_epoch=max_epoch,
                    center_calibration_image=center_calibration_image,
                    incident_ray_direction=incident_ray_direction,
                    num_log=num_log,
                    device=device,
                )
            )
        log.info("Alignment optimized.")
        return optimized_parameters, self.scenario

    def _optimize_kinematic_parameters_with_motor_positions(
        self,
        tolerance: float,
        max_epoch: int,
        center_calibration_image: torch.Tensor,
        incident_ray_direction: torch.Tensor,
        motor_positions: torch.Tensor,
        num_log: int = 3,
        device: Union[torch.device, str] = "cuda",
    ) -> tuple[list[torch.Tensor], Scenario]:
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
        center_calibration_image : torch.Tensor
            The center of the calibration flux density.
        incident_ray_direction : torch.Tensor
            The incident ray direction specified in the calibration.
        motor_positions : torch.Tensor
            The motor positions specified in the calibration.
        num_log : int
            Number of log messages during training (default is 3).
        device : Union[torch.device, str]
            The device on which to initialize tensors (default: cuda).

        Returns
        -------
        list[torch.Tensor]
            The list of optimized parameters.
        Scenario
            The scenario with aligned heliostat and optimized kinematic parameters.
        """
        log.info("Alignment optimization with motor positions.")
        device = torch.device(device)
        loss = torch.inf
        epoch = 0

        preferred_reflection_direction_calibration = (
            center_calibration_image
            - self.scenario.heliostats.heliostat_list[0].position
        )
        preferred_reflection_direction_calibration = (
            preferred_reflection_direction_calibration
            / torch.norm(preferred_reflection_direction_calibration)
        )

        log_step = max_epoch // num_log
        while loss > tolerance and epoch <= max_epoch:
            orientation = self.scenario.heliostats.heliostat_list[
                0
            ].get_orientation_from_motor_positions(
                motor_positions=motor_positions, device=device
            )

            preferred_reflection_direction = raytracing_utils.reflect(
                -incident_ray_direction, orientation[0:4, 2]
            )

            self.optimizer.zero_grad()

            loss = (
                (
                    preferred_reflection_direction
                    - preferred_reflection_direction_calibration
                )
                .abs()
                .mean()
            )
            loss.backward()

            self.optimizer.step()
            self.scheduler.step(loss)

            if epoch % log_step == 0:
                log.info(
                    f"Epoch: {epoch}, Loss: {loss.item()}, LR: {self.optimizer.param_groups[0]['lr']}",
                )

            epoch += 1

        self.scenario.heliostats.heliostat_list[
            0
        ].set_aligned_surface_with_motor_positions(
            motor_positions=motor_positions.to(device), device=device
        )

        return self.optimizer.param_groups[0]["params"], self.scenario

    def _optimize_kinematic_parameters_with_raytracing(
        self,
        tolerance: float,
        max_epoch: int,
        center_calibration_image: torch.Tensor,
        incident_ray_direction: torch.Tensor,
        num_log: int = 3,
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
            The optimzer tolerance.
        max_epoch : int
            The maximum number of optimization epochs.
        center_calibration_image : torch.Tensor
            The center of the calibration flux density.
        incident_ray_direction : torch.Tensor
            The incident ray direction specified in the calibration.
        num_log : int
            Number of log messages during training (default is 3).
        device : Union[torch.device, str]
            The device on which to initialize tensors (default: cuda).

        Returns
        -------
        list[torch.Tensor]
            The list of optimized parameters.
        Scenario
            The scenario with aligned heliostat and optimized kinematic parameters.
        """
        log.info("Alignment optimization with raytracing.")
        device = torch.device(device)
        loss = torch.inf
        epoch = 0

        log_step = max_epoch // num_log
        while loss > tolerance and epoch <= max_epoch:
            # Align heliostat.
            self.scenario.heliostats.heliostat_list[
                0
            ].set_aligned_surface_with_incident_ray_direction(
                incident_ray_direction=incident_ray_direction, device=device
            )

            # Create raytracer.
            raytracer = HeliostatRayTracer(scenario=self.scenario)

            final_bitmap = raytracer.trace_rays(
                incident_ray_direction=incident_ray_direction, device=device
            )

            final_bitmap = raytracer.normalize_bitmap(final_bitmap)

            center = utils.get_center_of_mass(
                bitmap=torch.flip(final_bitmap, dims=(0, 1)),
                target_center=self.scenario.receivers.receiver_list[0].position_center,
                plane_e=self.scenario.receivers.receiver_list[0].plane_e,
                plane_u=self.scenario.receivers.receiver_list[0].plane_u,
                device=device,
            )

            self.optimizer.zero_grad()

            loss = (center - center_calibration_image).abs().mean()
            loss.backward()

            self.optimizer.step()
            self.scheduler.step(loss)

            if epoch % log_step == 0:
                log.info(
                    f"Epoch: {epoch}, Loss: {loss.item()}, LR: {self.optimizer.param_groups[0]['lr']}",
                )

            epoch += 1

        return self.optimizer.param_groups[0]["params"], self.scenario
