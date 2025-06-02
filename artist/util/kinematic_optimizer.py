import logging
from typing import Optional, Union

import torch
from torch.optim import Optimizer

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
    heliostat_group : HeliostatGroup
        The heliostat group to be calibrated.
    optimizer : Optimizer
        The optimizer.

    Methods
    -------
    optimize()
        Optimize the kinematic parameters.
    """

    def __init__(
        self,
        scenario: Scenario,
        heliostat_group: HeliostatGroup,
        optimizer: Optimizer,
    ) -> None:
        """
        Initialize the kinematic optimizer.

        Parameters
        ----------
        scenario : Scenario
            The scenario.
        heliostat_group : HeliostatGroup
            The heliostat group to be calibrated.
        optimizer : Optimizer
            The optimizer.
        """
        log.info("Create a kinematic optimizer.")
        self.scenario = scenario
        self.heliostat_group = heliostat_group
        self.optimizer = optimizer

    def optimize(
        self,
        focal_spots_calibration: torch.Tensor,
        incident_ray_directions: torch.Tensor,
        active_heliostats_mask: torch.Tensor,
        target_area_mask_calibration: Optional[torch.Tensor] = None,
        motor_positions_calibration: Optional[torch.Tensor] = None,
        tolerance: float = 5e-5,
        max_epoch: int = 10000,
        num_log: int = 3,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """
        Optimize the kinematic parameters.

        Parameters
        ----------
        focal_spots_calibration : torch.Tensor
            The center coordinates of the calibration flux densities.
        incident_ray_directions : torch.Tensor
            The incident ray directions specified in the calibrations.
        active_heliostats_mask : torch.Tensor
            A mask for the selected heliostats for calibration.
        target_area_mask_calibration : Optional[torch.Tensor]
            The indices of the target area for each calibration (default is None).
        motor_positions_calibration : Optional[torch.Tensor]
            The motor positions specified in the calibration files (default is None).
        tolerance : float
            The optimizer tolerance.
        max_epoch : int
            The maximum number of optimization epochs.
        num_log : int
            Number of log messages during training (default is 3).
        device : Union[torch.device, str] = "cuda"
            The device on which to initialize tensors (default is cuda).
        """
        log.info("Start the kinematic calibration.")

        device = torch.device(device)

        if motor_positions_calibration is not None:
            self._optimize_kinematic_parameters_with_motor_positions(
                focal_spots_calibration=focal_spots_calibration,
                incident_ray_directions=incident_ray_directions,
                active_heliostats_mask=active_heliostats_mask,
                motor_positions_calibration=motor_positions_calibration,
                tolerance=tolerance,
                max_epoch=max_epoch,
                num_log=num_log,
                device=device,
            )

        else:
            self._optimize_kinematic_parameters_with_raytracing(
                focal_spots_calibration=focal_spots_calibration,
                incident_ray_directions=incident_ray_directions,
                active_heliostats_mask=active_heliostats_mask,
                target_area_mask_calibration=target_area_mask_calibration,
                tolerance=tolerance,
                max_epoch=max_epoch,
                num_log=num_log,
                device=device,
            )

        log.info("Kinematic parameters optimized.")

    def _optimize_kinematic_parameters_with_motor_positions(
        self,
        focal_spots_calibration: torch.Tensor,
        incident_ray_directions: torch.Tensor,
        active_heliostats_mask: torch.Tensor,
        motor_positions_calibration: torch.Tensor,
        tolerance: float,
        max_epoch: int,
        num_log: int = 3,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """
        Optimize the kinematic parameters using the motor positions.

        This optimizer method optimizes the kinematic parameters by extracting the motor positions
        and incident ray directions from calibration data and using the scene's geometry.

        Parameters
        ----------
        focal_spots_calibration : torch.Tensor
            The center coordinates of the calibration flux densities.
        incident_ray_directions : torch.Tensor
            The incident ray directions specified in the calibration data.
        active_heliostats_mask : torch.Tensor
            A mask where 0 indicates a deactivated heliostat and 1 an activated one.
            An integer greater than 1 indicates that this heliostat is regarded multiple times.
        motor_positions_calibration : torch.Tensor
            The motor positions specified in the calibration data.
        tolerance : float
            The optimizer tolerance.
        max_epoch : int
            The maximum number of optimization epochs.
        num_log : int
            Number of log messages during training (default is 3).
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).
        """
        log.info("Kinematic calibration with motor positions.")

        device = torch.device(device)

        loss = torch.inf
        epoch = 0

        preferred_reflection_directions_calibration = torch.nn.functional.normalize(
            (
                focal_spots_calibration
                - self.heliostat_group.positions.repeat_interleave(
                    active_heliostats_mask, dim=0
                )
            ),
            p=2,
            dim=1,
        )

        log_step = max_epoch // num_log
        while loss > tolerance and epoch <= max_epoch:
            self.optimizer.zero_grad()

            # Activate heliostats
            self.heliostat_group.activate_heliostats(
                active_heliostats_mask=active_heliostats_mask
            )

            # Retrieve the orientation of the heliostats for given motor positions.
            orientations = self.heliostat_group.get_orientations_from_motor_positions(
                motor_positions=motor_positions_calibration,
                device=device,
            )

            # Determine the preferred reflection directions for each heliostat.
            preferred_reflection_directions = raytracing_utils.reflect(
                incident_ray_directions=incident_ray_directions,
                reflection_surface_normals=orientations[:, 0:4, 2],
            )

            loss = (
                (
                    preferred_reflection_directions
                    - preferred_reflection_directions_calibration
                )
                .abs()
                .mean()
            )

            loss.backward()

            self.optimizer.step()

            if epoch % log_step == 0:
                log.info(
                    f"Epoch: {epoch}, Loss: {loss}, LR: {self.optimizer.param_groups[0]['lr']}",
                )

            epoch += 1

    def _optimize_kinematic_parameters_with_raytracing(
        self,
        focal_spots_calibration: torch.Tensor,
        incident_ray_directions: torch.Tensor,
        active_heliostats_mask: torch.Tensor,
        target_area_mask_calibration: torch.Tensor,
        tolerance: float,
        max_epoch: int,
        num_log: int = 3,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """
        Optimize the kinematic parameters using ray tracing.

        This optimizer method optimizes the kinematic parameters by extracting the focus points
        of calibration images and using heliostat-tracing.

        Parameters
        ----------
        focal_spots_calibration : torch.Tensor
            The center coordinates of the calibration flux densities.
        incident_ray_directions : torch.Tensor
            The incident ray directions specified in the calibration data.
        active_heliostats_mask : torch.Tensor
            A mask where 0 indicates a deactivated heliostat and 1 an activated one.
            An integer greater than 1 indicates that this heliostat is regarded multiple times.
        target_area_mask_calibration : Optional[torch.Tensor]
            The indices of the target area for each calibration (default is None).
        tolerance : float
            The optimizer tolerance.
        max_epoch : int
            The maximum number of optimization epochs.
        num_log : int
            Number of log messages during training (default is 3).
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).
        """
        log.info("Kinematic optimization with ray tracing.")

        device = torch.device(device)

        loss = torch.inf
        epoch = 0

        # Start the optimization.
        log_step = max_epoch // num_log
        while loss > tolerance and epoch <= max_epoch:
            self.optimizer.zero_grad()

            # Align heliostats.
            self.heliostat_group.align_surfaces_with_incident_ray_directions(
                aim_points=self.scenario.target_areas.centers[
                    target_area_mask_calibration
                ],
                incident_ray_directions=incident_ray_directions,
                active_heliostats_mask=active_heliostats_mask,
                device=device,
            )

            # Create a ray tracer.
            ray_tracer = HeliostatRayTracer(
                scenario=self.scenario,
                heliostat_group=self.heliostat_group,
                batch_size=self.heliostat_group.number_of_active_heliostats,
            )

            # Perform heliostat-based ray tracing.
            flux_distributions = ray_tracer.trace_rays(
                incident_ray_directions=incident_ray_directions,
                active_heliostats_mask=active_heliostats_mask,
                target_area_mask=target_area_mask_calibration,
                device=device,
            )

            # Determine the focal spots of all flux density distributions
            focal_spots = utils.get_center_of_mass(
                bitmaps=flux_distributions,
                target_centers=self.scenario.target_areas.centers[
                    target_area_mask_calibration
                ],
                target_widths=self.scenario.target_areas.dimensions[
                    target_area_mask_calibration
                ][:, 0],
                target_heights=self.scenario.target_areas.dimensions[
                    target_area_mask_calibration
                ][:, 1],
                device=device,
            )

            loss = (focal_spots - focal_spots_calibration).abs().mean()
            loss.backward()

            self.optimizer.step()

            if epoch % log_step == 0:
                log.info(
                    f"Epoch: {epoch}, Loss: {loss.item()}, LR: {self.optimizer.param_groups[0]['lr']}",
                )

            epoch += 1
