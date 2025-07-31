import logging
import pathlib

import torch

from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.data_loader import paint_loader
from artist.field.heliostat_group import HeliostatGroup
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, raytracing_utils, utils
from artist.util.environment_setup import get_device

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
    calibration_method : str
        The calibration method. Either using ray tracing or motor positions (default is ray_tracing).
    focal_spots_measured : torch.Tensor
        The center coordinates of the calibration flux densities.
    incident_ray_directions : torch.Tensor
        The incident ray directions specified in the calibrations.
    motor_positions : torch.Tensor
        The motor positions specified in the calibration files.
    heliostats_mask : torch.Tensor
        A mask for the selected heliostats for calibration.
    target_area_mask : torch.Tensor
        The indices of the target area for each calibration.
    num_log : int
        The number of log statements during optimization.
    initial_learning_rate : float
        The initial learning rate for the optimizer (default is 0.0004).
    tolerance : float
        The optimizer tolerance.
    max_epoch : int
        The maximum number of optimization epochs.
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
        heliostat_data_mapping: list[
            tuple[str, list[pathlib.Path], list[pathlib.Path]]
        ],
        calibration_method: str = config_dictionary.kinematic_calibration_raytracing,
        initial_learning_rate: float = 0.0004,
        tolerance: float = 0.035,
        max_epoch: int = 600,
        num_log: int = 3,
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize the kinematic optimizer.

        Parameters
        ----------
        scenario : Scenario
            The scenario.
        heliostat_group : HeliostatGroup
            The heliostat group to be calibrated.
        heliostat_data_mapping : list[tuple[str, list[pathlib.Path, list[pathlib.Path]]]
            The mapping of heliostat and calibration data.
        calibration_method : str
            The calibration method. Either using ray tracing or motor positions (default is ray_tracing).
        initial_learning_rate : float
            The initial learning rate for the optimizer (default is 0.0004).
        tolerance : float
            The tolerance during optimization (default is 0.035).
        max_epoch : int
            The maximum optimization epoch (default is 600).
        num_log : int
            The number of log statements during optimization (default is 3).
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        device = get_device(device=device)

        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        if rank == 0:
            log.info("Create a kinematic optimizer.")

        self.scenario = scenario
        self.heliostat_group = heliostat_group
        self.calibration_method = calibration_method

        heliostat_calibration_mapping = [
            (heliostat_name, calibration_properties_paths)
            for heliostat_name, calibration_properties_paths, _ in heliostat_data_mapping
            if heliostat_name in self.heliostat_group.names
        ]

        # Load the calibration data.
        (
            self.focal_spots_measured,
            self.incident_ray_directions,
            self.motor_positions,
            self.heliostats_mask,
            self.target_area_mask,
        ) = paint_loader.extract_paint_calibration_properties_data(
            heliostat_calibration_mapping=heliostat_calibration_mapping,
            heliostat_names=heliostat_group.names,
            target_area_names=scenario.target_areas.names,
            power_plant_position=scenario.power_plant_position,
            device=device,
        )

        if (
            self.calibration_method
            == config_dictionary.kinematic_calibration_raytracing
        ):
            self.motor_positions = None

        self.num_log = num_log

        # Create the optimizer.
        self.initial_learning_rate = initial_learning_rate
        self.tolerance = tolerance
        self.max_epoch = max_epoch

        self.optimizer = torch.optim.Adam(
            [
                self.heliostat_group.kinematic.deviation_parameters.requires_grad_(),
                self.heliostat_group.kinematic.actuators.actuator_parameters.requires_grad_(),
            ],
            lr=self.initial_learning_rate,
        )

    def optimize(
        self,
        device: torch.device | None = None,
    ) -> None:
        """
        Optimize the kinematic parameters.

        Parameters
        ----------
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        device = get_device(device=device)

        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        if rank == 0:
            log.info("Start the kinematic calibration.")

        if self.heliostats_mask.sum() > 0:
            if (
                self.calibration_method
                == config_dictionary.kinematic_calibration_motor_positions
            ):
                self._optimize_kinematic_parameters_with_motor_positions(
                    device=device,
                )

            if (
                self.calibration_method
                == config_dictionary.kinematic_calibration_raytracing
            ):
                self._optimize_kinematic_parameters_with_raytracing(
                    device=device,
                )

    def _optimize_kinematic_parameters_with_motor_positions(
        self,
        device: torch.device | None = None,
    ) -> None:
        """
        Optimize the kinematic parameters using the motor positions.

        This optimizer method optimizes the kinematic parameters by extracting the motor positions
        and incident ray directions from calibration data and using the scene's geometry.

        Parameters
        ----------
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        device = get_device(device=device)

        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        if rank == 0:
            log.info("Kinematic calibration with motor positions.")

        preferred_reflection_directions_measured = torch.nn.functional.normalize(
            (
                self.focal_spots_measured
                - self.heliostat_group.positions.repeat_interleave(
                    self.heliostats_mask, dim=0
                )
            ),
            p=2,
            dim=1,
        )

        loss = torch.inf
        epoch = 0
        log_step = self.max_epoch // self.num_log
        while loss > self.tolerance and epoch <= self.max_epoch:
            self.optimizer.zero_grad()

            # Activate heliostats
            self.heliostat_group.activate_heliostats(
                active_heliostats_mask=self.heliostats_mask, device=device
            )

            # Retrieve the orientation of the heliostats for given motor positions.
            orientations = (
                self.heliostat_group.kinematic.motor_positions_to_orientations(
                    motor_positions=self.motor_positions,
                    device=device,
                )
            )

            # Determine the preferred reflection directions for each heliostat.
            preferred_reflection_directions = raytracing_utils.reflect(
                incident_ray_directions=self.incident_ray_directions,
                reflection_surface_normals=orientations[:, 0:4, 2],
            )

            loss = (
                (
                    preferred_reflection_directions
                    - preferred_reflection_directions_measured
                )
                .abs()
                .mean()
            )

            loss.backward()

            self.optimizer.step()

            if epoch % log_step == 0:
                log.info(
                    f"Rank: {rank}, Epoch: {epoch}, Loss: {loss}, LR: {self.optimizer.param_groups[0]['lr']}",
                )

            epoch += 1

        log.info(f"Kinematic parameters of group {rank} optimized.")

    def _optimize_kinematic_parameters_with_raytracing(
        self,
        device: torch.device | None = None,
    ) -> None:
        """
        Optimize the kinematic parameters using ray tracing.

        This optimizer method optimizes the kinematic parameters by extracting the focus points
        of calibration images and using heliostat-tracing.

        Parameters
        ----------
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        device = get_device(device=device)

        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        if rank == 0:
            log.info("Kinematic optimization with ray tracing.")

        loss = torch.inf
        epoch = 0

        # Start the optimization.
        log_step = self.max_epoch // self.num_log
        while loss > self.tolerance and epoch <= self.max_epoch:
            self.optimizer.zero_grad()

            self.heliostat_group.activate_heliostats(
                active_heliostats_mask=self.heliostats_mask, device=device
            )

            # Align heliostats.
            self.heliostat_group.align_surfaces_with_incident_ray_directions(
                aim_points=self.scenario.target_areas.centers[self.target_area_mask],
                incident_ray_directions=self.incident_ray_directions,
                active_heliostats_mask=self.heliostats_mask,
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
                incident_ray_directions=self.incident_ray_directions,
                active_heliostats_mask=self.heliostats_mask,
                target_area_mask=self.target_area_mask,
                device=device,
            )

            # Determine the focal spots of all flux density distributions
            focal_spots = utils.get_center_of_mass(
                bitmaps=flux_distributions,
                target_centers=self.scenario.target_areas.centers[
                    self.target_area_mask
                ],
                target_widths=self.scenario.target_areas.dimensions[
                    self.target_area_mask
                ][:, 0],
                target_heights=self.scenario.target_areas.dimensions[
                    self.target_area_mask
                ][:, 1],
                device=device,
            )

            loss = (focal_spots - self.focal_spots_measured).abs().mean()
            loss.backward()

            self.optimizer.step()

            if epoch % log_step == 0:
                log.info(
                    f"Rank: {rank}, Epoch: {epoch}, Loss: {loss}, LR: {self.optimizer.param_groups[0]['lr']}",
                )

            epoch += 1

        log.info(f"Kinematic parameters of group {rank} optimized.")
