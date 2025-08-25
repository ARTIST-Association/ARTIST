import logging
import pathlib

import torch

from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.data_loader import paint_loader
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, raytracing_utils, utils
from artist.util.environment_setup import DistributedEnvironmentTypedDict, get_device

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
    ddp_setup : DistributedEnvironmentTypedDict
        Information about the distributed environment, process_groups, devices, ranks, world_Size, heliostat group to ranks mapping.
    scenario : Scenario
        The scenario.
    data : dict[str, str | list[tuple[str, list[pathlib.Path, list[pathlib.Path]]]]]
        The data source name and the mapping of heliostat name and calibration data.
    calibration_method : str
        The calibration method. Either using ray tracing or motor positions (default is ray_tracing).
    initial_learning_rate : float
        The initial learning rate for the optimizer (default is 0.0004).
    tolerance : float
        The optimizer tolerance.
    max_epoch : int
        The maximum number of optimization epochs.
    num_log : int
        The number of log statements during optimization.

    Methods
    -------
    optimize()
        Optimize the kinematic parameters.
    """

    def __init__(
        self,
        ddp_setup: DistributedEnvironmentTypedDict,
        scenario: Scenario,
        data: dict[str, str | list[tuple[str, list[pathlib.Path, list[pathlib.Path]]]]],
        optimization_configuration: dict[str, float],
        calibration_method: str = config_dictionary.kinematic_calibration_raytracing,
    ) -> None:
        """
        Initialize the kinematic optimizer.

        Parameters
        ----------
        ddp_setup : DistributedEnvironmentTypedDict
            Information about the distributed environment, process_groups, devices, ranks, world_Size, heliostat group to ranks mapping.
        scenario : Scenario
            The scenario.
        data : dict[str, str | list[tuple[str, list[pathlib.Path, list[pathlib.Path]]]]]
            The data source name and the mapping of heliostat name and calibration data.
        optimization_configuration : dict[str, float]
            The parameters for the optimizer, learning rate scheduler and early stopping.
        calibration_method : str
            The calibration method. Either using ray tracing or motor positions (default is ray_tracing).
        """
        rank = ddp_setup["rank"]
        if rank == 0:
            log.info("Create a kinematic optimizer.")

        self.ddp_setup = ddp_setup
        self.scenario = scenario
        self.data = data
        self.calibration_method = calibration_method
        self.optimization_configuration = optimization_configuration

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

        rank = self.ddp_setup["rank"]

        if rank == 0:
            log.info("Beginning kinematic calibration with motor positions.")

        for heliostat_group_index in self.ddp_setup["groups_to_ranks_mapping"][rank]:
            heliostat_group = self.scenario.heliostat_field.heliostat_groups[
                heliostat_group_index
            ]

            # Load the calibration data.
            heliostat_calibration_mapping = []

            for heliostat, path_properties, _ in self.data[config_dictionary.heliostat_data_mapping]:
                if heliostat in heliostat_group.names:
                    heliostat_calibration_mapping.append((heliostat, path_properties))

            if self.data[config_dictionary.data_source] == config_dictionary.paint:
                (
                    focal_spots_measured,
                    incident_ray_directions,
                    motor_positions,
                    active_heliostats_mask,
                    _,
                ) = paint_loader.extract_paint_calibration_properties_data(
                    heliostat_calibration_mapping=heliostat_calibration_mapping,
                    heliostat_names=heliostat_group.names,
                    target_area_names=self.scenario.target_areas.names,
                    power_plant_position=self.scenario.power_plant_position,
                    device=device,
                )
            else:
                raise ValueError(
                    f"There is no data loader for the data source: {self.data[config_dictionary.data_source]}. Please use PAINT data instead."
                )

            if active_heliostats_mask.sum() > 0:
                # Calculate the reflection directions of the measured calibration data.
                preferred_reflection_directions_measured = (
                    torch.nn.functional.normalize(
                        (
                            focal_spots_measured
                            - heliostat_group.positions.repeat_interleave(
                                active_heliostats_mask, dim=0
                            )
                        ),
                        p=2,
                        dim=1,
                    )
                )

                # Create the optimizer.
                optimizer = torch.optim.Adam(
                    [
                        heliostat_group.kinematic.deviation_parameters.requires_grad_(),
                        heliostat_group.kinematic.actuators.actuator_parameters.requires_grad_(),
                    ],
                    lr=self.optimization_configuration[config_dictionary.initial_learning_rate],
                )

                # Start the optimization.
                loss = torch.inf
                best_loss = torch.inf
                patience_counter = 0
                epoch = 0
                log_step = self.optimization_configuration[config_dictionary.max_epoch] // self.optimization_configuration[config_dictionary.num_log]
                while loss > self.optimization_configuration[config_dictionary.tolerance] and epoch <= self.optimization_configuration[config_dictionary.max_epoch]:
                    optimizer.zero_grad()

                    # Activate heliostats.
                    heliostat_group.activate_heliostats(
                        active_heliostats_mask=active_heliostats_mask, device=device
                    )

                    # Retrieve the orientation of the heliostats for given motor positions.
                    orientations = (
                        heliostat_group.kinematic.motor_positions_to_orientations(
                            motor_positions=motor_positions,
                            device=device,
                        )
                    )

                    # Determine the preferred reflection directions for each heliostat.
                    preferred_reflection_directions = raytracing_utils.reflect(
                        incident_ray_directions=incident_ray_directions,
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

                    optimizer.step()

                    if epoch % log_step == 0:
                        log.info(
                            f"Epoch: {epoch}, Loss: {loss}, LR: {optimizer.param_groups[0]['lr']}",
                        )

                    # Early stopping when loss has reached a plateau.
                    if loss < best_loss - self.optimization_configuration[config_dictionary.early_stopping_delta]:
                        best_loss = loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    if patience_counter > self.optimization_configuration[config_dictionary.early_stopping_patience]:
                        log.info(f"Early stopping at epoch {epoch}. The loss did not improve significantly for {self.optimization_configuration[config_dictionary.early_stopping_patience]} epochs.")
                        break

                    epoch += 1

                log.info("Kinematic parameters optimized.")

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

        rank = self.ddp_setup["rank"]

        if rank == 0:
            log.info("Beginning kinematic optimization with ray tracing.")

        for heliostat_group_index in self.ddp_setup["groups_to_ranks_mapping"][rank]:
            heliostat_group = self.scenario.heliostat_field.heliostat_groups[
                heliostat_group_index
            ]

            # Load the calibration data.
            heliostat_calibration_mapping = []

            for heliostat, path_properties, _ in self.heliostat_data_mapping:
                if heliostat in heliostat_group.names:
                    heliostat_calibration_mapping.append((heliostat, path_properties))

            if self.data[config_dictionary.data_source] == config_dictionary.paint:
                (
                    focal_spots_measured,
                    incident_ray_directions,
                    _,
                    active_heliostats_mask,
                    target_area_mask,
                ) = paint_loader.extract_paint_calibration_properties_data(
                    heliostat_calibration_mapping=heliostat_calibration_mapping,
                    heliostat_names=heliostat_group.names,
                    target_area_names=self.scenario.target_areas.names,
                    power_plant_position=self.scenario.power_plant_position,
                    device=device,
                )
            else:
                raise ValueError(
                    f"There is no data loader for the data source: {self.data[config_dictionary.data_source]}. Please use PAINT data instead."
                )

            if active_heliostats_mask.sum() > 0:
                # Create the optimizer.
                optimizer = torch.optim.Adam(
                    [
                        heliostat_group.kinematic.deviation_parameters.requires_grad_(),
                        heliostat_group.kinematic.actuators.actuator_parameters.requires_grad_(),
                    ],
                    lr=self.optimization_configuration[config_dictionary.initial_learning_rate],
                )

                # Start the optimization.
                loss = torch.inf
                best_loss = torch.inf
                patience_counter = 0
                epoch = 0
                log_step = self.optimization_configuration[config_dictionary.max_epoch] // self.optimization_configuration[config_dictionary.num_log]
                while loss > self.optimization_configuration[config_dictionary.tolerance] and epoch <= self.optimization_configuration[config_dictionary.max_epoch]:
                    optimizer.zero_grad()

                    # Activate heliostats.
                    heliostat_group.activate_heliostats(
                        active_heliostats_mask=active_heliostats_mask, device=device
                    )

                    # Align heliostats.
                    heliostat_group.align_surfaces_with_incident_ray_directions(
                        aim_points=self.scenario.target_areas.centers[target_area_mask],
                        incident_ray_directions=incident_ray_directions,
                        active_heliostats_mask=active_heliostats_mask,
                        device=device,
                    )

                    # Create a parallelized ray tracer.
                    ray_tracer = HeliostatRayTracer(
                        scenario=self.scenario,
                        heliostat_group=heliostat_group,
                        world_size=self.ddp_setup["heliostat_group_world_size"],
                        rank=self.ddp_setup["heliostat_group_rank"],
                        batch_size=heliostat_group.number_of_active_heliostats,
                        random_seed=self.ddp_setup["heliostat_group_rank"],
                    )

                    # Perform heliostat-based ray tracing.
                    flux_distributions = ray_tracer.trace_rays(
                        incident_ray_directions=incident_ray_directions,
                        active_heliostats_mask=active_heliostats_mask,
                        target_area_mask=target_area_mask,
                        device=device,
                    )

                    if self.ddp_setup["is_nested"]:
                        flux_distributions = torch.distributed.nn.functional.all_reduce(
                            flux_distributions,
                            group=self.ddp_setup["process_subgroup"],
                            op=torch.distributed.ReduceOp.SUM,
                        )

                    # Determine the focal spots of all flux density distributions
                    focal_spots = utils.get_center_of_mass(
                        bitmaps=flux_distributions,
                        target_centers=self.scenario.target_areas.centers[
                            target_area_mask
                        ],
                        target_widths=self.scenario.target_areas.dimensions[
                            target_area_mask
                        ][:, 0],
                        target_heights=self.scenario.target_areas.dimensions[
                            target_area_mask
                        ][:, 1],
                        device=device,
                    )

                    loss = (focal_spots - focal_spots_measured).abs().mean()
                    loss.backward()

                    if self.ddp_setup["is_nested"]:
                        # Reduce gradients within each heliostat group.
                        for param_group in optimizer.param_groups:
                            for param in param_group["params"]:
                                if param.grad is not None:
                                    param.grad = (
                                        torch.distributed.nn.functional.all_reduce(
                                            param.grad,
                                            op=torch.distributed.ReduceOp.SUM,
                                            group=self.ddp_setup["process_subgroup"],
                                        )
                                    )

                    optimizer.step()

                    if epoch % log_step == 0:
                        log.info(
                            f"Rank: {rank}, Epoch: {epoch}, Loss: {loss}, LR: {optimizer.param_groups[0]['lr']}",
                        )
                    
                    # Early stopping when loss has reached a plateau.
                    if loss < best_loss - self.optimization_configuration[config_dictionary.early_stopping_delta]:
                        best_loss = loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    if patience_counter > self.optimization_configuration[config_dictionary.early_stopping_patience]:
                        log.info(f"Early stopping at epoch {epoch}. The loss did not improve significantly for {self.optimization_configuration[config_dictionary.early_stopping_patience]} epochs.")
                        break

                    epoch += 1

                log.info(f"Rank: {rank}, kinematic parameters optimized.")

        if self.ddp_setup["is_distributed"]:
            for index, heliostat_group in enumerate(
                self.scenario.heliostat_field.heliostat_groups
            ):
                source = self.ddp_setup["ranks_to_groups_mapping"][index]
                torch.distributed.broadcast(
                    heliostat_group.kinematic.deviation_parameters, src=source[0]
                )
                torch.distributed.broadcast(
                    heliostat_group.kinematic.actuators.actuator_parameters,
                    src=source[0],
                )

            log.info(f"Rank: {rank}, synchronised after kinematic calibration.")
