import logging
import pathlib
from typing import Any, cast

import torch
from torch.optim.lr_scheduler import LRScheduler

from artist.core import learning_rate_schedulers
from artist.core.core_utils import per_heliostat_reduction
from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.core.loss_functions import Loss
from artist.data_parser.calibration_data_parser import CalibrationDataParser
from artist.field.heliostat_group import HeliostatGroup
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, index_mapping
from artist.util.environment_setup import get_device

log = logging.getLogger(__name__)
"""A logger for the kinematic reconstructor."""


class KinematicReconstructor:
    """
    An optimizer used to reconstruct real-world kinematic deviation parameters.

    The kinematic reconstructor learns kinematic parameters. These parameters are
    specific to a certain kinematic type and can for example include the four
    kinematic rotation deviation parameters as well as the two initial actuator parameters
    for each actuator of a rigid body kinematic.

    Attributes
    ----------
    ddp_setup : dict[str, Any]
        Information about the distributed environment, process_groups, devices, ranks, world_Size, heliostat group to ranks mapping.
    scenario : Scenario
        The scenario.
    data : dict[str, CalibrationDataParser | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]]
        The data parser and the mapping of heliostat name and calibration data.
    optimization_configuration : dict[str, Any]
        The parameters for the optimizer, learning rate scheduler, regularizers and early stopping.
    reconstruction_method : str
        The reconstruction method. Currently only reconstruction via ray tracing is available.

    Methods
    -------
    reconstruct_kinematic()
        Reconstruct the kinematic parameters.
    """

    def __init__(
        self,
        ddp_setup: dict[str, Any],
        scenario: Scenario,
        data: dict[
            str,
            CalibrationDataParser
            | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
        ],
        optimization_configuration: dict[str, Any],
        reconstruction_method: str = config_dictionary.kinematic_reconstruction_raytracing,
    ) -> None:
        """
        Initialize the kinematic optimizer.

        Parameters
        ----------
        ddp_setup : dict[str, Any]
            Information about the distributed environment, process_groups, devices, ranks, world_Size, heliostat group to ranks mapping.
        scenario : Scenario
            The scenario.
        data : dict[str, CalibrationDataParser | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]]
            The data parser and the mapping of heliostat name and calibration data.
        optimization_configuration : dict[str, Any]
            The parameters for the optimizer, learning rate scheduler, regularizers and early stopping.
        reconstruction_method : str
            The reconstruction method. Currently only reconstruction via ray tracing is available (default is ray_tracing).
        """
        rank = ddp_setup[config_dictionary.rank]
        if rank == 0:
            log.info("Create a kinematic reconstructor.")

        self.ddp_setup = ddp_setup
        self.scenario = scenario
        self.data = data
        self.optimization_configuration = optimization_configuration
        if (
            reconstruction_method
            == config_dictionary.kinematic_reconstruction_raytracing
        ):
            self.reconstruction_method = reconstruction_method
        else:
            raise ValueError(
                f"ARTIST currently only supports the {config_dictionary.kinematic_reconstruction_raytracing} reconstruction method. The reconstruction method {reconstruction_method} is not recognized. Please select another reconstruction method and try again!"
            )

    def reconstruct_kinematic(
        self,
        loss_definition: Loss,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Reconstruct the kinematic parameters.

        Parameters
        ----------
        loss_definition : Loss
            The definition of the loss function and pre-processing of the prediction.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The final loss of the kinematic reconstruction for each heliostat in each group.
            Tensor of shape [total_number_of_heliostats_in_scenario].
        """
        device = get_device(device=device)

        if (
            self.reconstruction_method
            == config_dictionary.kinematic_reconstruction_raytracing
        ):
            loss = self._reconstruct_kinematic_parameters_with_raytracing(
                loss_definition=loss_definition,
                device=device,
            )

        return loss

    def _reconstruct_kinematic_parameters_with_raytracing(
        self,
        loss_definition: Loss,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Reconstruct the kinematic parameters using ray tracing.

        This reconstruction method optimizes the kinematic parameters by extracting the focal points
        of calibration images and using heliostat-tracing.

        Parameters
        ----------
        loss_definition : Loss
            The definition of the loss function and pre-processing of the prediction.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The final loss of the kinematic reconstruction for each heliostat in each group.
            Tensor of shape [total_number_of_heliostats_in_scenario].
        """
        device = get_device(device=device)

        rank = self.ddp_setup[config_dictionary.rank]

        if rank == 0:
            log.info("Beginning kinematic reconstruction with ray tracing.")

        final_loss_per_heliostat = torch.full(
            (self.scenario.heliostat_field.number_of_heliostats_per_group.sum(),),
            torch.inf,
            device=device,
        )
        final_loss_start_indices = torch.cat(
            [
                torch.tensor([0], device=device),
                self.scenario.heliostat_field.number_of_heliostats_per_group.cumsum(
                    index_mapping.heliostat_dimension
                ),
            ]
        )

        for heliostat_group_index in self.ddp_setup[
            config_dictionary.groups_to_ranks_mapping
        ][rank]:
            heliostat_group: HeliostatGroup = (
                self.scenario.heliostat_field.heliostat_groups[heliostat_group_index]
            )
            parser = cast(
                CalibrationDataParser, self.data[config_dictionary.data_parser]
            )
            heliostat_mapping = cast(
                list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
                self.data[config_dictionary.heliostat_data_mapping],
            )
            (
                _,
                focal_spots_measured,
                incident_ray_directions,
                _,
                active_heliostats_mask,
                target_area_mask,
            ) = parser.parse_data_for_reconstruction(
                heliostat_data_mapping=heliostat_mapping,
                heliostat_group=heliostat_group,
                scenario=self.scenario,
                device=device,
            )

            if active_heliostats_mask.sum() > 0:
                # Create the optimizer.
                optimizer = torch.optim.Adam(
                    [
                        heliostat_group.kinematic.rotation_deviation_parameters.requires_grad_(),
                        heliostat_group.kinematic.actuators.optimizable_parameters.requires_grad_(),
                    ],
                    lr=self.optimization_configuration[
                        config_dictionary.initial_learning_rate
                    ],
                )

                # Create a learning rate scheduler.
                scheduler_fn = getattr(
                    learning_rate_schedulers,
                    self.optimization_configuration[config_dictionary.scheduler],
                )
                scheduler: LRScheduler = scheduler_fn(
                    optimizer=optimizer,
                    parameters=self.optimization_configuration[
                        config_dictionary.scheduler_parameters
                    ],
                )

                # Start the optimization.
                loss = torch.inf
                best_loss = torch.inf
                patience_counter = 0
                epoch = 0
                log_step = (
                    self.optimization_configuration[config_dictionary.max_epoch]
                    if self.optimization_configuration[config_dictionary.log_step] == 0
                    else self.optimization_configuration[config_dictionary.log_step]
                )
                while (
                    loss > self.optimization_configuration[config_dictionary.tolerance]
                    and epoch
                    <= self.optimization_configuration[config_dictionary.max_epoch]
                ):
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
                        world_size=self.ddp_setup[
                            config_dictionary.heliostat_group_world_size
                        ],
                        rank=self.ddp_setup[config_dictionary.heliostat_group_rank],
                        batch_size=heliostat_group.number_of_active_heliostats,
                        random_seed=self.ddp_setup[
                            config_dictionary.heliostat_group_rank
                        ],
                    )

                    # Perform heliostat-based ray tracing.
                    flux_distributions = ray_tracer.trace_rays(
                        incident_ray_directions=incident_ray_directions,
                        active_heliostats_mask=active_heliostats_mask,
                        target_area_mask=target_area_mask,
                        device=device,
                    )

                    if self.ddp_setup[config_dictionary.is_nested]:
                        flux_distributions = torch.distributed.nn.functional.all_reduce(
                            flux_distributions,
                            group=self.ddp_setup[config_dictionary.process_subgroup],
                            op=torch.distributed.ReduceOp.SUM,
                        )

                    loss_per_sample = loss_definition(
                        prediction=flux_distributions,
                        ground_truth=focal_spots_measured,
                        target_area_mask=target_area_mask,
                        reduction_dimensions=(index_mapping.focal_spots,),
                        device=device,
                    )

                    loss_per_heliostat = per_heliostat_reduction(
                        per_sample_values=loss_per_sample,
                        active_heliostats_mask=active_heliostats_mask,
                        device=device,
                    )

                    loss = loss_per_heliostat[torch.isfinite(loss_per_heliostat)].sum()

                    loss.backward()

                    if self.ddp_setup[config_dictionary.is_nested]:
                        # Reduce gradients within each heliostat group.
                        for param_group in optimizer.param_groups:
                            for param in param_group["params"]:
                                if param.grad is not None:
                                    param.grad = (
                                        torch.distributed.nn.functional.all_reduce(
                                            param.grad,
                                            op=torch.distributed.ReduceOp.SUM,
                                            group=self.ddp_setup[
                                                config_dictionary.process_subgroup
                                            ],
                                        )
                                    )

                    optimizer.step()
                    if isinstance(
                        scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        scheduler.step(loss.detach())
                    else:
                        scheduler.step()

                    if epoch % log_step == 0:
                        log.info(
                            f"Rank: {rank}, Epoch: {epoch}, Loss: {loss}, LR: {optimizer.param_groups[index_mapping.optimizer_param_group_0]['lr']}",
                        )

                    # Early stopping when loss has reached a plateau.
                    if (
                        loss
                        < best_loss
                        - self.optimization_configuration[
                            config_dictionary.early_stopping_delta
                        ]
                    ):
                        best_loss = loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    if (
                        patience_counter
                        >= self.optimization_configuration[
                            config_dictionary.early_stopping_patience
                        ]
                    ):
                        log.info(
                            f"Early stopping at epoch {epoch}. The loss did not improve significantly for {patience_counter} epochs."
                        )
                        break

                    epoch += 1

                final_loss_per_heliostat[
                    final_loss_start_indices[
                        heliostat_group_index
                    ] : final_loss_start_indices[heliostat_group_index + 1]
                ] = loss_per_heliostat

                log.info(f"Rank: {rank}, kinematic parameters optimized.")

        if self.ddp_setup[config_dictionary.is_distributed]:
            for index, heliostat_group in enumerate(
                self.scenario.heliostat_field.heliostat_groups
            ):
                source = self.ddp_setup[config_dictionary.ranks_to_groups_mapping][
                    index
                ]
                torch.distributed.broadcast(
                    heliostat_group.kinematic.rotation_deviation_parameters,
                    src=source[index_mapping.first_rank_from_group],
                )
                torch.distributed.broadcast(
                    heliostat_group.kinematic.actuators.optimizable_parameters,
                    src=source[index_mapping.first_rank_from_group],
                )
            torch.distributed.all_reduce(
                final_loss_per_heliostat, op=torch.distributed.ReduceOp.MIN
            )

            log.info(f"Rank: {rank}, synchronized after kinematic reconstruction.")

        return final_loss_per_heliostat
