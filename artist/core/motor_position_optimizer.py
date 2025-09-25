import logging
from typing import Any

import torch
from torch.optim.lr_scheduler import LRScheduler

from artist.core import learning_rate_schedulers
from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.core.loss_functions import Loss
from artist.field.heliostat_group import HeliostatGroup
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary
from artist.util.environment_setup import get_device

log = logging.getLogger(__name__)
"""A logger for the motor positions optimizer."""


class MotorPositionsOptimizer:
    """
    An optimizer used to find optimal motor positions for the heliostats.

    Attributes
    ----------
    ddp_setup : dict[str, Any]
        Information about the distributed environment, process_groups, devices, ranks, world_Size, heliostat group to ranks mapping.
    scenario : Scenario
        The scenario.
    optimization_configuration : dict[str, Any]
        The parameters for the optimizer, learning rate scheduler, regularizers and early stopping.
    incident_ray_direction : torch.Tensor
        The incident ray direction during the optimization.
        Tensor of shape [4].
    target_area_index : int
        The index of the target used for the optimization.
    ground_truth : torch.Tensor
        The desired focal spot or distribution.
        Tensor of shape [4] or tensor of shape [bitmap_resolution_e, bitmap_resolution_u].
    bitmap_resolution : torch.Tensor
        The resolution of all bitmaps during reconstruction.
        Tensor of shape [2].

    Methods
    -------
    optimize()
        Optimize the motor positions.
    """

    def __init__(
        self,
        ddp_setup: dict[str, Any],
        scenario: Scenario,
        optimization_configuration: dict[str, Any],
        incident_ray_direction: torch.Tensor,
        target_area_index: int,
        ground_truth: torch.Tensor,
        bitmap_resolution: torch.Tensor = torch.tensor([256, 256]),
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize the motor positions optimizer.

        Parameters
        ----------
        ddp_setup : dict[str, Any]
            Information about the distributed environment, process_groups, devices, ranks, world_Size, heliostat group to ranks mapping.
        scenario : Scenario
            The scenario.
        optimization_configuration : dict[str, Any]
            The parameters for the optimizer, learning rate scheduler, regularizers and early stopping.
        incident_ray_direction : torch.Tensor
            The incident ray direction during the optimization.
            Tensor of shape [4].
        target_area_index : int
            The index of the target used for the optimization.
        ground_truth : torch.Tensor
            The desired focal spot or distribution.
            Tensor of shape [4] or tensor of shape [bitmap_resolution_e, bitmap_resolution_u].
        bitmap_resolution : torch.Tensor
            The resolution of all bitmaps during optimization (default is torch.tensor([256,256])).
            Tensor of shape [2].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        device = get_device(device=device)

        rank = ddp_setup[config_dictionary.rank]

        if rank == 0:
            log.info("Create a motor positions optimizer.")

        self.ddp_setup = ddp_setup
        self.scenario = scenario
        self.optimization_configuration = optimization_configuration
        self.incident_ray_direction = incident_ray_direction
        self.target_area_index = target_area_index
        self.ground_truth = ground_truth
        self.bitmap_resolution = bitmap_resolution.to(device)

    def optimize(
        self,
        loss_definition: Loss,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        r"""
        Optimize the motor positions.

        The motor positions are optimized through a reparameterization to ensure stable training
        across different heliostats with widely varying initial motor positions and ranges. Motor
        positions can range from 0 to up to ~80000. Instead of directly optimizing the absolute
        motor positions, which can differ in magnitudes, an unconstrained parameter is optimized.
        Directly optimizing the absolute motor positions, would have very different effects depending
        on the scale of the motors. For small initial motor positions (e.g. ~100), a gradient update
        of size 10 may cause a ~10% relative change, drastically altering the motor positions of this
        heliostat. For large initial motor positions (e.g. ~50000), the same optimizer step would
        correspond to only a 0.02% relative change in motor positions, effectively freezing the
        optimization of this heliostat. This mismatch makes it impossible to choose a single learning
        rate that works robustly across all heliostats.
        The reparametrization of the optimizable parameter (motor positions) defines the optimizable
        parameter as:

        .. math::

            \text{motor\_positions\_optimized} = \tanh(
                \text{torch.nn.Parameter(optimizable\_parameter)}
            )

        The true motor positions can be reconstructed by:

        .. math::

            \text{motor\_positions} = \text{initial\_motor\_positions} +
            \text{motor\_positions\_normalized} \cdot \text{scale}

        where scale defines the range (e.g. up to ~80000) for adjustments.
        By optimizing as explained above instead of raw motor positions, every heliostat sees updates
        of comparable relative magnitude, regardless of the absolute size of its motors positions.

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
            The final loss of the motor position optimization.
        """
        device = get_device(device)

        rank = self.ddp_setup[config_dictionary.rank]

        if rank == 0:
            log.info("Start the motor positions optimization.")

        optimizable_parameters_all_groups = []
        scales_all_groups = []
        initial_motor_positions_all_groups = []

        active_heliostats_masks_all_groups = []
        target_area_masks_all_groups = []
        incident_ray_directions_all_groups = []

        for group_index, group in enumerate(
            self.scenario.heliostat_field.heliostat_groups
        ):
            active_heliostats_masks_all_groups.append(
                torch.ones(
                    group.number_of_heliostats,
                    dtype=torch.int32,
                    device=device,
                )
            )
            target_area_masks_all_groups.append(
                torch.full(
                    (group.number_of_heliostats,),
                    self.target_area_index,
                    dtype=torch.int32,
                    device=device,
                )
            )
            incident_ray_directions_all_groups.append(
                self.incident_ray_direction.repeat(group.number_of_heliostats, 1)
            )

            # Align all heliostats once, to the given incident ray direction and target, to set initial motor
            # positions. The motor positions are set automatically within the ``align_surfaces_with_incident_ray_directions()``
            # method.
            # Activate heliostats.
            group.activate_heliostats(
                active_heliostats_mask=active_heliostats_masks_all_groups[group_index],
                device=device,
            )

            # Align heliostats.
            group.align_surfaces_with_incident_ray_directions(
                aim_points=self.scenario.target_areas.centers[
                    target_area_masks_all_groups[group_index]
                ],
                incident_ray_directions=incident_ray_directions_all_groups[group_index],
                active_heliostats_mask=active_heliostats_masks_all_groups[group_index],
                device=device,
            )

            # Reparametrization of the motor positions (optimizable parameter).
            initial_motor_positions = (
                group.kinematic.active_motor_positions.detach().clone()
            )
            initial_motor_positions_all_groups.append(initial_motor_positions)
            motor_positions_minimum = group.kinematic.actuators.actuator_parameters[
                :, 2
            ]
            motor_positions_maximum = group.kinematic.actuators.actuator_parameters[
                :, 3
            ]
            lower_margin = initial_motor_positions - motor_positions_minimum
            upper_margin = motor_positions_maximum - initial_motor_positions
            scales_all_groups.append(
                torch.minimum(lower_margin, upper_margin).clamp(min=1.0)
            )

            # Create the optimizer.
            optimizable_parameters_all_groups.append(
                torch.nn.Parameter(
                    torch.zeros_like(initial_motor_positions, device=device)
                )
            )

        optimizer = torch.optim.Adam(
            optimizable_parameters_all_groups,
            lr=self.optimization_configuration[config_dictionary.initial_learning_rate],
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
        log_step = max(
            1,
            self.optimization_configuration[config_dictionary.max_epoch]
            // self.optimization_configuration[config_dictionary.num_log],
        )
        while (
            loss > self.optimization_configuration[config_dictionary.tolerance]
            and epoch <= self.optimization_configuration[config_dictionary.max_epoch]
        ):
            optimizer.zero_grad()

            total_flux = torch.zeros(
                (self.bitmap_resolution[0], self.bitmap_resolution[1]), device=device
            )

            for heliostat_group_index in self.ddp_setup[
                config_dictionary.groups_to_ranks_mapping
            ][rank]:
                heliostat_group: HeliostatGroup = (
                    self.scenario.heliostat_field.heliostat_groups[
                        heliostat_group_index
                    ]
                )

                # Reconstruct true motor positions from reparameterized version.
                motor_positions_normalized = torch.tanh(
                    optimizer.param_groups[0]["params"][heliostat_group_index]
                )
                heliostat_group.kinematic.motor_positions = (
                    initial_motor_positions_all_groups[heliostat_group_index]
                    + motor_positions_normalized
                    * scales_all_groups[heliostat_group_index]
                )

                # Activate heliostats.
                heliostat_group.activate_heliostats(
                    active_heliostats_mask=active_heliostats_masks_all_groups[
                        heliostat_group_index
                    ],
                    device=device,
                )

                # Align heliostats.
                heliostat_group.align_surfaces_with_motor_positions(
                    motor_positions=heliostat_group.kinematic.active_motor_positions,
                    active_heliostats_mask=active_heliostats_masks_all_groups[
                        heliostat_group_index
                    ],
                    device=device,
                )

                # Create a ray tracer.
                ray_tracer = HeliostatRayTracer(
                    scenario=self.scenario,
                    heliostat_group=heliostat_group,
                    world_size=self.ddp_setup[
                        config_dictionary.heliostat_group_world_size
                    ],
                    rank=self.ddp_setup[config_dictionary.heliostat_group_rank],
                    batch_size=heliostat_group.number_of_active_heliostats,
                    random_seed=self.ddp_setup[config_dictionary.heliostat_group_rank],
                    bitmap_resolution=self.bitmap_resolution,
                )

                # Perform heliostat-based ray tracing.
                flux_distributions = ray_tracer.trace_rays(
                    incident_ray_directions=incident_ray_directions_all_groups[
                        heliostat_group_index
                    ],
                    active_heliostats_mask=active_heliostats_masks_all_groups[
                        heliostat_group_index
                    ],
                    target_area_mask=target_area_masks_all_groups[
                        heliostat_group_index
                    ],
                    device=device,
                )

                if self.ddp_setup[config_dictionary.is_nested]:
                    flux_distributions = torch.distributed.nn.functional.all_reduce(
                        flux_distributions,
                        group=self.ddp_setup[config_dictionary.process_subgroup],
                        op=torch.distributed.ReduceOp.SUM,
                    )

                flux_distribution_on_target = ray_tracer.get_bitmaps_per_target(
                    bitmaps_per_heliostat=flux_distributions,
                    target_area_mask=target_area_masks_all_groups[
                        heliostat_group_index
                    ],
                    device=device,
                )[self.target_area_index]

                total_flux = total_flux + flux_distribution_on_target

            if self.ddp_setup[config_dictionary.is_distributed]:
                total_flux = torch.distributed.nn.functional.all_reduce(
                    total_flux,
                    op=torch.distributed.ReduceOp.SUM,
                )

            loss = loss_definition(
                prediction=total_flux.unsqueeze(0),
                ground_truth=self.ground_truth.unsqueeze(0),
                target_area_mask=torch.tensor([self.target_area_index], device=device),
                reduction_dimensions=(1,),
                device=device,
            ).sum()

            loss.backward()

            optimizer.step()
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss.detach())
            else:
                scheduler.step()

            if epoch % log_step == 0 and rank == 0:
                log.info(
                    f"Epoch: {epoch}, Loss: {loss.item()}, LR: {optimizer.param_groups[0]['lr']}",
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

        log.info(f"Rank: {rank}, motor positions optimized.")

        if self.ddp_setup[config_dictionary.is_distributed]:
            for index, heliostat_group in enumerate(
                self.scenario.heliostat_field.heliostat_groups
            ):
                source = self.ddp_setup[config_dictionary.ranks_to_groups_mapping][
                    index
                ]
                torch.distributed.broadcast(
                    heliostat_group.kinematic.motor_positions, src=source[0]
                )

            log.info(f"Rank: {rank}, synchronized after motor positions optimization.")

        return loss
