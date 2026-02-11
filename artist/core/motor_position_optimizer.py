import logging
from typing import Any

import torch
from torch.optim.lr_scheduler import LRScheduler

from artist.core import learning_rate_schedulers
from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.core.loss_functions import FocalSpotLoss, KLDivergenceLoss, Loss
from artist.field.heliostat_group import HeliostatGroup
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, index_mapping
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
        constraint_parameters: dict[str, Any],
        incident_ray_direction: torch.Tensor,
        target_area_index: int,
        ground_truth: torch.Tensor,
        dni: float,
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
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        device = get_device(device=device)

        rank = ddp_setup[config_dictionary.rank]

        if rank == 0:
            log.info("Create a motor positions optimizer.")

        self.ddp_setup = ddp_setup
        self.scenario = scenario
        self.optimization_configuration = optimization_configuration
        self.constraint_parameters = constraint_parameters
        self.incident_ray_direction = incident_ray_direction
        self.target_area_index = target_area_index
        self.ground_truth = ground_truth
        self.dni = dni
        self.bitmap_resolution = bitmap_resolution.to(device)

        self.epsilon = 1e-12

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
            If None, ``ARTIST`` will automatically select the most appropriate
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

        heliostat_surface_areas = []

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

            # Align all heliostats once, to the given incident ray direction and target, to set initial motor positions.
            # The motor positions are set automatically within the align_surfaces_with_incident_ray_directions() method.
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
            motor_positions_minimum = (
                group.kinematic.actuators.non_optimizable_parameters[
                    :, index_mapping.actuator_min_motor_position
                ]
            )
            motor_positions_maximum = (
                group.kinematic.actuators.non_optimizable_parameters[
                    :, index_mapping.actuator_max_motor_position
                ]
            )
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

            # Calculate surface area of all heliostats in the group.
            canting_norm = (torch.norm(group.canting[0], dim=1)[0])[:2]
            dimensions = (canting_norm * 4) + 0.02
            heliostat_surface_areas.append(dimensions[0] * dimensions[1])

        optimizer = torch.optim.Adam(
            optimizable_parameters_all_groups,
            lr=float(
                self.optimization_configuration[config_dictionary.initial_learning_rate]
            ),
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

        # Set up early stopping.
        early_stopper = learning_rate_schedulers.EarlyStopping(
            window_size=self.optimization_configuration[
                config_dictionary.early_stopping_window
            ],
            patience=self.optimization_configuration[
                config_dictionary.early_stopping_patience
            ],
            min_improvement=self.optimization_configuration[
                config_dictionary.early_stopping_delta
            ],
            relative=True,
        )

        lambda_energy = None
        rho_energy = self.constraint_parameters["rho_energy"]
        max_flux_density = self.constraint_parameters["max_flux_density"]
        rho_pixel = self.constraint_parameters["rho_pixel"]
        lambda_lr = self.constraint_parameters["lambda_lr"]

        # Start the optimization.
        loss = torch.inf
        epoch = 0
        log_step = (
            self.optimization_configuration[config_dictionary.max_epoch]
            if self.optimization_configuration[config_dictionary.log_step] == 0
            else self.optimization_configuration[config_dictionary.log_step]
        )
        while (
            loss > float(self.optimization_configuration[config_dictionary.tolerance])
            and epoch <= self.optimization_configuration[config_dictionary.max_epoch]
        ):
            optimizer.zero_grad()

            total_flux = torch.zeros(
                (
                    self.bitmap_resolution[index_mapping.unbatched_bitmap_e],
                    self.bitmap_resolution[index_mapping.unbatched_bitmap_u],
                ),
                device=device,
            )

            for heliostat_group_index in self.ddp_setup[
                config_dictionary.groups_to_ranks_mapping
            ][rank]:
                heliostat_alignment_group: HeliostatGroup = (
                    self.scenario.heliostat_field.heliostat_groups[
                        heliostat_group_index
                    ]
                )

                # Reconstruct true motor positions from reparameterized version.
                motor_positions_normalized = torch.tanh(
                    optimizer.param_groups[index_mapping.optimizer_param_group_0][
                        "params"
                    ][heliostat_group_index]
                )
                heliostat_alignment_group.kinematic.motor_positions = (
                    initial_motor_positions_all_groups[heliostat_group_index]
                    + motor_positions_normalized
                    * scales_all_groups[heliostat_group_index]
                )

                # Activate heliostats.
                heliostat_alignment_group.activate_heliostats(
                    active_heliostats_mask=active_heliostats_masks_all_groups[
                        heliostat_group_index
                    ],
                    device=device,
                )

                # Align heliostats.
                heliostat_alignment_group.align_surfaces_with_motor_positions(
                    motor_positions=heliostat_alignment_group.kinematic.active_motor_positions,
                    active_heliostats_mask=active_heliostats_masks_all_groups[
                        heliostat_group_index
                    ],
                    device=device,
                )

            for heliostat_group_index in self.ddp_setup[
                config_dictionary.groups_to_ranks_mapping
            ][rank]:
                heliostat_group: HeliostatGroup = (
                    self.scenario.heliostat_field.heliostat_groups[
                        heliostat_group_index
                    ]
                )

                # Calculate ray magnitude.
                power_single_heliostat = (
                    self.dni * heliostat_surface_areas[heliostat_group_index]
                )
                rays_per_heliostat = (
                    heliostat_group.surface_points.shape[1]
                    * self.scenario.light_sources.light_source_list[0].number_of_rays
                )
                ray_magnitude = power_single_heliostat / rays_per_heliostat

                # Create a ray tracer.
                ray_tracer = HeliostatRayTracer(
                    scenario=self.scenario,
                    heliostat_group=heliostat_group,
                    blocking_active=True,
                    world_size=self.ddp_setup[
                        config_dictionary.heliostat_group_world_size
                    ],
                    rank=self.ddp_setup[config_dictionary.heliostat_group_rank],
                    batch_size=self.optimization_configuration[
                        config_dictionary.batch_size
                    ],
                    random_seed=self.ddp_setup[config_dictionary.heliostat_group_rank],
                    bitmap_resolution=self.bitmap_resolution,
                    ray_magnitude=ray_magnitude,
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

            # Flux loss.
            flux_loss = loss_definition(
                prediction=total_flux.unsqueeze(index_mapping.heliostat_dimension),
                ground_truth=self.ground_truth.unsqueeze(
                    index_mapping.heliostat_dimension
                ),
                target_area_mask=torch.tensor([self.target_area_index], device=device),
                reduction_dimensions=(
                    index_mapping.batched_bitmap_e,
                    index_mapping.batched_bitmap_u,
                ),
                device=device,
            )

            if isinstance(loss_definition, FocalSpotLoss):
                loss = flux_loss

            if isinstance(loss_definition, KLDivergenceLoss):
                # Augmented Lagrangian energy integral.
                energy_integral_prediction = total_flux.sum()
                energy_integral_target = self.ground_truth.sum()
                g_energy = torch.relu(
                    (energy_integral_target - energy_integral_prediction)
                    / (energy_integral_target + self.epsilon)
                )
                # Regularizer, maximum allowable flux density.
                pixel_violation = (total_flux - max_flux_density) / (
                    max_flux_density + self.epsilon
                )
                pixel_violation = torch.clamp(pixel_violation, min=0.0)
                pixel_constraint_loss = rho_pixel * (pixel_violation**2).mean()

                if lambda_energy is None:
                    lambda_energy = torch.clamp(
                        flux_loss.detach() / (g_energy + 1e-12), min=1.0
                    )
                loss = (
                    flux_loss
                    + lambda_energy * g_energy
                    + 0.5 * rho_energy * (g_energy**2)
                    + pixel_constraint_loss
                )
                with torch.no_grad():
                    lambda_energy = torch.clamp(
                        lambda_energy + lambda_lr * g_energy.detach(), min=0.0
                    )

            loss.backward()

            # Reduce gradients across all ranks (global process group)
            if self.ddp_setup[config_dictionary.is_distributed]:
                for param_group in optimizer.param_groups:
                    for param in param_group["params"]:
                        if param.grad is not None:
                            torch.distributed.all_reduce(
                                param.grad, op=torch.distributed.ReduceOp.SUM
                            )
                            # Average the gradients
                            param.grad /= self.ddp_setup[config_dictionary.world_size]

            optimizer.step()
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss.detach())
            else:
                scheduler.step()

            if epoch % log_step == 0 and rank == 0:
                log.info(
                    f"Epoch: {epoch}, Loss: {loss.item()}, LR: {optimizer.param_groups[index_mapping.optimizer_param_group_0]['lr']}",
                )

            # Early stopping when loss did not improve since a predefined number of epochs.
            stop = early_stopper.step(loss)

            if stop:
                log.info(f"Early stopping at epoch {epoch}.")
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
                    heliostat_group.kinematic.motor_positions,
                    src=source[index_mapping.first_rank_from_group],
                )

            log.info(f"Rank: {rank}, synchronized after motor positions optimization.")

        return loss
