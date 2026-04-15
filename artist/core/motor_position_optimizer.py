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

    The optimization loss is defined as the loss between the combined predicted and target
    flux densities. Additionally, there is one constraint that maximizes the flux integral and
    one that constrains the maximum pixel intensity (maximum allowed flux density).

    Attributes
    ----------
    ddp_setup : dict[str, Any]
        Information about the distributed environment, process_groups, devices, ranks, world_size, heliostat group_to_ranks mapping.
    scenario : Scenario
        The scenario.
    optimizer_dict : dict[str, Any]
        The parameters for the optimization.
    scheduler_dict : dict[str, Any]
        The parameters for the scheduler.
    constraint_dict : dict[str, Any]
        The parameters for the constraints.
    incident_ray_direction : torch.Tensor
        The incident ray direction during the optimization.
        Tensor of shape [4].
    target_area_index : int
        The index of the target used for the optimization.
    ground_truth : torch.Tensor
        The desired focal spot or distribution.
        Tensor of shape [4] or tensor of shape [bitmap_resolution_e, bitmap_resolution_u].
    dni : float
        Direct normal irradiance in W/m^2.
    bitmap_resolution : torch.Tensor
        The resolution of all bitmaps during reconstruction.
        Tensor of shape [2].
    epsilon : float
        A small value.

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
        dni: float,
        bitmap_resolution: torch.Tensor = torch.tensor([256, 256]),
        epsilon: float = 1e-12,
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
        dni : float
            Direct normal irradiance in W/m^2.
        bitmap_resolution : torch.Tensor
            The resolution of all bitmaps during optimization (default is torch.tensor([256,256])).
            Tensor of shape [2].
        epsilon : float | None
            A small value (default is 1e-12).
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
        self.optimizer_dict = optimization_configuration[config_dictionary.optimization]
        self.scheduler_dict = optimization_configuration[config_dictionary.scheduler]
        self.constraint_dict = optimization_configuration[config_dictionary.constraints]
        self.incident_ray_direction = incident_ray_direction
        self.target_area_index = target_area_index
        self.ground_truth = ground_truth
        self.dni = dni
        self.bitmap_resolution = bitmap_resolution.to(device)
        self.epsilon = epsilon

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
        target_area_indices_all_groups = []
        incident_ray_directions_all_groups = []

        group_offsets = torch.cat(
            [
                torch.tensor([0], device=device),
                self.scenario.heliostat_field.number_of_heliostats_per_group.cumsum(0)[
                    :-1
                ],
            ]
        )

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
            target_area_indices_all_groups.append(
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
                aim_points=self.scenario.solar_tower.get_centers_of_target_areas(
                    target_area_indices=target_area_indices_all_groups[group_index],
                    device=device,
                ),
                incident_ray_directions=incident_ray_directions_all_groups[group_index],
                active_heliostats_mask=active_heliostats_masks_all_groups[group_index],
                device=device,
            )

            # Reparametrization of the motor positions (optimizable parameter).
            initial_motor_positions = (
                group.kinematics.active_motor_positions.detach().clone()
            )
            initial_motor_positions_all_groups.append(initial_motor_positions)
            motor_positions_minimum = (
                group.kinematics.actuators.non_optimizable_parameters[
                    :, index_mapping.actuator_min_motor_position
                ]
            )
            motor_positions_maximum = (
                group.kinematics.actuators.non_optimizable_parameters[
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

        optimizer = torch.optim.Adam(
            optimizable_parameters_all_groups,
            lr=float(self.optimizer_dict[config_dictionary.initial_learning_rate]),
        )

        # Create a learning rate scheduler.
        scheduler_fn = getattr(
            learning_rate_schedulers,
            self.scheduler_dict[config_dictionary.scheduler_type],
        )
        scheduler: LRScheduler = scheduler_fn(
            optimizer=optimizer, parameters=self.scheduler_dict
        )

        # Set up early stopping.
        early_stopper = learning_rate_schedulers.EarlyStopping(
            window_size=self.optimizer_dict[config_dictionary.early_stopping_window],
            patience=self.optimizer_dict[config_dictionary.early_stopping_patience],
            min_improvement=self.optimizer_dict[config_dictionary.early_stopping_delta],
            relative=True,
        )

        target_plane_dimensions = torch.empty(2, device=device)
        target_areas, index = self.scenario.solar_tower.index_to_target_area[
            self.target_area_index
        ]
        if (
            self.target_area_index
            < self.scenario.solar_tower.number_of_target_areas_per_type[0]
        ):
            target_plane_dimensions = target_areas.dimensions[self.target_area_index]  # type: ignore[attr-defined]
        else:
            cylinder_indices = (
                self.target_area_index
                - self.scenario.solar_tower.number_of_target_areas_per_type[0]
            )
            target_plane_dimensions[0] = (  # type: ignore[attr-defined]
                target_areas.radii[cylinder_indices]  # type: ignore[attr-defined]
                * target_areas.opening_angles[cylinder_indices]  # type: ignore[attr-defined]
            )
            target_plane_dimensions[1] = target_areas.heights[cylinder_indices]  # type: ignore[attr-defined]

        # Set up constraints.
        flux_integral_reference = 0.0
        intercept_factors_reference = 0.0
        lambda_local_flux = 0.0
        lambda_flux_integral = 0.0
        lambda_intercept = 0.0
        rho_local_flux = self.constraint_dict[config_dictionary.rho_local_flux]
        rho_flux_integral = self.constraint_dict[config_dictionary.rho_flux_integral]
        rho_intercept = self.constraint_dict[config_dictionary.rho_intercept]
        max_flux_density_per_pixel = (
            torch.prod(target_plane_dimensions) / torch.prod(self.bitmap_resolution)
        ) * self.constraint_dict[config_dictionary.max_flux_density]

        # For the loss plot.
        total_loss_history = []
        flux_loss_history = []
        flux_integral = []
        local_flux_constraint_history = []
        intercept_constraint_history = []
        flux_integral_constraint_history = []

        # Start the optimization.
        loss = torch.inf
        epoch = 0
        log_step = (
            self.optimizer_dict[config_dictionary.max_epoch]
            if self.optimizer_dict[config_dictionary.log_step] == 0
            else self.optimizer_dict[config_dictionary.log_step]
        )
        while (
            loss > float(self.optimizer_dict[config_dictionary.tolerance])
            and epoch <= self.optimizer_dict[config_dictionary.max_epoch]
        ):
            optimizer.zero_grad()

            total_flux = torch.zeros(
                (
                    self.bitmap_resolution[index_mapping.unbatched_bitmap_e],
                    self.bitmap_resolution[index_mapping.unbatched_bitmap_u],
                ),
                device=device,
            )
            intercept_factors = torch.zeros(
                (sum(actives.sum() for actives in active_heliostats_masks_all_groups)),
                device=device,
            )
            on_target_factors = torch.zeros_like(intercept_factors, device=device)
            blocking_factors = torch.zeros_like(intercept_factors, device=device)

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
                heliostat_alignment_group.kinematics.motor_positions = (
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
                    motor_positions=heliostat_alignment_group.kinematics.active_motor_positions,
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

                # Create a ray tracer.
                ray_tracer = HeliostatRayTracer(
                    scenario=self.scenario,
                    heliostat_group=heliostat_group,
                    blocking_active=True,
                    world_size=self.ddp_setup[
                        config_dictionary.heliostat_group_world_size
                    ],
                    rank=self.ddp_setup[config_dictionary.heliostat_group_rank],
                    batch_size=self.optimizer_dict[config_dictionary.batch_size],
                    random_seed=self.ddp_setup[config_dictionary.heliostat_group_rank],
                    bitmap_resolution=self.bitmap_resolution,
                    dni=self.dni,
                )

                # Perform heliostat-based ray tracing.
                (
                    flux_distributions,
                    intercept_factor,
                    on_target_factor,
                    blocking_factor,
                ) = ray_tracer.trace_rays(
                    incident_ray_directions=incident_ray_directions_all_groups[
                        heliostat_group_index
                    ],
                    active_heliostats_mask=active_heliostats_masks_all_groups[
                        heliostat_group_index
                    ],
                    target_area_indices=target_area_indices_all_groups[
                        heliostat_group_index
                    ],
                    device=device,
                )
                sample_indices_for_local_rank = ray_tracer.get_sampler_indices()
                flux_distribution_on_target = ray_tracer.get_bitmaps_per_target(
                    bitmaps_per_heliostat=flux_distributions,
                    target_area_indices=target_area_indices_all_groups[
                        heliostat_group_index
                    ][sample_indices_for_local_rank],
                    device=device,
                )[self.target_area_index]

                total_flux = total_flux + flux_distribution_on_target

                global_indices = (
                    group_offsets[heliostat_group_index] + sample_indices_for_local_rank
                )
                intercept_factors[global_indices] = intercept_factor
                on_target_factors[global_indices] = on_target_factor
                blocking_factors[global_indices] = blocking_factor

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
                target_area_indices=torch.tensor(
                    [self.target_area_index], device=device
                ),
                reduction_dimensions=(
                    index_mapping.batched_bitmap_e,
                    index_mapping.batched_bitmap_u,
                ),
                device=device,
            )

            if isinstance(loss_definition, FocalSpotLoss):
                loss = flux_loss

            if isinstance(loss_definition, KLDivergenceLoss):
                # Augmented Lagrangian to ensure that flux integral is maximized, i.e., intensity increases or stays the same.
                if epoch == 0:
                    flux_integral_reference = total_flux.sum().detach()
                    intercept_factors_reference = intercept_factors.detach()
                flux_integral_difference = (
                    flux_integral_reference - total_flux.sum()
                ) / (flux_integral_reference + self.epsilon)
                flux_integral_difference_clamped = torch.clamp(
                    flux_integral_difference, min=0.0
                )
                flux_integral_constraint = (
                    lambda_flux_integral * flux_integral_difference_clamped
                    + 0.5 * rho_flux_integral * flux_integral_difference_clamped**2
                )

                # Augmented Lagrangian to ensure that spillage is reduced.
                intercept_factors_differences = (
                    intercept_factors_reference - intercept_factors
                ) / (intercept_factors_reference + self.epsilon)
                intercept_factors_differences_clamped = torch.clamp(
                    intercept_factors_differences, min=0.0
                )
                intercept_factor_constraint = (
                    lambda_intercept * intercept_factors_differences_clamped
                    + 0.5 * rho_intercept * intercept_factors_differences_clamped**2
                ).mean()

                # Augmented Lagrangian to ensure that local heat spikes are avoided.
                local_flux_violation = (
                    total_flux - max_flux_density_per_pixel.detach()
                ) / (max_flux_density_per_pixel.detach() + self.epsilon)
                local_flux_violation_clamped = torch.clamp(
                    local_flux_violation, min=0.0
                )
                local_flux_constraint = (
                    lambda_local_flux * local_flux_violation_clamped
                    + 0.5 * rho_local_flux * local_flux_violation_clamped**2
                ).max()

                loss = (
                    flux_loss
                    + flux_integral_constraint
                    + intercept_factor_constraint
                    + local_flux_constraint
                )
                with torch.no_grad():
                    lambda_local_flux = torch.clamp(
                        lambda_local_flux + rho_local_flux * local_flux_violation.max(),
                        min=0.0,
                    )
                    lambda_intercept = torch.clamp(
                        lambda_intercept
                        + rho_intercept * intercept_factors_differences.mean(),
                        min=0.0,
                    )
                    lambda_flux_integral = torch.clamp(
                        lambda_flux_integral
                        + rho_flux_integral * flux_integral_difference,
                        min=0.0,
                    )

            loss.backward()

            # Reduce gradients across all ranks (global process group).
            if self.ddp_setup[config_dictionary.is_distributed]:
                for param_group in optimizer.param_groups:
                    for param in param_group["params"]:
                        if param.grad is not None:
                            torch.distributed.all_reduce(
                                param.grad, op=torch.distributed.ReduceOp.SUM
                            )
                            # Average the gradients.
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

            total_loss_history.append(loss.detach().cpu().item())
            flux_loss_history.append(flux_loss.detach().cpu().item())
            if isinstance(loss_definition, KLDivergenceLoss):
                flux_integral.append(
                    (
                        100
                        / flux_integral_reference
                        * (total_flux.sum() - flux_integral_reference + 1e-8)
                    )
                    .detach()
                    .cpu()
                    .item()
                )
                local_flux_constraint_history.append(
                    local_flux_constraint.detach().cpu().item()
                )
                intercept_constraint_history.append(
                    intercept_factor_constraint.detach().cpu().item()
                )
                flux_integral_constraint_history.append(
                    flux_integral_constraint.detach().cpu().item()
                )

            # Early stopping when loss did not improve since a predefined number of epochs.
            stop = early_stopper.step(loss)

            if stop:
                log.info(f"Early stopping at epoch {epoch}.")
                break

            epoch += 1

        loss_history = {
            "total_loss": total_loss_history,
            "flux_loss": flux_loss_history,
            "local_flux_constraint": local_flux_constraint_history,
            "intercept_constraint": intercept_constraint_history,
            "flux_integral_constraint": flux_integral_constraint_history,
            "flux_integral": flux_integral,
        }
        log.info(f"Rank: {rank}, motor positions optimized.")

        if self.ddp_setup[config_dictionary.is_distributed]:
            for index, heliostat_group in enumerate(
                self.scenario.heliostat_field.heliostat_groups
            ):
                source = self.ddp_setup[config_dictionary.ranks_to_groups_mapping][
                    index
                ]
                torch.distributed.broadcast(
                    heliostat_group.kinematics.motor_positions,
                    src=source[index_mapping.first_rank_from_group],
                )

            log.info(f"Rank: {rank}, synchronized after motor positions optimization.")

        return (
            loss.detach().cpu(),
            loss_history,
            intercept_factors,
            on_target_factors,
            blocking_factors,
        )
