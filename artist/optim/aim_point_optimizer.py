import logging
from typing import Any

from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import torch
from torch.optim.lr_scheduler import LRScheduler

from artist.field.heliostat_group import HeliostatGroup
from artist.flux import bitmap
from artist.optim import training
from artist.optim.loss import KLDivergenceLoss, Loss
from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer
from artist.scenario.scenario import Scenario
from artist.util import constants, indices
from artist.util.env import DdpSetup, get_device

log = logging.getLogger(__name__)
"""A logger for the aim point optimizer."""


class AimPointOptimizer:
    """
    An optimizer used to find optimal aim points via individual motor positions for the heliostats.

    The optimization loss is defined as the loss between the combined predicted and target
    flux densities. Additionally, there is one constraint that maximizes the flux integral,
    one that maximizes the intercept factor, and one that constrains the local maximum intensity
    (maximum allowed flux density).

    Attributes
    ----------
    ddp_setup : DdpSetup
        Information about the distributed environment, process groups, devices, ranks, world size,
        and heliostat-group-to-ranks mapping.
    scenario : Scenario
        The scenario.
    optimizer_dict : dict[str, Any]
        Parameters for the optimization.
    scheduler_dict : dict[str, Any]
        Parameters for the scheduler.
    constraint_dict : dict[str, Any]
        Parameters for the constraints.
    incident_ray_direction : torch.Tensor
        Incident ray direction during the optimization.
        Shape is ``[4]``.
    target_area_index : int
        Index of the target used for the optimization.
    ground_truth : torch.Tensor
        Desired focal spot or distribution.
        Shape is ``[4]`` or ``[bitmap_resolution_e, bitmap_resolution_u]``.
    dni : float
        Direct normal irradiance in W/m^2.
    bitmap_resolution : torch.Tensor
        Resolution of all bitmaps during reconstruction.
        Shape is ``[2]``.
    epsilon : float
        A small value to avoid division by zero.

    Methods
    -------
    optimize()
        Optimize the motor positions.
    """

    def __init__(
        self,
        ddp_setup: DdpSetup,
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
        Initialize the aim points optimizer.

        Parameters
        ----------
        ddp_setup : DdpSetup
            Information about the distributed environment, process groups, devices, ranks, world size, and
            heliostat-group-to-ranks mapping.
        scenario : Scenario
            The scenario.
        optimization_configuration : dict[str, Any]
            Parameters for the optimizer, learning rate scheduler, regularizers, and early stopping.
        incident_ray_direction : torch.Tensor
            Incident ray direction during the optimization.
            Shape is ``[4]``.
        target_area_index : int
            Index of the target used for the optimization.
        ground_truth : torch.Tensor
            Desired focal spot or distribution.
            Shape is ``[4]`` or ``[bitmap_resolution_e, bitmap_resolution_u]``.
        dni : float
            Direct normal irradiance in W/m^2.
        bitmap_resolution : torch.Tensor
            Resolution of all bitmaps during optimization (default is ``torch.tensor([256,256])``).
            Shape is ``[2]``.
        epsilon : float
            A small value to avoid division by zero (default is 1e-12).
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        device = get_device(device=device)

        rank = ddp_setup["rank"]

        if rank == 0:
            log.info("Create an aim points optimizer.")

        self.ddp_setup = ddp_setup
        self.scenario = scenario
        self.optimizer_dict = optimization_configuration[constants.optimization]
        self.scheduler_dict = optimization_configuration[constants.scheduler]
        self.constraint_dict = optimization_configuration[constants.constraints]
        self.incident_ray_direction = incident_ray_direction
        self.target_area_index = target_area_index
        self.ground_truth = ground_truth
        self.dni = dni
        self.bitmap_resolution = bitmap_resolution.to(device)
        self.epsilon = epsilon

        self.pixel_area = (4.335 / self.bitmap_resolution[1]) * (5.2292 / self.bitmap_resolution[1])

    def pixel_to_meter(self, center, width, height):
        x = (center[..., 0]) / width * 4.335 - 4.335 / 2
        y = (height - center[..., 1]) / height * 5.2292 - 5.2292 / 2
        return torch.stack((x, y), dim=-1)

    def _initialize_group_parameters(
        self, device: torch.device
    ) -> tuple[
        list[torch.nn.Parameter],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        torch.Tensor,
    ]:
        """
        Pre-align all heliostat groups and set up their reparameterized motor positions.

        Each group is aligned once to the given incident ray direction and target to obtain the
        initial motor positions. The optimizable parameter is a zero-initialized reparameterization
        of the motor positions (see :meth:`optimize`).

        Parameters
        ----------
        device : torch.device
            The device on which to perform computations or load tensors and models.

        Returns
        -------
        list[torch.nn.Parameter]
            The optimizable reparameterized motor position parameters per group.
        list[torch.Tensor]
            The reparameterization scales per group.
        list[torch.Tensor]
            The initial motor positions per group.
        list[torch.Tensor]
            The active heliostats masks per group.
        list[torch.Tensor]
            The target area indices per group.
        list[torch.Tensor]
            The incident ray directions per group.
        torch.Tensor
            Offsets mapping group-local heliostat indices to global-flat index positions.
        """
        optimizable_parameters_all_groups: list[torch.nn.Parameter] = []
        scales_all_groups: list[torch.Tensor] = []
        initial_motor_positions_all_groups: list[torch.Tensor] = []

        active_heliostats_masks_all_groups: list[torch.Tensor] = []
        target_area_indices_all_groups: list[torch.Tensor] = []
        incident_ray_directions_all_groups: list[torch.Tensor] = []

        # Map group-local heliostat indices to global-flat index positions.
        group_offsets = torch.cat(
            [
                torch.tensor([0], device=device),
                self.scenario.heliostat_field.number_of_heliostats_per_group.cumsum(0)[
                    :-1
                ],
            ]
        )

        # Per-group pre-alignment.
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
            group.activate_heliostats(
                active_heliostats_mask=active_heliostats_masks_all_groups[group_index],
                device=device,
            )
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
                    :, indices.actuator_min_motor_position
                ]
            )
            motor_positions_maximum = (
                group.kinematics.actuators.non_optimizable_parameters[
                    :, indices.actuator_max_motor_position
                ]
            )
            lower_margin = initial_motor_positions - motor_positions_minimum
            upper_margin = motor_positions_maximum - initial_motor_positions

            scales_all_groups.append(
                torch.minimum(
                    torch.minimum(lower_margin, upper_margin),
                    torch.tensor(500.0, device=device),
                ).clamp(min=1.0)
            )

            optimizable_parameters_all_groups.append(
                torch.nn.Parameter(
                    torch.zeros_like(initial_motor_positions, device=device)
                )
            )

        return (
            optimizable_parameters_all_groups,
            scales_all_groups,
            initial_motor_positions_all_groups,
            active_heliostats_masks_all_groups,
            target_area_indices_all_groups,
            incident_ray_directions_all_groups,
            group_offsets,
        )

    def _setup_optimizer_scheduler_early_stopping(
        self, optimizable_parameters_all_groups: list[torch.nn.Parameter]
    ) -> tuple[torch.optim.Optimizer, LRScheduler, training.EarlyStopping]:
        """
        Create the optimizer, learning rate scheduler, and early stopping.

        Parameters
        ----------
        optimizable_parameters_all_groups : list[torch.nn.Parameter]
            The optimizable reparameterized motor position parameters per group.

        Returns
        -------
        torch.optim.Optimizer
            The Adam optimizer over all group parameter tensors.
        LRScheduler
            The learning rate scheduler.
        training.EarlyStopping
            The early stopping monitor.
        """
        # Create one Adam optimizer over all group parameter tensors.
        optimizer = torch.optim.Adam(
            optimizable_parameters_all_groups,
            lr=float(self.optimizer_dict[constants.initial_learning_rate]),
        )

        # Create a learning rate scheduler.
        scheduler_fn = getattr(
            training,
            self.scheduler_dict[constants.scheduler_type],
        )
        scheduler: LRScheduler = scheduler_fn(
            optimizer=optimizer, parameters=self.scheduler_dict
        )

        # Set up early stopping.
        early_stopper = training.EarlyStopping(
            window_size=self.optimizer_dict[constants.early_stopping_window],
            patience=self.optimizer_dict[constants.early_stopping_patience],
            min_improvement=self.optimizer_dict[constants.early_stopping_delta],
            relative=True,
        )

        return optimizer, scheduler, early_stopper

    def _get_target_plane_dimensions(self, device: torch.device) -> torch.Tensor:
        """
        Determine the target plane dimensions for the optimization target.

        Handles both planar and cylindrical target areas.

        Parameters
        ----------
        device : torch.device
            The device on which to perform computations or load tensors and models.

        Returns
        -------
        torch.Tensor
            The width and height of the target plane.
            Shape is ``[2]``.
        """
        target_plane_dimensions = torch.empty(2, device=device)
        target_areas, index = self.scenario.solar_tower.index_to_target_area[
            self.target_area_index
        ]
        if (
            self.target_area_index
            < self.scenario.solar_tower.number_of_target_areas_per_type[
                indices.planar_target_areas
            ]
        ):
            target_plane_dimensions = target_areas.dimensions[self.target_area_index]  # type: ignore[attr-defined]
        else:
            cylinder_indices = (
                self.target_area_index
                - self.scenario.solar_tower.number_of_target_areas_per_type[
                    indices.planar_target_areas
                ]
            )
            target_plane_dimensions[indices.target_area_width] = (  # type: ignore[attr-defined]
                target_areas.radii[cylinder_indices]  # type: ignore[attr-defined]
                * target_areas.opening_angles[cylinder_indices]  # type: ignore[attr-defined]
            )
            target_plane_dimensions[indices.target_area_height] = target_areas.heights[  # type: ignore[attr-defined]
                cylinder_indices
            ]

        return target_plane_dimensions

    def _align_all_groups(
        self,
        optimizer: torch.optim.Optimizer,
        initial_motor_positions_all_groups: list[torch.Tensor],
        scales_all_groups: list[torch.Tensor],
        active_heliostats_masks_all_groups: list[torch.Tensor],
        device: torch.device,
    ) -> None:
        """
        Align all heliostat groups on this rank from the reparameterized motor positions.

        The true motor positions are reconstructed from the ``tanh``-reparameterized parameters.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer holding the reparameterized motor position parameters.
        initial_motor_positions_all_groups : list[torch.Tensor]
            The initial motor positions per group.
        scales_all_groups : list[torch.Tensor]
            The reparameterization scales per group.
        active_heliostats_masks_all_groups : list[torch.Tensor]
            The active heliostats masks per group.
        device : torch.device
            The device on which to perform computations or load tensors and models.
        """
        rank = self.ddp_setup["rank"]

        for heliostat_group_index in self.ddp_setup["groups_to_ranks_mapping"][rank]:
            heliostat_alignment_group: HeliostatGroup = (
                self.scenario.heliostat_field.heliostat_groups[heliostat_group_index]
            )

            # Reconstruct true motor positions from reparameterized version.
            motor_positions_normalized = torch.tanh(
                optimizer.param_groups[indices.optimizer_param_group_0]["params"][
                    heliostat_group_index
                ]
            )
            heliostat_alignment_group.kinematics.motor_positions = (
                initial_motor_positions_all_groups[heliostat_group_index]
                + motor_positions_normalized * scales_all_groups[heliostat_group_index]
            )

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

    def _trace_and_accumulate_flux(
        self,
        active_heliostats_masks_all_groups: list[torch.Tensor],
        target_area_indices_all_groups: list[torch.Tensor],
        incident_ray_directions_all_groups: list[torch.Tensor],
        group_offsets: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Ray trace all heliostat groups on this rank and accumulate the flux on the target.

        Parameters
        ----------
        active_heliostats_masks_all_groups : list[torch.Tensor]
            The active heliostats masks per group.
        target_area_indices_all_groups : list[torch.Tensor]
            The target area indices per group.
        incident_ray_directions_all_groups : list[torch.Tensor]
            The incident ray directions per group.
        group_offsets : torch.Tensor
            Offsets mapping group-local heliostat indices to global-flat index positions.
        device : torch.device
            The device on which to perform computations or load tensors and models.

        Returns
        -------
        torch.Tensor
            The accumulated flux distribution on the target.
        torch.Tensor
            The intercept factors per heliostat.
        torch.Tensor
            The on-target factors per heliostat.
        torch.Tensor
            The blocking factors per heliostat.
        """
        rank = self.ddp_setup["rank"]

        total_flux = torch.zeros(
            (
                int(self.bitmap_resolution[indices.unbatched_bitmap_u]),
                int(self.bitmap_resolution[indices.unbatched_bitmap_e]),
            ),
            device=device,
        )

        # Trace rays and accumulate fluxes.
        for heliostat_group_index in self.ddp_setup["groups_to_ranks_mapping"][rank]:
            heliostat_group: HeliostatGroup = (
                self.scenario.heliostat_field.heliostat_groups[heliostat_group_index]
            )

            # Create a ray tracer.
            ray_tracer = HeliostatRayTracer(
                scenario=self.scenario,
                heliostat_group=heliostat_group,
                blocking_active=True,
                world_size=self.ddp_setup["heliostat_group_world_size"],
                rank=self.ddp_setup["heliostat_group_rank"],
                batch_size=self.optimizer_dict["batch_size"],
                random_seed=self.ddp_setup["heliostat_group_rank"],
                bitmap_resolution=self.bitmap_resolution,
                dni=self.dni,
            )

            # Perform heliostat-based ray tracing.
            (
                flux_distributions,
                intercept_factor, 
                on_target_factor, 
                blocking_factor
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

            print(intercept_factor.mean())
            print(on_target_factor.mean())
            print(blocking_factor.mean())

            sample_indices_for_local_rank = ray_tracer.get_sampler_indices()
            flux_distribution_on_target = ray_tracer.get_bitmaps_per_target(
                bitmaps_per_heliostat=flux_distributions,
                target_area_indices=target_area_indices_all_groups[
                    heliostat_group_index
                ][sample_indices_for_local_rank],
                device=device,
            )[self.target_area_index]
            total_flux = total_flux + flux_distribution_on_target

            flux_centers = bitmap.get_center_of_mass(
                bitmaps=flux_distributions,
                device=device
            )

        if self.ddp_setup["is_distributed"]:
            total_flux = torch.distributed.nn.functional.all_reduce(
                total_flux,
                op=torch.distributed.ReduceOp.SUM,
            )

        return total_flux, flux_centers

    def _synchronize_distributed_gradients(
        self, optimizer: torch.optim.Optimizer
    ) -> None:
        """
        Reduce and average gradients across all ranks in the global process group.

        This is a no-op when not running in distributed mode.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer whose parameter gradients are synchronized.
        """
        if self.ddp_setup[constants.is_distributed]:  # type: ignore
            for param_group in optimizer.param_groups:
                for param in param_group["params"]:
                    if param.grad is not None:
                        torch.distributed.all_reduce(
                            param.grad, op=torch.distributed.ReduceOp.SUM
                        )
                        # Average the gradients.
                        param.grad /= self.ddp_setup[constants.world_size]  # type: ignore

    def _broadcast_motor_positions(self) -> None:
        """
        Broadcast the final motor positions of each heliostat group from its source rank.

        This is a no-op when not running in distributed mode.
        """
        rank = self.ddp_setup["rank"]

        if self.ddp_setup["is_distributed"]:
            for index, heliostat_group in enumerate(
                self.scenario.heliostat_field.heliostat_groups
            ):
                source = self.ddp_setup["ranks_to_groups_mapping"][index]
                torch.distributed.broadcast(
                    heliostat_group.kinematics.motor_positions,
                    src=source[indices.first_rank_from_group],
                )

            log.info(f"Rank: {rank}, synchronized after aim point optimization.")

    def optimize(
        self,
        loss_definition: Loss,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, dict[str, list], torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Optimize the motor positions for optimal aim points.

        The motor positions are optimized through a reparameterization to ensure stable training
        across different heliostats with widely varying initial motor positions and ranges. Motor
        positions can range from 0 to up to ~80000. Instead of directly optimizing the absolute
        motor positions, which can differ in magnitudes, an unconstrained parameter is optimized.
        Directly optimizing the absolute motor positions would have very different effects depending
        on the scale of the motors. For small initial motor positions (e.g. ~100), a gradient update
        of size 10 may cause a ~10% relative change, drastically altering the motor positions of this
        heliostat. For large initial motor positions (e.g. ~50000), the same optimizer step would
        correspond to only a 0.02% relative change in motor positions, effectively freezing the
        optimization of this heliostat. This mismatch makes it impossible to choose a single learning
        rate that works robustly across all heliostats.
        Reparameterizing the motor positions to be optimized defines the optimizable parameter as:

        .. math::

            \text{motor\_positions\_optimized} = \tanh(
                \text{torch.nn.Parameter(optimizable\_parameter)}
            )

        The true motor positions can be reconstructed by:

        .. math::

            \text{motor\_positions} = \text{initial\_motor\_positions} +
            \text{motor\_positions\_normalized} \cdot \text{scale}

        where scale defines the range (e.g. up to ~80000) for adjustments.
        By optimizing reparameterized instead of raw motor positions, every heliostat sees updates
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
            Final loss of the aim point optimization.
        dict[str, list]
            Loss history over epochs, with keys ``"total_loss"``, ``"flux_loss"``,
            ``"local_flux_constraint"``, ``"intercept_constraint"``, ``"flux_integral_constraint"``,
            and ``"flux_integral"``. Each value is a list of per-epoch scalar floats.
        torch.Tensor
            Final intercept factors for each heliostat.
        torch.Tensor
            Final fraction of rays hitting the target, neglecting blocking effects, for each heliostat.
        torch.Tensor
            Final fraction of rays not being blocked, for each heliostat.
        """
        device = get_device(device)
        rank = self.ddp_setup["rank"]

        if rank == 0:
            log.info("Start the aim point optimization.")

        (
            optimizable_parameters_all_groups,
            scales_all_groups,
            initial_motor_positions_all_groups,
            active_heliostats_masks_all_groups,
            target_area_indices_all_groups,
            incident_ray_directions_all_groups,
            group_offsets,
        ) = self._initialize_group_parameters(device=device)

        optimizer, scheduler, early_stopper = (
            self._setup_optimizer_scheduler_early_stopping(
                optimizable_parameters_all_groups=optimizable_parameters_all_groups
            )
        )

        flux_reference = None
        # Start the optimization.
        loss = torch.tensor(torch.inf)
        epoch = 0
        log_step = (
            self.optimizer_dict[constants.max_epoch]
            if self.optimizer_dict[constants.log_step] == 0
            else self.optimizer_dict[constants.log_step]
        )
        while (
            loss > float(self.optimizer_dict[constants.tolerance])
            and epoch <= self.optimizer_dict[constants.max_epoch]
        ):
            optimizer.zero_grad()

            # Align all heliostats from all groups.
            self._align_all_groups(
                optimizer=optimizer,
                initial_motor_positions_all_groups=initial_motor_positions_all_groups,
                scales_all_groups=scales_all_groups,
                active_heliostats_masks_all_groups=active_heliostats_masks_all_groups,
                device=device,
            )

            # Trace rays and accumulate fluxes.
            total_flux, flux_centers = (
                self._trace_and_accumulate_flux(
                    active_heliostats_masks_all_groups=active_heliostats_masks_all_groups,
                    target_area_indices_all_groups=target_area_indices_all_groups,
                    incident_ray_directions_all_groups=incident_ray_directions_all_groups,
                    group_offsets=group_offsets,
                    device=device,
                )
            )

            if flux_reference  is None:
                flux_reference = total_flux.sum().detach()
            predicted_integrals = total_flux.sum()
            relative_decrease = 1.0 - (
                predicted_integrals / (flux_reference + self.epsilon)
            )
            relative_decrease = torch.clamp(
                relative_decrease,
                min=0.0,
                max=0.4,
            )
            integral_losses = 2.0 * (
                relative_decrease / 0.4
            ) ** 2

            # Flux loss: Compare predicted total flux vs. ground truth.
            flux_loss = loss_definition(
                prediction=total_flux.unsqueeze(indices.heliostat_dimension),
                ground_truth=self.ground_truth.unsqueeze(indices.heliostat_dimension),
                reduction_dimensions=(
                    indices.batched_bitmap_e,
                    indices.batched_bitmap_u,
                ),
                device=device,
            )

            loss = flux_loss + integral_losses

            loss.backward()


            #0 <= l1_losses <= 2

            self._synchronize_distributed_gradients(optimizer=optimizer)

            optimizer.step()
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss.detach())
            else:
                scheduler.step()

            if epoch % log_step == 0 and rank == 0:
                print(
                    f"Epoch: {epoch}, Loss: {loss.item():.5e}, LR: {optimizer.param_groups[indices.optimizer_param_group_0]['lr']}",
                    f"flux loss={flux_loss.item():.3f} | integral loss={integral_losses.item():.3f} | "
                    f"intensity={(total_flux.sum().item()):.4f}"
                )

            # if epoch % 1 == 0 and rank == 0:
                
            #     flux = total_flux / self.pixel_area

            #     fig, ax = plt.subplots()
            #     im = ax.imshow(
            #         flux.cpu().detach(),
            #         extent=[
            #             -4.335 / 2, 4.335 / 2,
            #             -5.2292 / 2, 5.2292 / 2
            #         ],
            #     )
            #     flux_centers = self.pixel_to_meter(
            #         center=flux_centers, 
            #         width=self.bitmap_resolution[1],
            #         height=self.bitmap_resolution[0]
            #     )
            #     flux_centers += torch.tensor([0.5, 0.5], device=device)
            #     ax.scatter(
            #         flux_centers[:, 0].cpu().detach(),
            #         flux_centers[:, 1].cpu().detach(),
            #         alpha=0.3,
            #         s=2,
            #         color="magenta",
            #         #edgecolors="white",
            #     )

            #     ax.set_title(
            #         #f"Aim point optimization\n{((flux * self.pixel_area).sum().item()/1000000):.1f} MW",
            #         f"Aim point optimization\n{(total_flux.sum().item() / 1000000):.3f} MW"
            #     )
            #     ax.set_ylabel("Target height relative to center (m)", labelpad=2.5)
            #     ax.set_xlabel("Target width relative to center (m)", labelpad=4)


            #     cbar = fig.colorbar(im, ax=ax)
            #     cbar.set_label(
            #         r"Flux density ($10^6\,\mathrm{W}/\mathrm{m}^2$)",
            #     )
            #     cbar.ax.yaxis.set_major_formatter(
            #         FuncFormatter(lambda x, pos: f"{x/1e6:g}")
            #     )

            #     fig.tight_layout()
            #     fig.savefig(f"opt_{epoch}.png", dpi=300, bbox_inches="tight")
            #     plt.close(fig)

            # Early stopping when loss did not improve for a predefined number of epochs.
            stop = early_stopper.step(loss.item())

            if stop:
                log.info(f"Early stopping at epoch {epoch}.")
                break

            epoch += 1

        log.info(f"Rank: {rank}, aim points optimized.")

        # Broadcast final motor positions for each heliostat group from source rank to others.
        self._broadcast_motor_positions()

        return loss.detach().cpu()
