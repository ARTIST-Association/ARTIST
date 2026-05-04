import logging
import pathlib
from typing import Any, cast

import torch
from torch.optim.lr_scheduler import LRScheduler

import artist.nurbs.utils
from artist.core import core_utils
from artist.field.heliostat_group import HeliostatGroup
from artist.flux import bitmap
from artist.io.calibration_parser import CalibrationDataParser
from artist.nurbs.surfaces import NURBSSurfaces
from artist.optimization import training
from artist.optimization.loss_functions import Loss
from artist.optimization.regularizers import (
    IdealSurfaceRegularizer,
    SmoothnessRegularizer,
)
from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer
from artist.scenario.scenario import Scenario
from artist.util import (
    constants,
    indices,
)
from artist.util.environment import DdpSetup, get_device

log = logging.getLogger(__name__)
"""A logger for the surface reconstructor."""


class SurfaceReconstructor:
    """
    An optimizer used to reconstruct surfaces using NURBS and measured flux distributions.

    The surface reconstructor learns a surface representation from measured flux density
    distributions. The optimizable parameters for this optimization process are the
    NURBS control points.
    The reconstruction loss is defined by the loss between the flux density predictions and measurements.
    Further, the reconstruction is constrained by flux integral constraints to preserve energy in the reconstructed
    surfaces. There are also optional regularizers to keep the NURBS control points close to the ideal
    surface and smooth.

    Attributes
    ----------
    ddp_setup : DdpSetup
        Information about the distributed environment, process groups, devices, ranks, world size,
        and heliostat-group-to-ranks mapping.
    scenario : Scenario
        The scenario.
    data : dict[str, CalibrationDataParser | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]]
        The data parser and the mapping of heliostat name and calibration data.
    optimizer_dict : dict[str, Any]
        The parameters for the optimization.
    scheduler_dict : dict[str, Any]
        The parameters for the scheduler.
    constraint_dict : dict[str, Any]
        The parameters for the constraints.
    number_of_surface_points : torch.Tensor
        The number of surface points of the reconstructed surfaces.
        Shape is ``[2]``.
    dni : float | None
        Direct normal irradiance in W/m² used to scale the ray-traced flux. If None, the
        ``HeliostatRayTracer`` uses its own default.
    bitmap_resolution : torch.Tensor
        The resolution of all bitmaps during reconstruction.
        Shape is ``[2]``.
    epsilon : float | None
        Small numerical offset used to avoid division by zero in the energy constraint.

    Note
    ----
    Each heliostat selected for reconstruction needs to have the same number of samples as all others.

    Methods
    -------
    reconstruct_surfaces()
        Reconstruct NURBS surfaces from bitmaps.
    lock_control_points_on_outer_edges()
        Lock the u and v values of the control points on the outer edges of each facet.
    """

    def __init__(
        self,
        ddp_setup: DdpSetup,
        scenario: Scenario,
        data: dict[
            str,
            CalibrationDataParser
            | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
        ],
        optimization_configuration: dict[str, Any],
        dni: float | None = None,
        number_of_surface_points: torch.Tensor = torch.tensor([50, 50]),
        bitmap_resolution: torch.Tensor = torch.tensor([256, 256]),
        epsilon: float | None = 1e-12,
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize the surface reconstructor.

        Parameters
        ----------
        ddp_setup : DdpSetup
            Information about the distributed environment, process groups, devices, ranks, world size,
            and heliostat-group-to-ranks mapping.
        scenario : Scenario
            The scenario.
        data : dict[str, CalibrationDataParser | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]]
            The data parser and the mapping of heliostat name and calibration data.
        optimization_configuration : dict[str, Any]
            The parameters for the optimizer, learning rate scheduler, early stopping, and constraints.
        dni : float | None
            Direct normal irradiance in W/m² used to scale the ray-traced flux (default is None).
            If None, the ``HeliostatRayTracer`` uses its own default.
        number_of_surface_points : torch.Tensor
            The number of surface points of the reconstructed surfaces (default is ``torch.tensor([50, 50])``).
            Shape is ``[2]``.
        bitmap_resolution : torch.Tensor
            The resolution of all bitmaps during reconstruction (default is ``torch.tensor([256, 256])``).
            Shape is ``[2]``.
        epsilon : float | None
            Small numerical offset used to avoid division by zero in the energy constraint (default is 1e-12).
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        device = get_device(device=device)

        rank = ddp_setup["rank"]

        if rank == 0:
            log.info("Create a surface reconstructor.")

        self.ddp_setup = ddp_setup
        self.scenario = scenario
        self.data = data
        self.optimizer_dict = optimization_configuration[constants.optimization]
        self.scheduler_dict = optimization_configuration[constants.scheduler]
        self.constraint_dict = optimization_configuration[constants.constraints]
        self.number_of_surface_points = number_of_surface_points.to(device)
        self.dni = dni
        self.bitmap_resolution = bitmap_resolution.to(device)
        self.epsilon = epsilon

    def reconstruct_surfaces(
        self,
        loss_definition: Loss,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, list[list[dict[str, list[float]]]]]:
        """
        Reconstruct NURBS surfaces from bitmaps.

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
            The final reconstruction loss per heliostat, one entry per heliostat in the scenario.
            Shape is ``[total_number_of_heliostats_in_scenario]``.
        list[list[dict[str, list[float]]]]
            Loss histories over epochs grouped by rank.

            - Outer list: one entry per rank.
            - Inner list: one entry per heliostat group processed on that rank.
            - Each group entry is a dict with keys:
              ``"total_loss"``, ``"flux_loss"``, ``"smoothness_regularizer"``,
              ``"ideal_regularizer"``, ``"flux_integral"``, and
              ``"flux_integral_constraint"``.
              Each value is a list of per-epoch scalar floats.

              In non-distributed mode, this is a single-rank container: ``[local_group_histories]``.
        """
        device = get_device(device=device)
        rank = self.ddp_setup["rank"]

        if rank == 0:
            log.info("Beginning surface reconstruction.")

        # Final per-heliostat loss container (global over all groups), initialized with + inf.
        final_loss_per_heliostat = torch.full(
            (self.scenario.heliostat_field.number_of_heliostats_per_group.sum(),),
            torch.inf,
            device=device,
        )

        # Prefix sums to map group-local heliostat indices to global heliostat indices.
        final_loss_start_indices = torch.cat(
            [
                torch.tensor([0], device=device),
                self.scenario.heliostat_field.number_of_heliostats_per_group.cumsum(
                    indices.heliostat_dimension
                ),
            ]
        )

        # Rank-local history: one dict per processed heliostat group.
        loss_history: list[dict[str, list[float]]] = []

        # Process only groups assigned to this rank.
        for heliostat_group_index in self.ddp_setup["groups_to_ranks_mapping"][rank]:
            heliostat_group: HeliostatGroup = (
                self.scenario.heliostat_field.heliostat_groups[heliostat_group_index]
            )

            # Load calibration parser and input file mapping.
            parser = cast(CalibrationDataParser, self.data[constants.data_parser])
            heliostat_mapping = cast(
                list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
                self.data[constants.heliostat_data_mapping],
            )

            # Obtain measured flux + metadata for this group.
            (
                measured_flux_distributions,
                _,
                incident_ray_directions,
                _,
                active_heliostats_mask,
                target_area_indices,
            ) = parser.parse_data_for_reconstruction(
                heliostat_data_mapping=heliostat_mapping,
                heliostat_group=heliostat_group,
                scenario=self.scenario,
                bitmap_resolution=self.bitmap_resolution,
                device=device,
            )

            # Skip groups with no active heliostats.
            if active_heliostats_mask.sum() > 0:
                # Create NURBS evaluation points: Build a UV grid for NURBS sampling and expand to required shape.
                evaluation_points = (
                    artist.nurbs.utils.create_nurbs_evaluation_grid(
                        number_of_evaluation_points=self.number_of_surface_points,
                        device=device,
                    )
                    .unsqueeze(indices.heliostat_dimension)
                    .unsqueeze(indices.facet_index_unbatched)
                    .expand(
                        int(active_heliostats_mask.sum()),
                        heliostat_group.number_of_facets_per_heliostat,
                        -1,
                        -1,
                    )
                )

                # Keep a frozen copy of original control points for regularization terms.
                with torch.no_grad():
                    original_control_points = heliostat_group.nurbs_control_points[
                        active_heliostats_mask > 0
                    ].clone()

                # Create the optimizer: Optimize NURBS control points directly.
                optimizer = torch.optim.Adam(
                    [heliostat_group.nurbs_control_points.requires_grad_()],
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

                # Set up early stopping on stagnating loss.
                early_stopper = training.EarlyStopping(
                    window_size=self.optimizer_dict[constants.early_stopping_window],
                    patience=self.optimizer_dict[constants.early_stopping_patience],
                    min_improvement=self.optimizer_dict[constants.early_stopping_delta],
                    relative=True,
                )

                # Set up Augmented-Lagrangian constraint for energy conservation.
                flux_integrals_reference = torch.zeros_like(active_heliostats_mask)
                lambda_flux_integral = 0.0
                rho_flux_integral = self.constraint_dict[constants.rho_flux_integral]
                energy_tolerance = self.constraint_dict[constants.energy_tolerance]
                # Set up regularizers: Keep reconstructed surface smooth and close to ideal/original.
                ideal_surface_regularizer = IdealSurfaceRegularizer(
                    reduction_dimensions=(1,)
                )
                smoothness_regularizer = SmoothnessRegularizer(
                    reduction_dimensions=(1,)
                )
                weight_smoothness = self.constraint_dict[constants.weight_smoothness]
                weight_ideal_surface = self.constraint_dict[
                    constants.weight_ideal_surface
                ]

                # Set up per-epoch logging/history buffers.
                total_loss_history = []
                flux_loss_history = []
                flux_integral_history = []
                smoothness_history = []
                ideal_history = []
                flux_integral = []

                # Start the optimization.
                total_loss = torch.inf
                epoch = 0
                log_step = (
                    self.optimizer_dict[constants.max_epoch]
                    if self.optimizer_dict[constants.log_step] == 0
                    else self.optimizer_dict[constants.log_step]
                )
                max_epoch = int(
                    torch.tensor(
                        [self.optimizer_dict[constants.max_epoch]],
                        device=device,
                    )
                )

                # Optimization loop.
                while (
                    total_loss > float(self.optimizer_dict[constants.tolerance])
                    and epoch <= max_epoch
                ):
                    optimizer.zero_grad()

                    # Activate heliostats according to current mask.
                    heliostat_group.activate_heliostats(
                        active_heliostats_mask=active_heliostats_mask, device=device
                    )

                    # Build NURBS surface from current control points.
                    nurbs_surfaces = NURBSSurfaces(
                        degrees=heliostat_group.nurbs_degrees,
                        control_points=heliostat_group.active_nurbs_control_points,
                        device=device,
                    )

                    # Calculate surface points and normals.
                    (
                        new_surface_points,
                        new_surface_normals,
                    ) = nurbs_surfaces.calculate_surface_points_and_normals(
                        evaluation_points=evaluation_points,
                        canting=heliostat_group.active_canting,
                        facet_translations=heliostat_group.active_facet_translations,
                        device=device,
                    )

                    # Flatten faceted tensors to the shape expected by alignment module and ray tracer.
                    heliostat_group.active_surface_points = new_surface_points.reshape(
                        heliostat_group.active_surface_points.shape[
                            indices.heliostat_dimension
                        ],
                        -1,
                        4,
                    )
                    heliostat_group.active_surface_normals = (
                        new_surface_normals.reshape(
                            heliostat_group.active_surface_normals.shape[
                                indices.heliostat_dimension
                            ],
                            -1,
                            4,
                        )
                    )

                    # Align heliostat surfaces toward target under current incident ray directions.
                    heliostat_group.align_surfaces_with_incident_ray_directions(
                        aim_points=self.scenario.solar_tower.get_centers_of_target_areas(
                            target_area_indices=target_area_indices, device=device
                        ),
                        incident_ray_directions=incident_ray_directions,
                        active_heliostats_mask=active_heliostats_mask,
                        device=device,
                    )

                    # Create a parallelized ray tracer. Blocking is always deactivated for this reconstruction.
                    ray_tracer = HeliostatRayTracer(
                        scenario=self.scenario,
                        heliostat_group=heliostat_group,
                        blocking_active=False,
                        world_size=self.ddp_setup["heliostat_group_world_size"],
                        rank=self.ddp_setup["heliostat_group_rank"],
                        batch_size=self.optimizer_dict[constants.batch_size],
                        random_seed=self.ddp_setup["heliostat_group_rank"],
                        bitmap_resolution=self.bitmap_resolution,
                        dni=self.dni,
                    )

                    # Perform heliostat-based ray tracing to obtain simulated flux from current reconstructed surfaces.
                    flux_distributions, _, _, _ = ray_tracer.trace_rays(
                        incident_ray_directions=incident_ray_directions,
                        active_heliostats_mask=active_heliostats_mask,
                        target_area_indices=target_area_indices,
                        device=device,
                    )

                    # Recover local sampler ordering and samples-per-heliostat factor.
                    sample_indices_for_local_rank = ray_tracer.get_sampler_indices()
                    number_of_samples_per_heliostat = int(
                        heliostat_group.active_heliostats_mask.sum()
                        / (heliostat_group.active_heliostats_mask > 0).sum()
                    )
                    local_indices = (
                        sample_indices_for_local_rank[::number_of_samples_per_heliostat]
                        // number_of_samples_per_heliostat
                    )

                    # Crop predictions around center before comparing to measurements.
                    cropped_flux_distributions = (
                        bitmap.crop_flux_distributions_around_center(
                            flux_distributions=flux_distributions,
                            solar_tower=self.scenario.solar_tower,
                            target_area_indices=target_area_indices,
                            device=device,
                        )
                    )

                    # Flux loss (data-fit term): Compare predicted and measured flux maps.
                    flux_loss_per_sample = loss_definition(
                        prediction=cropped_flux_distributions,
                        ground_truth=measured_flux_distributions[
                            sample_indices_for_local_rank
                        ],
                        target_area_indices=target_area_indices[
                            sample_indices_for_local_rank
                        ],
                        reduction_dimensions=(
                            indices.batched_bitmap_e,
                            indices.batched_bitmap_u,
                        ),
                        device=device,
                    )
                    flux_loss_per_heliostat = core_utils.mean_loss_per_heliostat(
                        loss_per_sample=flux_loss_per_sample,
                        number_of_samples_per_heliostat=number_of_samples_per_heliostat,
                    )

                    # Add Augmented-Lagrangian constraint to ensure that flux integral is conserved,
                    # i.e., intensity does not get lost.
                    if epoch == 0:
                        flux_integrals_reference = cropped_flux_distributions.sum(
                            dim=(1, 2)
                        ).detach()
                    flux_integrals_relative_differences = (
                        cropped_flux_distributions.sum(dim=(1, 2))
                        - flux_integrals_reference
                    ) / (flux_integrals_reference + torch.tensor(self.epsilon))
                    flux_constraint_per_sample = torch.clamp(
                        -energy_tolerance - flux_integrals_relative_differences, min=0.0
                    )
                    flux_constraint_per_heliostat = core_utils.mean_loss_per_heliostat(
                        loss_per_sample=flux_constraint_per_sample,
                        number_of_samples_per_heliostat=number_of_samples_per_heliostat,
                    )
                    flux_integrals_constraint = (
                        lambda_flux_integral * flux_constraint_per_heliostat
                        + 0.5 * rho_flux_integral * flux_constraint_per_heliostat**2
                    )

                    # Regularization terms.
                    smoothness_loss_per_heliostat = torch.zeros_like(
                        flux_loss_per_heliostat, device=device
                    )
                    ideal_surface_loss_per_heliostat = torch.zeros_like(
                        flux_loss_per_heliostat, device=device
                    )
                    if weight_smoothness > 0:
                        smoothness_loss_per_heliostat = smoothness_regularizer(
                            current_control_points=heliostat_group.active_nurbs_control_points[
                                ::number_of_samples_per_heliostat
                            ][local_indices],
                            original_control_points=original_control_points[
                                local_indices
                            ],
                            device=device,
                        )
                    if weight_ideal_surface > 0:
                        ideal_surface_loss_per_heliostat = ideal_surface_regularizer(
                            current_control_points=heliostat_group.active_nurbs_control_points[
                                ::number_of_samples_per_heliostat
                            ][local_indices],
                            original_control_points=original_control_points[
                                local_indices
                            ],
                            device=device,
                        )
                    # Dynamic balancing of regularization magnitudes relative to data term.
                    alpha = (
                        weight_smoothness
                        * flux_loss_per_heliostat.mean()
                        / (
                            smoothness_loss_per_heliostat.mean()
                            + torch.tensor(self.epsilon)
                        )
                    )
                    beta = (
                        weight_ideal_surface
                        * flux_loss_per_heliostat.mean()
                        / (
                            ideal_surface_loss_per_heliostat.mean()
                            + torch.tensor(self.epsilon)
                        )
                    )

                    # Final per-heliostat loss
                    total_loss_per_heliostat = (
                        flux_loss_per_heliostat
                        + flux_integrals_constraint
                        + alpha * smoothness_loss_per_heliostat
                        + beta * ideal_surface_loss_per_heliostat
                    )
                    # Average over all heliostats in group.
                    total_loss = total_loss_per_heliostat.mean()

                    # Back-propagate through ray tracing and NURBS evaluation pipeline.
                    total_loss.backward()

                    # Update Augmented-Lagrangian multiplier.
                    with torch.no_grad():
                        lambda_flux_integral = torch.clamp(
                            lambda_flux_integral
                            + rho_flux_integral * flux_constraint_per_heliostat,
                            min=0.0,
                        )

                    # Nested-DDP gradient synchronization within heliostat-group subgroup.
                    if self.ddp_setup["is_nested"]:
                        # Reduce gradients within each heliostat group.
                        for param_group in optimizer.param_groups:
                            for param in param_group["params"]:
                                if param.grad is not None:
                                    torch.distributed.all_reduce(
                                        param.grad,
                                        op=torch.distributed.ReduceOp.SUM,
                                        group=self.ddp_setup["process_subgroup"],
                                    )
                                    param.grad /= self.ddp_setup[
                                        "heliostat_group_world_size"
                                    ]

                    # Geometry-preserving constraint: Keep the surfaces in their original geometric shape by locking
                    # the control points on the outer edges, i.e., zero/fix gradient on outer-edge control points.
                    optimizer.param_groups[indices.optimizer_param_group_0]["params"][
                        indices.optimizable_control_points
                    ].grad = self.lock_control_points_on_outer_edges(
                        gradients=optimizer.param_groups[
                            indices.optimizer_param_group_0
                        ]["params"][indices.optimizable_control_points].grad,
                        device=device,
                    )

                    optimizer.step()
                    if isinstance(
                        scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        scheduler.step(total_loss.detach())
                    else:
                        scheduler.step()

                    if epoch % log_step == 0 and rank == 0:
                        log.info(
                            f"Rank: {rank}, Epoch: {epoch}, Loss: {total_loss}, LR: {optimizer.param_groups[indices.optimizer_param_group_0]['lr']}"
                        )

                    total_loss_history.append(total_loss.detach().cpu().item())
                    flux_loss_history.append(
                        flux_loss_per_heliostat.mean().detach().cpu().item()
                    )
                    flux_integral.append(
                        flux_integrals_relative_differences.mean().detach().cpu().item()
                    )
                    smoothness_history.append(
                        (alpha * smoothness_loss_per_heliostat)
                        .mean()
                        .detach()
                        .cpu()
                        .item()
                    )
                    ideal_history.append(
                        (beta * ideal_surface_loss_per_heliostat)
                        .mean()
                        .detach()
                        .cpu()
                        .item()
                    )
                    flux_integral_history.append(
                        flux_integrals_constraint.mean().detach().cpu().item()
                    )

                    # Early stopping when loss did not improve for a predefined number of epochs.
                    stop = early_stopper.step(total_loss.item())

                    if stop:
                        log.info(f"Early stopping at epoch {epoch}.")
                        break

                    epoch += 1

                loss_history.append(
                    {
                        "total_loss": total_loss_history,
                        "flux_loss": flux_loss_history,
                        "smoothness_regularizer": smoothness_history,
                        "ideal_regularizer": ideal_history,
                        "flux_integral": flux_integral,
                        "flux_integral_constraint": flux_integral_history,
                    }
                )

                global_active_indices = torch.nonzero(
                    active_heliostats_mask != 0, as_tuple=True
                )[0]

                rank_active_indices_global = global_active_indices[local_indices]

                final_indices = (
                    rank_active_indices_global
                    + final_loss_start_indices[heliostat_group_index]
                )

                final_loss_per_heliostat[final_indices] = total_loss_per_heliostat

                log.info(f"Rank: {rank}, Surfaces reconstructed.")

        if self.ddp_setup["is_distributed"]:
            for index, heliostat_group in enumerate(
                self.scenario.heliostat_field.heliostat_groups
            ):
                source = self.ddp_setup["ranks_to_groups_mapping"][index]
                torch.distributed.broadcast(
                    heliostat_group.nurbs_control_points,
                    src=source[indices.first_rank_from_group],
                )
            torch.distributed.all_reduce(
                final_loss_per_heliostat, op=torch.distributed.ReduceOp.MIN
            )
            final_loss_history_all_groups: list[list[dict[str, list[float]]]] = [
                [] for _ in range(self.ddp_setup["world_size"])
            ]
            torch.distributed.all_gather_object(
                final_loss_history_all_groups, loss_history
            )

            log.info(f"Rank: {rank}, synchronized after surface reconstruction.")

        else:
            final_loss_history_all_groups = [loss_history]

        self.scenario.heliostat_field.update_surfaces(device=device)

        return final_loss_per_heliostat.detach().cpu(), final_loss_history_all_groups

    @staticmethod
    def lock_control_points_on_outer_edges(
        gradients: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Lock the u and v values of the control points on the outer edges of each facet.

        As the knots of the first and last knots on each facet have full multiplicity, the
        NURBS surfaces all start and end in control points. If the outer control points
        are not fixed in their u and v values, the reconstructed surfaces may not be
        rectangular anymore. To keep them rectangular, this function zeros the gradients
        of the u and v coordinates of all outer control points.

        Parameters
        ----------
        gradients : torch.Tensor
            The full control point gradient tensor for all active heliostats. Gradients on the
            outer edges will be zeroed; interior gradients are returned unchanged.
            Shape is ``[number_of_active_heliostats, number_of_facets_per_surface, number_of_control_points_u_direction, number_of_control_points_v_direction, 3]``.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The updated gradients.
            Shape is ``[number_of_active_heliostats, number_of_facets_per_surface, number_of_control_points_u_direction, number_of_control_points_v_direction, 3]``.
        """
        device = get_device(device=device)

        with torch.no_grad():
            fixed_gradients = gradients.clone()

            height = gradients.shape[indices.nurbs_control_points_u]
            width = gradients.shape[indices.nurbs_control_points_v]

            rows = (
                torch.arange(height, device=device)
                .unsqueeze(indices.unbatched_bitmap_u)
                .expand(height, width)
            )
            cols = (
                torch.arange(width, device=device)
                .unsqueeze(indices.unbatched_bitmap_e)
                .expand(height, width)
            )

            edge_mask = (
                (rows == 0) | (rows == height - 1) | (cols == 0) | (cols == width - 1)
            )

            fixed_gradients[:, :, :, :, : indices.z_coordinates] = torch.where(
                edge_mask.unsqueeze(indices.heliostat_dimension)
                .unsqueeze(indices.facet_index_unbatched)
                .unsqueeze(indices.nurbs_control_points),
                torch.tensor(0.0, device=device),
                gradients[:, :, :, :, : indices.z_coordinates],
            )

            return fixed_gradients
