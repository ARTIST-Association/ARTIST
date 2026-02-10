import logging
import pathlib
from typing import Any, cast

import torch
from torch.optim.lr_scheduler import LRScheduler

from artist.core import core_utils, learning_rate_schedulers
from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.core.loss_functions import Loss
from artist.core.regularizers import IdealSurfaceRegularizer, SmoothnessRegularizer
from artist.data_parser.calibration_data_parser import CalibrationDataParser
from artist.field.heliostat_group import HeliostatGroup
from artist.scenario.scenario import Scenario
from artist.util import (
    config_dictionary,
    index_mapping,
    utils,
)
from artist.util.environment_setup import get_device
from artist.util.nurbs import NURBSSurfaces

log = logging.getLogger(__name__)
"""A logger for the surface reconstructor."""


class SurfaceReconstructor:
    """
    An optimizer used to reconstruct surfaces using NURBS and measured flux distributions.

    The surface reconstructor learns a surface representation from measured flux density
    distributions. The optimizable parameters for this optimization process are the
    NURBS control points.

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
    number_of_surface_points : torch.Tensor
        The number of surface points of the reconstructed surfaces.
        Tensor of shape [2].
    bitmap_resolution : torch.Tensor
        The resolution of all bitmaps during reconstruction.
        Tensor of shape [2].

    Note
    ----
    Each heliostat selected for reconstruction needs to have the same amount of samples as all others.

    Methods
    -------
    reconstruct_surfaces()
        Reconstruct NURBS surfaces from bitmaps.
    lock_control_points_on_outer_edges()
        Lock the u and v values of the control points on the outer edges of each facet.
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
        constraint_parameters: dict[str, Any],
        number_of_surface_points: torch.Tensor = torch.tensor([50, 50]),
        bitmap_resolution: torch.Tensor = torch.tensor([256, 256]),
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize the surface reconstructor.

        Parameters
        ----------
        ddp_setup : dict[str, Any]
           Information about the distributed environment, process_groups, devices, ranks, world_Size, heliostat group to ranks mapping.
        scenario : Scenario
            The scenario.
        data : dict[str, CalibrationDataParser | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]]
            The data parser and the mapping of heliostat name and calibration data.
        optimization_configuration : dict[str, Any]
            The parameters for the optimizer, learning rate scheduler and early stopping.
        number_of_surface_points : torch.Tensor
            The number of surface points of the reconstructed surfaces (default is torch.tensor([50,50])).
            Tensor of shape [2].
        bitmap_resolution : torch.Tensor
            The resolution of all bitmaps during reconstruction (default is torch.tensor([256,256])).
            Tensor of shape [2].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        device = get_device(device=device)

        rank = ddp_setup[config_dictionary.rank]

        if rank == 0:
            log.info("Create a surface reconstructor.")

        self.ddp_setup = ddp_setup
        self.scenario = scenario
        self.data = data
        self.optimization_configuration = optimization_configuration
        self.constraint_parameters = constraint_parameters
        self.number_of_surface_points = number_of_surface_points.to(device)
        self.bitmap_resolution = bitmap_resolution.to(device)

        self.epsilon = 1e-12

    def reconstruct_surfaces(
        self,
        loss_definition: Loss,
        device: torch.device | None = None,
    ) -> torch.Tensor:
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
            The final loss of the surface reconstruction for each heliostat in each group.
            Tensor of shape [total_number_of_heliostats_in_scenario].
        """
        device = get_device(device=device)

        rank = self.ddp_setup[config_dictionary.rank]

        if rank == 0:
            log.info("Beginning surface reconstruction.")

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
                measured_flux_distributions,
                _,
                incident_ray_directions,
                _,
                active_heliostats_mask,
                target_area_mask,
            ) = parser.parse_data_for_reconstruction(
                heliostat_data_mapping=heliostat_mapping,
                heliostat_group=heliostat_group,
                scenario=self.scenario,
                bitmap_resolution=self.bitmap_resolution,
                device=device,
            )

            if active_heliostats_mask.sum() > 0:
                # Create NURBS evaluation points.
                evaluation_points = (
                    utils.create_nurbs_evaluation_grid(
                        number_of_evaluation_points=self.number_of_surface_points,
                        device=device,
                    )
                    .unsqueeze(index_mapping.heliostat_dimension)
                    .unsqueeze(index_mapping.facet_index_unbatched)
                    .expand(
                        active_heliostats_mask.sum(),
                        heliostat_group.number_of_facets_per_heliostat,
                        -1,
                        -1,
                    )
                )

                with torch.no_grad():
                    original_control_points = heliostat_group.nurbs_control_points[
                        active_heliostats_mask > 0
                    ].clone()

                # Create the optimizer.
                optimizer = torch.optim.Adam(
                    [heliostat_group.nurbs_control_points.requires_grad_()],
                    lr=float(
                        self.optimization_configuration[
                            config_dictionary.initial_learning_rate
                        ]
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

                energy_per_flux_reference = torch.zeros_like(active_heliostats_mask)
                initial_lambda_energy = self.constraint_parameters[
                    config_dictionary.initial_lambda_energy
                ]
                lambda_energy = torch.full_like(
                    active_heliostats_mask,
                    initial_lambda_energy,
                    dtype=torch.float32,
                    device=device,
                )
                rho_energy = self.constraint_parameters[config_dictionary.rho_energy]
                energy_tolerance = self.constraint_parameters[
                    config_dictionary.energy_tolerance
                ]
                weight_smoothness = self.constraint_parameters[
                    config_dictionary.weight_smoothness
                ]
                weight_ideal_surface = self.constraint_parameters[
                    config_dictionary.weight_ideal_surface
                ]

                # Start the optimization.
                loss = torch.inf
                epoch = 0
                log_step = (
                    self.optimization_configuration[config_dictionary.max_epoch]
                    if self.optimization_configuration[config_dictionary.log_step] == 0
                    else self.optimization_configuration[config_dictionary.log_step]
                )
                max_epoch = torch.tensor(
                    [self.optimization_configuration[config_dictionary.max_epoch]],
                    device=device,
                )
                while (
                    loss
                    > float(
                        self.optimization_configuration[config_dictionary.tolerance]
                    )
                    and epoch <= max_epoch
                ):
                    optimizer.zero_grad()

                    # Activate heliostats.
                    heliostat_group.activate_heliostats(
                        active_heliostats_mask=active_heliostats_mask, device=device
                    )

                    # Create NURBS.
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

                    # The alignment module and the ray tracer do not accept facetted points and normals, therefore they need to be reshaped.
                    heliostat_group.active_surface_points = new_surface_points.reshape(
                        heliostat_group.active_surface_points.shape[
                            index_mapping.heliostat_dimension
                        ],
                        -1,
                        4,
                    )
                    heliostat_group.active_surface_normals = (
                        new_surface_normals.reshape(
                            heliostat_group.active_surface_normals.shape[
                                index_mapping.heliostat_dimension
                            ],
                            -1,
                            4,
                        )
                    )

                    # Align heliostats.
                    heliostat_group.align_surfaces_with_incident_ray_directions(
                        aim_points=self.scenario.target_areas.centers[target_area_mask],
                        incident_ray_directions=incident_ray_directions,
                        active_heliostats_mask=active_heliostats_mask,
                        device=device,
                    )

                    # Create a parallelized ray tracer. Blocking is always deactivated for this reconstruction.
                    ray_tracer = HeliostatRayTracer(
                        scenario=self.scenario,
                        heliostat_group=heliostat_group,
                        blocking_active=False,
                        world_size=self.ddp_setup[
                            config_dictionary.heliostat_group_world_size
                        ],
                        rank=self.ddp_setup[config_dictionary.heliostat_group_rank],
                        batch_size=self.optimization_configuration[
                            config_dictionary.batch_size
                        ],
                        random_seed=self.ddp_setup[
                            config_dictionary.heliostat_group_rank
                        ],
                        bitmap_resolution=self.bitmap_resolution,
                    )

                    # Perform heliostat-based ray tracing.
                    flux_distributions = ray_tracer.trace_rays(
                        incident_ray_directions=incident_ray_directions,
                        active_heliostats_mask=active_heliostats_mask,
                        target_area_mask=target_area_mask,
                        device=device,
                    )

                    sample_indices_for_local_rank = ray_tracer.get_sampler_indices()
                    number_of_samples_per_heliostat = int(
                        heliostat_group.active_heliostats_mask.sum()
                        / (heliostat_group.active_heliostats_mask > 0).sum()
                    )
                    local_indices = (
                        sample_indices_for_local_rank[::number_of_samples_per_heliostat]
                        // number_of_samples_per_heliostat
                    )

                    cropped_flux_distributions = (
                        utils.crop_flux_distributions_around_center(
                            flux_distributions=flux_distributions,
                            crop_width=config_dictionary.utis_crop_width,
                            crop_height=config_dictionary.utis_crop_height,
                            target_plane_widths=self.scenario.target_areas.dimensions[
                                target_area_mask[sample_indices_for_local_rank]
                            ][:, index_mapping.target_area_width],
                            target_plane_heights=self.scenario.target_areas.dimensions[
                                target_area_mask[sample_indices_for_local_rank]
                            ][:, index_mapping.target_area_height],
                            device=device,
                        )
                    )

                    # Flux loss.
                    flux_loss_per_sample = loss_definition(
                        prediction=cropped_flux_distributions,
                        ground_truth=measured_flux_distributions[
                            sample_indices_for_local_rank
                        ],
                        target_area_mask=target_area_mask[
                            sample_indices_for_local_rank
                        ],
                        reduction_dimensions=(
                            index_mapping.batched_bitmap_e,
                            index_mapping.batched_bitmap_u,
                        ),
                        device=device,
                    )
                    flux_loss_per_heliostat = core_utils.mean_loss_per_heliostat(
                        loss_per_sample=flux_loss_per_sample,
                        number_of_samples_per_heliostat=number_of_samples_per_heliostat,
                        device=device,
                    )

                    # Augmented Lagrangian.
                    if epoch == 0:
                        energy_per_flux_reference = cropped_flux_distributions.sum(
                            dim=(1, 2)
                        ).detach()
                    g_energy = (
                        cropped_flux_distributions.sum(dim=(1, 2))
                        - energy_per_flux_reference
                    ) / (energy_per_flux_reference + self.epsilon)
                    energy_constraint = torch.minimum(
                        g_energy + energy_tolerance, torch.zeros_like(g_energy)
                    )
                    energy_constraint_per_heliostat = core_utils.mean_loss_per_heliostat(
                        loss_per_sample=energy_constraint,
                        number_of_samples_per_heliostat=number_of_samples_per_heliostat,
                        device=device,
                    )
                    constraint = (
                        lambda_energy.detach() * energy_constraint_per_heliostat
                        + 0.5 * rho_energy * energy_constraint_per_heliostat**2
                    )

                    # Regularization terms.
                    smoothness_loss_per_heliostat = torch.zeros_like(
                        flux_loss_per_heliostat, device=device
                    )
                    ideal_surface_loss_per_heliostat = torch.zeros_like(
                        flux_loss_per_heliostat, device=device
                    )
                    if (
                        self.optimization_configuration[config_dictionary.regularizers]
                        is not None
                    ):
                        for regularizer in self.optimization_configuration[
                            config_dictionary.regularizers
                        ]:
                            regularization_term_active_heliostats = regularizer(
                                current_control_points=heliostat_group.active_nurbs_control_points[
                                    ::number_of_samples_per_heliostat
                                ][local_indices],
                                original_control_points=original_control_points[
                                    local_indices
                                ],
                                device=device,
                            )
                            if isinstance(regularizer, SmoothnessRegularizer):
                                smoothness_loss_per_heliostat = (
                                    regularization_term_active_heliostats
                                )
                            if isinstance(regularizer, IdealSurfaceRegularizer):
                                ideal_surface_loss_per_heliostat = (
                                    regularization_term_active_heliostats
                                )
                    alpha = (
                        weight_smoothness
                        * flux_loss_per_heliostat.mean()
                        / (smoothness_loss_per_heliostat.mean() + self.epsilon)
                    )
                    beta = (
                        weight_ideal_surface
                        * flux_loss_per_heliostat.mean()
                        / (ideal_surface_loss_per_heliostat.mean() + self.epsilon)
                    )

                    total_loss_per_heliostat = (
                        flux_loss_per_heliostat
                        + constraint
                        + alpha * smoothness_loss_per_heliostat
                        + beta * ideal_surface_loss_per_heliostat
                    )

                    total_loss = total_loss_per_heliostat.mean()
                    total_loss.backward()

                    # Keep the surfaces in their original geometric shape by locking the control points on the outer edges.
                    optimizer.param_groups[index_mapping.optimizer_param_group_0][
                        "params"
                    ][
                        index_mapping.optimizable_control_points
                    ].grad = self.lock_control_points_on_outer_edges(
                        gradients=optimizer.param_groups[
                            index_mapping.optimizer_param_group_0
                        ]["params"][index_mapping.optimizable_control_points].grad,
                        device=device,
                    )

                    optimizer.step()
                    if isinstance(
                        scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        scheduler.step(total_loss.detach())
                    else:
                        scheduler.step()

                    with torch.no_grad():
                        lambda_energy += (
                            rho_energy * energy_constraint_per_heliostat.detach()
                        )
                        lambda_energy.clamp_(min=0.0)

                    if epoch % log_step == 0 and rank == 0:
                        log.info(
                            f"Rank: {rank}, Epoch: {epoch}, Loss: {total_loss}, LR: {optimizer.param_groups[index_mapping.optimizer_param_group_0]['lr']}"
                        )

                    # if rank == 0 and epoch % 10 == 0:
                    #     fig, axes = plt.subplots(nrows=cropped_flux_distributions.shape[0], ncols=2, figsize=(6, 3*cropped_flux_distributions.shape[0]))

                    #     for i in range(cropped_flux_distributions.shape[0]):
                    #         # Compute min/max across the pair for shared color scale
                    #         vmin = min(cropped_flux_distributions[i].detach().min(), measured_flux_distributions[i].detach().min()).item()
                    #         vmax = max(cropped_flux_distributions[i].detach().max(), measured_flux_distributions[i].detach().max()).item()

                    #         im0 = axes[i, 0].imshow(cropped_flux_distributions[i].detach().cpu(), cmap='inferno',) #vmin=vmin, vmax=vmax)
                    #         axes[i, 0].set_title(f"Predicted {cropped_flux_distributions[i].detach().cpu().sum()}")
                    #         axes[i, 0].axis('off')

                    #         im1 = axes[i, 1].imshow(measured_flux_distributions[i].detach().cpu(), cmap='inferno',) # vmin=vmin, vmax=vmax)
                    #         axes[i, 1].set_title(f"Ground Truth {i}")
                    #         axes[i, 1].axis('off')

                    #         # Shared colorbar for the pair
                    #         #fig.colorbar(im1, ax=axes[i, :], orientation='vertical', fraction=0.05)

                    #         plt.tight_layout()
                    #         plt.savefig(f"epoch_{epoch}")

                    # Early stopping when loss did not improve since a predefined number of epochs.
                    stop = early_stopper.step(loss)

                    if stop:
                        log.info(f"Early stopping at epoch {epoch}.")
                        break

                    epoch += 1

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

        if self.ddp_setup[config_dictionary.is_distributed]:
            for index, heliostat_group in enumerate(
                self.scenario.heliostat_field.heliostat_groups
            ):
                source = self.ddp_setup[config_dictionary.ranks_to_groups_mapping][
                    index
                ]
                torch.distributed.broadcast(
                    heliostat_group.nurbs_control_points,
                    src=source[index_mapping.first_rank_from_group],
                )
            torch.distributed.all_reduce(
                final_loss_per_heliostat, op=torch.distributed.ReduceOp.MIN
            )

            log.info(f"Rank: {rank}, synchronized after surface reconstruction.")

        self.scenario.heliostat_field.update_surfaces(device=device)

        return final_loss_per_heliostat

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
            The gradients of the outer control points.
            Tensor of shape [number_of_active_heliostats, number_of_facets_per_surface, number_of_control_points_u_direction, number_of_control_points_v_direction, 3].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The updated gradients.
            Tensor of shape [number_of_active_heliostats, number_of_facets_per_surface, number_of_control_points_u_direction, number_of_control_points_v_direction, 3].
        """
        device = get_device(device=device)

        with torch.no_grad():
            fixed_gradients = gradients.clone()

            height = gradients.shape[index_mapping.nurbs_control_points_u]
            width = gradients.shape[index_mapping.nurbs_control_points_v]

            rows = (
                torch.arange(height, device=device)
                .unsqueeze(index_mapping.unbatched_bitmap_u)
                .expand(height, width)
            )
            cols = (
                torch.arange(width, device=device)
                .unsqueeze(index_mapping.unbatched_bitmap_e)
                .expand(height, width)
            )

            edge_mask = (
                (rows == 0) | (rows == height - 1) | (cols == 0) | (cols == width - 1)
            )

            fixed_gradients[:, :, :, :, : index_mapping.z_coordinates] = torch.where(
                edge_mask.unsqueeze(index_mapping.heliostat_dimension)
                .unsqueeze(index_mapping.facet_index_unbatched)
                .unsqueeze(index_mapping.nurbs_control_points),
                torch.tensor(0.0, device=device),
                gradients[:, :, :, :, : index_mapping.z_coordinates],
            )

            return fixed_gradients
