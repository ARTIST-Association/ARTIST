import logging
import pathlib
from typing import Any, cast

import torch
from torch.optim.lr_scheduler import LRScheduler

from artist.core import learning_rate_schedulers
from artist.core.core_utils import per_heliostat_reduction, scale_loss
from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.core.loss_functions import Loss
from artist.data_loader import flux_distribution_loader, paint_loader
from artist.field.heliostat_group import HeliostatGroup
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, utils
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
    data : dict[str, str | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]]
        The data source name and the mapping of heliostat name and calibration data.
    optimization_configuration : dict[str, Any]
        The parameters for the optimizer, learning rate scheduler, regularizers and early stopping.
    number_of_surface_points : torch.Tensor
        The number of surface points of the reconstructed surfaces.
        Tensor of shape [2].
    bitmap_resolution : torch.Tensor
        The resolution of all bitmaps during reconstruction.
        Tensor of shape [2].

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
        data: dict[str, str | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]],
        optimization_configuration: dict[str, Any],
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
        data : dict[str, str | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]]
            The data source name and the mapping of heliostat name and calibration data.
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
        self.number_of_surface_points = number_of_surface_points.to(device)
        self.bitmap_resolution = bitmap_resolution.to(device)

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
            log.info("Start the surface reconstruction.")

        final_loss_per_heliostat = torch.full(
            (self.scenario.heliostat_field.number_of_heliostats_per_group.sum(),),
            torch.inf,
            device=device,
        )
        final_loss_start_indices = torch.cat(
            [
                torch.tensor([0], device=device),
                self.scenario.heliostat_field.number_of_heliostats_per_group.cumsum(0),
            ]
        )

        for heliostat_group_index in self.ddp_setup[
            config_dictionary.groups_to_ranks_mapping
        ][rank]:
            heliostat_group: HeliostatGroup = (
                self.scenario.heliostat_field.heliostat_groups[heliostat_group_index]
            )

            # Extract measured fluxes and their respective calibration properties data.
            heliostat_flux_path_mapping = []
            heliostat_calibration_mapping = []

            heliostat_data_mapping = cast(
                list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
                self.data[config_dictionary.heliostat_data_mapping],
            )
            for heliostat, path_properties, path_pngs in heliostat_data_mapping:
                if heliostat in heliostat_group.names:
                    heliostat_flux_path_mapping.append((heliostat, path_pngs))
                    heliostat_calibration_mapping.append((heliostat, path_properties))

            measured_flux_distributions = flux_distribution_loader.load_flux_from_png(
                heliostat_flux_path_mapping=heliostat_flux_path_mapping,
                heliostat_names=heliostat_group.names,
                resolution=self.bitmap_resolution,
                device=device,
            )

            if self.data[config_dictionary.data_source] == config_dictionary.paint:
                (
                    _,
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
                # Crop target fluxes.
                cropped_measured_flux_distributions = (
                    utils.crop_flux_distributions_around_center(
                        flux_distributions=measured_flux_distributions,
                        crop_width=config_dictionary.utis_crop_width,
                        crop_height=config_dictionary.utis_crop_height,
                        target_plane_widths=self.scenario.target_areas.dimensions[
                            target_area_mask
                        ][:, 0],
                        target_plane_heights=self.scenario.target_areas.dimensions[
                            target_area_mask
                        ][:, 1],
                        device=device,
                    )
                )

                # Activate heliostats.
                heliostat_group.activate_heliostats(
                    active_heliostats_mask=active_heliostats_mask, device=device
                )

                # Get the start indices for the separate heliostats in the active_-properties-tensors that contain heliostat duplicates for each sample.
                nonzero_active_heliostats_mask = active_heliostats_mask[
                    active_heliostats_mask > 0
                ]
                start_indices_heliostats = torch.cumsum(
                    torch.cat(
                        [
                            torch.tensor([0], device=device),
                            nonzero_active_heliostats_mask[:-1],
                        ]
                    ),
                    dim=0,
                )

                original_nurbs_control_points = (
                    heliostat_group.nurbs_control_points.clone().detach()
                )

                # Create NURBS evaluation points.
                evaluation_points = (
                    utils.create_nurbs_evaluation_grid(
                        number_of_evaluation_points=self.number_of_surface_points,
                        device=device,
                    )
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(
                        heliostat_group.number_of_active_heliostats,
                        heliostat_group.number_of_facets_per_heliostat,
                        -1,
                        -1,
                    )
                )

                # Create the optimizer.
                optimizer = torch.optim.Adam(
                    [heliostat_group.nurbs_control_points.requires_grad_()],
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
                    // self.optimization_configuration[config_dictionary.num_log]
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
                        evaluation_points=evaluation_points, device=device
                    )

                    # The alignment module and the ray tracer do not accept facetted points and normals, therefore they need to be reshaped.
                    heliostat_group.active_surface_points = new_surface_points.reshape(
                        heliostat_group.active_surface_points.shape[0], -1, 4
                    )
                    heliostat_group.active_surface_normals = (
                        new_surface_normals.reshape(
                            heliostat_group.active_surface_normals.shape[0], -1, 4
                        )
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
                        bitmap_resolution=self.bitmap_resolution,
                    )

                    # Perform heliostat-based ray tracing.
                    flux_distributions = ray_tracer.trace_rays(
                        incident_ray_directions=incident_ray_directions,
                        active_heliostats_mask=active_heliostats_mask,
                        target_area_mask=target_area_mask,
                        device=device,
                    )

                    # Reduce predicted fluxes from all ranks within each subgroup.
                    if self.ddp_setup[config_dictionary.is_nested]:
                        flux_distributions = torch.distributed.nn.functional.all_reduce(
                            flux_distributions,
                            group=self.ddp_setup[config_dictionary.process_subgroup],
                            op=torch.distributed.ReduceOp.SUM,
                        )

                    cropped_flux_distributions = (
                        utils.crop_flux_distributions_around_center(
                            flux_distributions=flux_distributions,
                            crop_width=config_dictionary.utis_crop_width,
                            crop_height=config_dictionary.utis_crop_height,
                            target_plane_widths=self.scenario.target_areas.dimensions[
                                target_area_mask
                            ][:, 0],
                            target_plane_heights=self.scenario.target_areas.dimensions[
                                target_area_mask
                            ][:, 1],
                            device=device,
                        )
                    )

                    # Loss comparing the predicted flux and the target flux.
                    flux_loss_per_sample = loss_definition(
                        prediction=cropped_flux_distributions,
                        ground_truth=cropped_measured_flux_distributions,
                        target_area_mask=target_area_mask,
                        reduction_dimensions=(1, 2),
                        device=device,
                    )

                    flux_loss_per_heliostat = per_heliostat_reduction(
                        per_sample_values=flux_loss_per_sample,
                        active_heliostats_mask=active_heliostats_mask,
                        device=device,
                    )

                    flux_loss_per_heliostat_summed = flux_loss_per_heliostat[
                        torch.isfinite(flux_loss_per_heliostat)
                    ].sum()
                    loss = flux_loss_per_heliostat_summed

                    # Include regularization terms.
                    for regularizer in self.optimization_configuration[
                        config_dictionary.regularizers
                    ]:
                        regularization_term = regularizer(
                            current_nurbs_control_points=heliostat_group.nurbs_control_points,
                            original_nurbs_control_points=original_nurbs_control_points,
                            surface_points=new_surface_points[start_indices_heliostats],
                            surface_normals=new_surface_normals[
                                start_indices_heliostats
                            ],
                            device=device,
                        ).sum()

                        scaled_regularization_term = scale_loss(
                            loss=regularization_term,
                            reference=flux_loss_per_heliostat_summed,
                            weight=regularizer.weight,
                        )

                        loss = loss + scaled_regularization_term

                    loss.backward()

                    if self.ddp_setup[config_dictionary.is_nested]:
                        # Reduce gradients within each heliostat group.
                        for param_group in optimizer.param_groups:
                            for param in param_group["params"]:
                                if param.grad is not None:
                                    torch.distributed.all_reduce(
                                        param.grad,
                                        op=torch.distributed.ReduceOp.SUM,
                                        group=self.ddp_setup[
                                            config_dictionary.process_subgroup
                                        ],
                                    )

                    # Keep the surfaces in their original geometric shape by locking the control points on the outer edges.
                    optimizer.param_groups[0]["params"][
                        0
                    ].grad = self.lock_control_points_on_outer_edges(
                        gradients=optimizer.param_groups[0]["params"][0].grad,
                        device=device,
                    )

                    optimizer.step()
                    if isinstance(
                        scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        scheduler.step(loss.detach())
                    else:
                        scheduler.step()

                    if epoch % log_step == 0 and rank == 0:
                        log.info(
                            f"Epoch: {epoch}, Loss: {loss}, LR: {optimizer.param_groups[0]['lr']}",
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
                ] = flux_loss_per_heliostat

                log.info(f"Rank: {rank}, surfaces reconstructed.")

        if self.ddp_setup[config_dictionary.is_distributed]:
            for index, heliostat_group in enumerate(
                self.scenario.heliostat_field.heliostat_groups
            ):
                source = self.ddp_setup[config_dictionary.ranks_to_groups_mapping][
                    index
                ]
                torch.distributed.broadcast(
                    heliostat_group.nurbs_control_points, src=source[0]
                )
            torch.distributed.all_reduce(
                final_loss_per_heliostat, op=torch.distributed.ReduceOp.MIN
            )

            log.info(f"Rank: {rank}, synchronized after surface reconstruction.")

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
            If None, ARTIST will automatically select the most appropriate
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

            height = gradients.shape[2]
            width = gradients.shape[3]

            rows = (
                torch.arange(height, device=device).unsqueeze(1).expand(height, width)
            )
            cols = torch.arange(width, device=device).unsqueeze(0).expand(height, width)

            edge_mask = (
                (rows == 0) | (rows == height - 1) | (cols == 0) | (cols == width - 1)
            )

            fixed_gradients[:, :, :, :, :2] = torch.where(
                edge_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1),
                torch.tensor(0.0, device=device),
                gradients[:, :, :, :, :2],
            )

            return fixed_gradients
