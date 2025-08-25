import logging
import pathlib
from typing import Callable

import torch

from artist.core import learning_rate_schedulers, loss_functions
from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.data_loader import flux_distribution_loader, paint_loader
from artist.scenario.scenario import Scenario
from artist.util import utils
from artist.util.environment_setup import DistributedEnvironmentTypedDict, get_device
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
    ddp_setup : DistributedEnvironmentTypedDict
        Information about the distributed environment, process_groups, devices, ranks, world_Size, heliostat group to ranks mapping.
    scenario : Scenario
        The scenario.
    heliostat_data_mapping : list[tuple[str, list[pathlib.Path, list[pathlib.Path]]]
        The mapping of heliostat and reconstruction data.
    number_of_surface_points : torch.Tensor
        The number of surface points of the reconstructed surfaces.
        Tensor of shape [2].
    bitmap_resolution : torch.Tensor
        The resolution of all bitmaps during reconstruction.
        Tensor of shape [2].
    flux_loss_weight : float
        A weight for the flux loss.
    ideal_surface_loss_weight : float
        A weight for the loss describing the deviation of predicted to ideal control points.
    total_variation_loss_points_weight : float
        A weight for the loss describing the total variation across predicted surface points.
    total_variation_loss_normals_weight : float
        A weight for the loss describing the total variation across predicted surface normals.
    number_of_neighbors_tv_loss : int
        The number of nearest neighbors to consider for the total variation loss.
    sigma_tv_loss : float
        Determines how quickly the weight falls off as the distance increases for the total variation loss.
    early_stopping_threshold : float
        A threshold to stop optimization if the loss has not improved significantly.
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
    reconstruct_surfaces()
        Reconstruct NURBS surfaces from bitmaps.
    fixate_control_points_on_outer_edges()
        Fixate the u and v values of the control points on the outer edges of each facet.
    total_variation_loss_per_surface()
        Compute the total variation loss for surfaces.
    loss_per_heliostat()
        Compute mean losses for each heliostat with multiple samples.
    """

    def __init__(
        self,
        ddp_setup: DistributedEnvironmentTypedDict,
        scenario: Scenario,
        heliostat_data_mapping: list[
            tuple[str, list[pathlib.Path], list[pathlib.Path]]
        ],
        number_of_surface_points: torch.Tensor = torch.tensor([50, 50]),
        bitmap_resolution: torch.Tensor = torch.tensor([256, 256]),
        flux_loss_weight: float = 1.0,
        ideal_surface_loss_weight: float = 0.5,
        total_variation_loss_points_weight: float = 0.5,
        total_variation_loss_normals_weight: float = 0.5,
        number_of_neighbors_tv_loss: int = 20,
        sigma_tv_loss: float | None = None,
        early_stopping_threshold: float = 1e-3,
        initial_learning_rate: float = 1e-5,
        scheduler: Callable[
            ..., torch.optim.lr_scheduler.LRScheduler
        ] = learning_rate_schedulers.NoOpScheduler,
        scheduler_parameters: dict[str, float] = {},
        tolerance: float = 0.0005,
        max_epoch: int = 1000,
        num_log: int = 3,
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize the surface reconstructor.

        Parameters
        ----------
        ddp_setup : DistributedEnvironmentTypedDict
           Information about the distributed environment, process_groups, devices, ranks, world_Size, heliostat group to ranks mapping.
        scenario : Scenario
            The scenario.
        heliostat_data_mapping : list[tuple[str, list[pathlib.Path, list[pathlib.Path]]]
            The mapping of heliostat and reconstruction data.
        number_of_surface_points : torch.Tensor
            The number of surface points of the reconstructed surfaces.
            Tensor of shape [2].
        bitmap_resolution : torch.Tensor
            The resolution of all bitmaps during reconstruction.
            Tensor of shape [2].
        flux_loss_weight : float
            A weight for the flux loss (default is 1.0).
        ideal_surface_loss_weight : float
            A weight for the loss describing the deviation of predicted to ideal control points (default is 1.0).
        total_variation_loss_points_weight : float
            A weight for the loss describing the total variation across predicted surface points (default is 1.0).
        total_variation_loss_normals_weight : float
            A weight for the loss describing the total variation across predicted surface normals (default is 1.0).
        number_of_neighbors_tv_loss : int
            The number of nearest neighbors to consider for the total variation loss (default is 20).
        sigma_tv_loss : float | None
            Determines how quickly the weight falls off as the distance increases for the total variation loss (default is None).
        early_stopping_threshold : float
            A threshold to stop optimization if the loss has not improved significantly (default is 1e-3).
        initial_learning_rate : float
            The initial learning rate for the optimizer (default is 1e-5).
        tolerance : float
            The tolerance during optimization (default is 0.0005).
        max_epoch : int
            The maximum optimization epoch (default is 1000).
        num_log : int
            The number of log statements during optimization (default is 3).
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        device = get_device(device=device)

        rank = ddp_setup["rank"]

        if rank == 0:
            log.info("Create a surface reconstructor.")

        self.ddp_setup = ddp_setup
        self.scenario = scenario
        self.heliostat_data_mapping = heliostat_data_mapping
        self.number_of_surface_points = number_of_surface_points.to(device)
        self.bitmap_resolution = bitmap_resolution.to(device)
        self.flux_loss_weight = flux_loss_weight
        self.ideal_surface_loss_weight = ideal_surface_loss_weight
        self.total_variation_loss_points_weight = total_variation_loss_points_weight
        self.total_variation_loss_normals_weight = total_variation_loss_normals_weight
        self.number_of_neighbors_tv_loss = number_of_neighbors_tv_loss
        self.sigma_tv_loss = sigma_tv_loss
        self.early_stopping_threshold = early_stopping_threshold
        self.initial_learning_rate = initial_learning_rate
        self.scheduler = scheduler
        self.scheduler_parameters = scheduler_parameters
        self.tolerance = tolerance
        self.max_epoch = max_epoch
        self.num_log = num_log

    def reconstruct_surfaces(
        self,
        loss_function: Callable[..., torch.Tensor],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Reconstruct NURBS surfaces from bitmaps.

        Parameters
        ----------
        loss_function : Callable[..., torch.Tensor]
            A callable function that computes the loss. It accepts predictions and targets
            and optionally other keyword arguments and return a tensor with loss values.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The final loss of the reconstruction for each heliostat group.
        """
        device = get_device(device=device)

        rank = self.ddp_setup["rank"]

        if rank == 0:
            log.info("Start the surface reconstruction.")

        final_loss_per_group = torch.full(
            self.scenario.heliostat_field.number_of_heliostat_groups,
            torch.inf,
            device=device,
        )

        for heliostat_group_index in self.ddp_setup["groups_to_ranks_mapping"][rank]:
            heliostat_group = self.scenario.heliostat_field.heliostat_groups[
                heliostat_group_index
            ]

            # Extract measured fluxes and their respective calibration properties data.
            heliostat_flux_path_mapping = []
            heliostat_calibration_mapping = []

            for heliostat, path_properties, path_pngs in self.heliostat_data_mapping:
                if heliostat in heliostat_group.names:
                    heliostat_flux_path_mapping.append((heliostat, path_pngs))
                    heliostat_calibration_mapping.append((heliostat, path_properties))

            normalized_measured_flux_distributions = (
                flux_distribution_loader.load_flux_from_png(
                    heliostat_flux_path_mapping=heliostat_flux_path_mapping,
                    heliostat_names=heliostat_group.names,
                    resolution=self.bitmap_resolution,
                    device=device,
                )
            )
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

            if active_heliostats_mask.sum() > 0:
                # Activate heliostats.
                heliostat_group.activate_heliostats(
                    active_heliostats_mask=active_heliostats_mask, device=device
                )

                # Get the start indices for the seperate heliostats in the active_-properties-tensors that contain heliostat duplicats for each sample.
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
                    lr=self.initial_learning_rate,
                )

                # Create a learning rate scheduler.
                scheduler = self.scheduler(
                    optimizer=optimizer, paramerters=self.scheduler_parameters
                )

                # Start the optimization.
                ideal_surface_loss_function = torch.nn.MSELoss()
                current_active_nurbs_control_points = torch.zeros_like(
                    heliostat_group.active_nurbs_control_points, device=device
                )
                loss_last_epoch = torch.inf
                loss_improvement = torch.inf
                epoch = 0
                log_step = self.max_epoch // self.num_log
                while (
                    loss_last_epoch > self.tolerance
                    and epoch <= self.max_epoch
                    and loss_improvement > self.early_stopping_threshold
                ):
                    optimizer.zero_grad()

                    current_active_nurbs_control_points = (
                        heliostat_group.active_nurbs_control_points.detach()
                    )

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
                        world_size=self.ddp_setup["heliostat_group_world_size"],
                        rank=self.ddp_setup["heliostat_group_rank"],
                        batch_size=heliostat_group.number_of_active_heliostats,
                        random_seed=self.ddp_setup["heliostat_group_rank"],
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
                    if self.ddp_setup["is_nested"]:
                        flux_distributions = torch.distributed.nn.functional.all_reduce(
                            flux_distributions,
                            group=self.ddp_setup["process_subgroup"],
                            op=torch.distributed.ReduceOp.SUM,
                        )

                    # Loss regarding the predicted flux and the target flux.
                    flux_loss_per_heliostat = self.loss_per_heliostat(
                        active_heliostats_mask=active_heliostats_mask,
                        predictions=flux_distributions,
                        targets=normalized_measured_flux_distributions,
                        loss_function=loss_function,
                        target_area_dimensions=ray_tracer.scenario.target_areas.dimensions[
                            target_area_mask
                        ],
                        number_of_rays=ray_tracer.light_source.number_of_rays,
                        device=device,
                    ).sum()

                    # Loss regarding predicted control points deviation from ideal (flat but canted) control points.
                    ideal_surface_loss = ideal_surface_loss_function(
                        heliostat_group.nurbs_control_points[
                            (active_heliostats_mask > 0).nonzero(as_tuple=True)[0]
                        ],
                        current_active_nurbs_control_points[start_indices_heliostats],
                    )

                    # Loss regarding smoothness of surface points.
                    total_variation_loss_points = self.total_variation_loss_per_surface(
                        surfaces=new_surface_points[start_indices_heliostats],
                        number_of_neighbors=self.number_of_neighbors_tv_loss,
                        sigma=self.sigma_tv_loss,
                        device=device,
                    ).sum()

                    # Loss regarding smoothness of surface normals.
                    total_variation_loss_normals = (
                        self.total_variation_loss_per_surface(
                            surfaces=new_surface_normals[start_indices_heliostats],
                            number_of_neighbors=self.number_of_neighbors_tv_loss,
                            sigma=self.sigma_tv_loss,
                            device=device,
                        ).sum()
                    )

                    # Sum of weighted losses.
                    loss = (
                        flux_loss_per_heliostat
                        + loss_functions.scale_loss(
                            loss=ideal_surface_loss,
                            reference_loss=flux_loss_per_heliostat,
                            weight=self.ideal_surface_loss_weight,
                        )
                        + loss_functions.scale_loss(
                            loss=total_variation_loss_points,
                            reference_loss=flux_loss_per_heliostat,
                            weight=self.total_variation_loss_points_weight,
                        )
                        + loss_functions.scale_loss(
                            loss=total_variation_loss_normals,
                            reference_loss=flux_loss_per_heliostat,
                            weight=self.total_variation_loss_normals_weight,
                        )
                    )
                    loss.backward()

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

                    # Keep the surfaces in their original geometric shape by fixating the control points on the outer edges.
                    optimizer.param_groups[0]["params"][
                        0
                    ].grad = self.fixate_control_points_on_outer_edges(
                        gradients=optimizer.param_groups[0]["params"][0].grad,
                        device=device,
                    )

                    optimizer.step()
                    scheduler.step()

                    if epoch % log_step == 0 and rank == 0:
                        log.info(
                            f"Epoch: {epoch}, Loss: {loss}, LR: {optimizer.param_groups[0]['lr']}",
                        )

                    # Early stopping.
                    loss_improvement = loss_last_epoch - loss

                    epoch += 1

                final_loss_per_group[heliostat_group_index] = loss
                log.info(f"Rank: {rank}, surfaces reconstructed.")

        if self.ddp_setup["is_distributed"]:
            for index, heliostat_group in enumerate(
                self.scenario.heliostat_field.heliostat_groups
            ):
                source = self.ddp_setup["ranks_to_groups_mapping"][index]
                torch.distributed.broadcast(
                    heliostat_group.nurbs_control_points, src=source[0]
                )

            log.info(f"Rank: {rank}, synchronised after surface reconstruction.")

    @staticmethod
    def fixate_control_points_on_outer_edges(
        gradients: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Fixate the u and v values of the control points on the outer edges of each facet.

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

    @staticmethod
    def total_variation_loss_per_surface(
        surfaces: torch.Tensor,
        number_of_neighbors: int = 20,
        sigma: float | None = None,
        batch_size: int = 512,
        epsilon: float = 1e-8,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Compute the total variation loss for surfaces.

        This loss term can be used as an addition to the overall loss during the surface reconstruction
        optimization. It supresses the noise in the surface. It measures the noise in the surface by
        taking absolute differences in the z values of the provided points. This loss implementation
        focuses on local smoothness by applying a Gaussian distance weight and thereby letting
        closer points contribute more. This loss implementation is batched and can handle multiple
        surfaces which are further batched in facets.

        Parameters
        ----------
        surfaces : torch.Tensor
            The surfaces.
            Tensor of shape [number_of_active_heliostats, number_of_facets_per_surface, number_of_surface_points_per_facet, 4].
        number_of_neighbors : int
            The number of nearest neighbors to consider (default is 20).
        sigma : float | None
            Determines how quickly the weight falls off as the distance increases (default is None).
        batch_size : int
            Used to process smaller batches of points instead of creating full distance matrices for all points (default is 512).
        epsilon : float
            A small vlaue used to prevent divisions by zero (defualt is 1e-8).
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The total variation loss for all provided surfaces.
            Tensor of shape [number_of_active_heliostats, number_of_facets_per_surface].
        """
        device = get_device(device=device)

        number_of_surfaces, number_of_facets, number_of_surface_points_per_facet, _ = (
            surfaces.shape
        )
        coordinates = surfaces[:, :, :, :2]
        z_values = surfaces[:, :, :, 2]

        # Set sigma. Determines how quickly the weight falls off as the distance increases.
        if sigma is None:
            coordinates_std = coordinates.std(dim=1).mean().item()
            sigma = max(coordinates_std * 0.1, 1e-6)
        sigma = float(sigma + 1e-12)

        variation_loss_sum = torch.zeros(
            (number_of_surfaces, number_of_facets), device=device
        )
        number_of_valid_neighbors = torch.zeros(
            (number_of_surfaces, number_of_facets), device=device
        )

        # Iterate over query points in batches to limit memory usage.
        for start_index in range(0, number_of_surface_points_per_facet, batch_size):
            # The loss will be distance weighted. Instead of building a full distance matrix for all points,
            # the search is batched to make it more efficient. Every batch uses torch.cdist to find up to k
            # nearest neighbors, this can optionally also be limited by a search radius.
            end_index = min(
                start_index + batch_size, number_of_surface_points_per_facet
            )
            number_of_points_in_batch = end_index - start_index

            batch_coordinates = coordinates[:, :, start_index:end_index, :]
            batch_z_values = z_values[:, :, start_index:end_index]

            # Compute pairwise distances between the current batch coordinates and all coordinates.
            distances = torch.cdist(batch_coordinates, coordinates)

            # Set distances where batch_coordinates == coordinate to a large value, to exclude them.
            rows = torch.arange(number_of_points_in_batch, device=device)
            cols = (start_index + rows).to(device)
            self_mask = torch.zeros_like(distances, dtype=torch.bool)
            self_mask[:, :, rows, cols] = True
            masked_distances = torch.where(
                self_mask, torch.full_like(distances, 1e9), distances
            )

            # Select the k nearest neighbors (or fewer if the coordinate is near an edge).
            number_of_neighbors_to_select = min(
                number_of_neighbors, number_of_surface_points_per_facet - 1
            )
            selected_distances, selected_indices = torch.topk(
                masked_distances, number_of_neighbors_to_select, largest=False, dim=3
            )
            valid_mask = selected_distances < 1e9

            # Get all z_values of the selected neighbors and the absolute z_value_variations.
            z_values_neighbors = torch.gather(
                z_values.unsqueeze(2).expand(-1, -1, number_of_points_in_batch, -1),
                3,
                selected_indices,
            )
            z_value_variations = torch.abs(
                batch_z_values.unsqueeze(-1) - z_values_neighbors
            )

            # Set Gaussian weights using the selected distances.
            z_value_variations = torch.abs(
                batch_z_values.unsqueeze(-1) - z_values_neighbors
            )
            weights = torch.exp(-0.5 * (selected_distances / sigma) ** 2)
            weights = weights * valid_mask.type_as(weights)
            z_value_variations = z_value_variations * valid_mask.type_as(
                z_value_variations
            )

            # Accumulate weighted z_value_variations.
            variation_loss_sum = variation_loss_sum + (
                weights * z_value_variations
            ).sum(dim=(2, 3))
            number_of_valid_neighbors = number_of_valid_neighbors + valid_mask.type_as(
                z_value_variations
            ).sum(dim=(2, 3))

        # Batched total variation losses.
        variation_loss_final = variation_loss_sum / (
            number_of_valid_neighbors + epsilon
        )

        return variation_loss_final

    @staticmethod
    def loss_per_heliostat(
        active_heliostats_mask: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_function: Callable[..., torch.Tensor],
        target_area_dimensions: torch.Tensor,
        number_of_rays: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Compute mean losses for each heliostat with multiple samples.

        If the active heliostats of one group have different amounts of samples to train on, i.e.
        one heliostat is trained with more flux images than another, this function makes sure that
        each heliostat still contributes equally to the overall loss. This function computes the MSE
        loss for each heliostat and sums across heliostats to create a single loss.

        Parameters
        ----------
        active_heliostats_mask : torch.Tensor
            A mask defining which heliostats are activated.
            Tensor of shape [number_of_heliostats].
        predictions : torch.Tensor
            The predicted values for all samples from all active heliostats.
            Tensor of shape [number_of_active_heliostats, bitmap_resolution_e, bitmap_resolution_u].
        targets : torch.Tensor
            The target values for all samples from all active heliostats.
            Tensor of shape [number_of_active_heliostats, bitmap_resolution_e, bitmap_resolution_u].
        loss_function : Callable[..., torch.Tensor]
            A callable function that computes the loss. It accepts predictions and targets
            and optionally other keyword arguments and return a tensor with loss values.
        target_area_dimensions : torch.Tensor
            The dimensions of the tower target areas aimed at.
            Tensor of shape [number_of_flux_distributions, 2].
        number_of_rays : int
            The number of rays used to generate the flux.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The mean loss per heliostat.
            Tensor of shape [number_of_active_heliostats].
        """
        device = get_device(device=device)

        # Compute per-sample losses.
        per_sample_losses = loss_function(
            predictions=predictions,
            targets=targets,
            target_area_dimensions=target_area_dimensions,
            number_of_rays=number_of_rays,
            device=device,
        )

        # A sample to heliostat index mapping.
        heliostat_ids = torch.repeat_interleave(
            torch.arange(len(active_heliostats_mask), device=device),
            active_heliostats_mask,
        )

        loss_sum_per_heliostat = torch.zeros(len(active_heliostats_mask), device=device)
        loss_sum_per_heliostat = loss_sum_per_heliostat.index_add(
            0, heliostat_ids, per_sample_losses
        )

        # Compute mean MSE per heliostat on each rank.
        number_of_samples_per_heliostat = torch.zeros(
            len(active_heliostats_mask), device=device
        )
        number_of_samples_per_heliostat.index_add_(
            0, heliostat_ids, torch.ones_like(per_sample_losses, device=device)
        )

        counts_clamped = number_of_samples_per_heliostat.clamp_min(1.0)
        mean_loss_per_heliostat = loss_sum_per_heliostat / counts_clamped
        mean_loss_per_heliostat = mean_loss_per_heliostat * (
            number_of_samples_per_heliostat > 0
        )

        return mean_loss_per_heliostat
