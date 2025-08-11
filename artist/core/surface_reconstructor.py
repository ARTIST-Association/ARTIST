import logging
import pathlib

import torch

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
    normalized_measured_flux_distributions : torch.Tensor
        The measured, normalized flux density distributions.
        Tensor of shape [number_of_active_heliostats, bitmap_resolution_e, bitmap_resolution_u].
    incident_ray_directions : torch.Tensor
        The incident ray directions of the measured fluxes.
        Tensor of shape [number_of_active_heliostats, 4].
    heliostats_mask : torch.Tensor
        A mask for the selected heliostats for reconstruction.
        Tensor of shape [number_of_heliostats].
    target_area_mask : torch.Tensor
        The indices of the target area for each reconstruction.
        Tensor of shape [number_of_active_heliostats].
    number_of_surface_points : torch.Tensor
        The number of surface points of the reconstructed surfaces.
        Tensor of shape [2].
    resolution : torch.Tensor
        The resolution of all bitmaps during reconstruction.
        Tensor of shape [2].
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
    """

    def __init__(
        self,
        ddp_setup: DistributedEnvironmentTypedDict,
        scenario: Scenario,
        heliostat_data_mapping: list[
            tuple[str, list[pathlib.Path], list[pathlib.Path]]
        ],
        number_of_surface_points: torch.Tensor = torch.tensor([50, 50]),
        resolution: torch.Tensor = torch.tensor([256, 256]),
        initial_learning_rate: float = 1e-5,
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
        resolution : torch.Tensor
            The resolution of all bitmaps during reconstruction.
            Tensor of shape [2].
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
        self.resolution = resolution.to(device)
        self.initial_learning_rate = initial_learning_rate
        self.tolerance = tolerance
        self.max_epoch = max_epoch
        self.num_log = num_log

    def reconstruct_surfaces(
        self,
        device: torch.device | None = None,
    ) -> None:
        """
        Reconstruct NURBS surfaces from bitmaps.

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
            log.info("Start the surface reconstruction.")

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
                    resolution=self.resolution,
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

                # Start the optimization.
                loss = torch.inf
                epoch = 0
                log_step = self.max_epoch // self.num_log
                loss_function = torch.nn.MSELoss()
                while loss > self.tolerance and epoch <= self.max_epoch:
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
                    # The points are a tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_surface_points, 4]
                    # The normals are a tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_surface_normals, 4]
                    (
                        heliostat_group.active_surface_points,
                        heliostat_group.active_surface_normals,
                    ) = nurbs_surfaces.calculate_surface_points_and_normals(
                        evaluation_points=evaluation_points, device=device
                    )

                    # The alignment module and the raytracer do not accept facetted points and normals, therefore they need to be reshaped.
                    heliostat_group.active_surface_points = (
                        heliostat_group.active_surface_points.reshape(
                            heliostat_group.active_surface_points.shape[0], -1, 4
                        )
                    )
                    heliostat_group.active_surface_normals = (
                        heliostat_group.active_surface_normals.reshape(
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
                        bitmap_resolution=self.resolution,
                    )

                    # Perform heliostat-based ray tracing.
                    flux_distributions = ray_tracer.trace_rays(
                        incident_ray_directions=incident_ray_directions,
                        active_heliostats_mask=active_heliostats_mask,
                        target_area_mask=target_area_mask,
                        device=device,
                    )

                    # Normalize the flux distributions.
                    normalized_flux_distributions = utils.normalize_bitmaps(
                        flux_distributions=flux_distributions,
                        target_area_widths=ray_tracer.scenario.target_areas.dimensions[
                            target_area_mask
                        ][:, 0],
                        target_area_heights=ray_tracer.scenario.target_areas.dimensions[
                            target_area_mask
                        ][:, 1],
                        number_of_rays=ray_tracer.light_source.number_of_rays,
                    )

                    if self.ddp_setup["is_nested"]:
                        normalized_flux_distributions = (
                            torch.distributed.nn.functional.all_reduce(
                                normalized_flux_distributions,
                                group=self.ddp_setup["process_subgroup"],
                                op=torch.distributed.ReduceOp.SUM,
                            )
                        )

                    loss = loss_function(
                        normalized_flux_distributions,
                        normalized_measured_flux_distributions,
                    )

                    loss.backward()

                    if self.ddp_setup["is_nested"]:
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

                    optimizer.step()

                    if epoch % log_step == 0 and rank == 0:
                        log.info(
                            f"Epoch: {epoch}, Loss: {loss}, LR: {optimizer.param_groups[0]['lr']}",
                        )

                    epoch += 1

                log.info("Surfaces reconstructed.")
