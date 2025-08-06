import logging
import pathlib

import torch

from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.data_loader import flux_distribution_loader, paint_loader
from artist.field.heliostat_group import HeliostatGroup
from artist.scenario.scenario import Scenario
from artist.util import utils
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
    scenario : Scenario
        The scenario.
    heliostat_group : HeliostatGroup
        The heliostat group to be reconstructed.
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
    num_log : int
        The number of log statements during optimization.
    initial_learning_rate : float
        The initial learning rate for the optimizer (default is 0.0004).
    tolerance : float
        The optimizer tolerance.
    max_epoch : int
        The maximum number of optimization epochs.

    Methods
    -------
    reconstruct_surfaces()
        Reconstruct NURBS surfaces from bitmaps.
    """

    def __init__(
        self,
        scenario: Scenario,
        heliostat_group: HeliostatGroup,
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
        scenario : Scenario
            The scenario.
        heliostat_group : HeliostatGroup
            The heliostat group to be reconstructed.
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

        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        if rank == 0:
            log.info("Create a surface reconstructor.")

        self.scenario = scenario
        self.heliostat_group = heliostat_group

        # Extract measured fluxes.
        heliostat_flux_path_mapping = [
            (heliostat_name, png_paths)
            for heliostat_name, _, png_paths in heliostat_data_mapping
            if heliostat_name in self.heliostat_group.names
        ]

        self.normalized_measured_flux_distributions = (
            flux_distribution_loader.load_flux_from_png(
                heliostat_flux_path_mapping=heliostat_flux_path_mapping,
                heliostat_names=heliostat_group.names,
                resolution=resolution,
                device=device,
            )
        )

        # Extract environmental data for measured fluxes.
        heliostat_calibration_mapping = [
            (heliostat_name, calibration_properties_paths)
            for heliostat_name, calibration_properties_paths, _ in heliostat_data_mapping
            if heliostat_name in self.heliostat_group.names
        ]
        (
            _,
            self.incident_ray_directions,
            _,
            self.heliostats_mask,
            self.target_area_mask,
        ) = paint_loader.extract_paint_calibration_properties_data(
            heliostat_calibration_mapping=heliostat_calibration_mapping,
            heliostat_names=heliostat_group.names,
            target_area_names=scenario.target_areas.names,
            power_plant_position=scenario.power_plant_position,
            device=device,
        )

        self.number_of_surface_points = number_of_surface_points.to(device)
        self.resolution = resolution.to(device)
        self.num_log = num_log

        # Create the optimizer.
        self.initial_learning_rate = initial_learning_rate
        self.tolerance = tolerance
        self.max_epoch = max_epoch

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

        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        if rank == 0:
            log.info("Start the surface reconstruction.")

        if self.heliostats_mask.sum() > 0:
            evaluation_points = (
                utils.create_nurbs_evaluation_grid(
                    number_of_evaluation_points=self.number_of_surface_points,
                    device=device,
                )
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(
                    self.heliostats_mask.sum(),
                    self.heliostat_group.number_of_facets_per_heliostat,
                    -1,
                    -1,
                )
            )

            # Activate heliostats.
            self.heliostat_group.activate_heliostats(
                active_heliostats_mask=self.heliostats_mask, device=device
            )

            optimizer = torch.optim.Adam(
                [self.heliostat_group.nurbs_control_points.requires_grad_()],
                lr=self.initial_learning_rate,
            )

            # Start the optimization.
            loss = torch.inf
            epoch = 0
            log_step = self.max_epoch // self.num_log
            while loss > self.tolerance and epoch <= self.max_epoch:
                optimizer.zero_grad()

                # Activate heliostats.
                self.heliostat_group.activate_heliostats(
                    active_heliostats_mask=self.heliostats_mask, device=device
                )

                # Create NURBS
                nurbs_surfaces = NURBSSurfaces(
                    degrees=self.heliostat_group.nurbs_degrees,
                    control_points=self.heliostat_group.active_nurbs_control_points,
                    device=device,
                )

                # Calculate surface points and normals
                (
                    self.heliostat_group.active_surface_points,
                    self.heliostat_group.active_surface_normals,
                ) = nurbs_surfaces.calculate_surface_points_and_normals(
                    evaluation_points=evaluation_points, device=device
                )

                self.heliostat_group.active_surface_points = (
                    self.heliostat_group.active_surface_points.reshape(
                        self.heliostat_group.active_surface_points.shape[0], -1, 4
                    )
                )
                self.heliostat_group.active_surface_normals = (
                    self.heliostat_group.active_surface_normals.reshape(
                        self.heliostat_group.active_surface_normals.shape[0], -1, 4
                    )
                )

                # Align heliostats.
                self.heliostat_group.align_surfaces_with_incident_ray_directions(
                    aim_points=self.scenario.target_areas.centers[
                        self.target_area_mask
                    ],
                    incident_ray_directions=self.incident_ray_directions,
                    active_heliostats_mask=self.heliostats_mask,
                    device=device,
                )

                # Create a ray tracer.
                ray_tracer = HeliostatRayTracer(
                    scenario=self.scenario,
                    heliostat_group=self.heliostat_group,
                    batch_size=self.heliostat_group.number_of_active_heliostats,
                    bitmap_resolution=self.resolution,
                )

                # Perform heliostat-based ray tracing.
                flux_distributions = ray_tracer.trace_rays(
                    incident_ray_directions=self.incident_ray_directions,
                    active_heliostats_mask=self.heliostats_mask,
                    target_area_mask=self.target_area_mask,
                    device=device,
                )

                normalized_flux_distributions = utils.normalize_bitmaps(
                    flux_distributions=flux_distributions,
                    target_area_widths=ray_tracer.scenario.target_areas.dimensions[
                        self.target_area_mask
                    ][:, 0],
                    target_area_heights=ray_tracer.scenario.target_areas.dimensions[
                        self.target_area_mask
                    ][:, 1],
                    number_of_rays=ray_tracer.light_source.number_of_rays,
                    device=device
                )

                loss_function = torch.nn.MSELoss()
                loss = loss_function(
                    normalized_flux_distributions,
                    self.normalized_measured_flux_distributions,
                )

                loss.backward()

                optimizer.step()

                if epoch % log_step == 0 and rank == 0:
                    log.info(
                        f"Epoch: {epoch}, Loss: {loss}, LR: {optimizer.param_groups[0]['lr']}",
                    )

                epoch += 1

            log.info("Surfaces reconstructed.")
