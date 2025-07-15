import logging

import torch
from torch.optim import Optimizer

from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.field.heliostat_group import HeliostatGroup
from artist.scenario.scenario import Scenario
from artist.util import utils
from artist.util.environment_setup import get_device
from artist.util.nurbs import NURBSSurfaces

log = logging.getLogger(__name__)
"""A logger for the surface reconstructor."""


class SurfaceReconstructor:
    """
    An optimizer used to reconstruct surfaces using NURBS.

    The surface reconstructor learns a surface representation from measured flux density 
    distributions. The optimizable parameters for this optimization process are the
    NURBS control points.

    Attributes
    ----------
    scenario : Scenario
        The scenario.
    heliostat_group : HeliostatGroup
        The heliostat group to be calibrated.

    Methods
    -------
    reconstruct_surfaces()
        Reconstruct NURBS surfaces from bitmaps.
    """

    def __init__(
        self,
        scenario: Scenario,
        heliostat_group: HeliostatGroup,
    ) -> None:
        """
        Initialize the surface reconstructor.

        Parameters
        ----------
        scenario : Scenario
            The scenario.
        heliostat_group : HeliostatGroup
            The heliostat group to be reconstructed.
        """
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        if rank == 0:
            log.info("Create a kinematic optimizer.")
        self.scenario = scenario
        self.heliostat_group = heliostat_group

    def reconstruct_surfaces(
        self,
        flux_distributions_measured: torch.Tensor,
        number_of_evaluation_points: torch.Tensor,
        incident_ray_directions: torch.Tensor,
        active_heliostats_mask: torch.Tensor,
        target_area_mask: torch.Tensor,
        initial_learning_rate: float,
        tolerance: float = 5e-5,
        max_epoch: int = 10000,
        num_log: int = 3,
        device: torch.device | None = None,
    ) -> None:
        """
        Reconstruct NURBS surfaces from bitmaps.

        Parameters
        ----------
        flux_distributions_measured : torch.Tensor
            The measured flux density distributions per heliostat.
        incident_ray_directions : torch.Tensor
            The incident ray directions specified in the calibration data.
        active_heliostats_mask : torch.Tensor
            A mask where 0 indicates a deactivated heliostat and 1 an activated one.
            An integer greater than 1 indicates that this heliostat is regarded multiple times.
        target_area_mask : torch.Tensor
            The indices of the target area for each calibration. 
        tolerance : float
            The optimizer tolerance.
        max_epoch : int
            The maximum number of optimization epochs.
        num_log : int
            Number of log messages during training (default is 3).
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

        loss = torch.inf
        epoch = 0

        # Activate heliostats.
        self.heliostat_group.activate_heliostats(active_heliostats_mask=active_heliostats_mask, device=device)

        nurbs_surfaces = NURBSSurfaces(
            degrees=self.heliostat_group.nurbs_degrees,
            control_points=self.heliostat_group.active_nurbs_control_points,
            device=device
        )

        evaluation_points = utils.create_nurbs_evaluation_grid(
            number_of_evaluation_points=number_of_evaluation_points,
            device=device
        ).unsqueeze(0).unsqueeze(0).expand(self.heliostat_group.number_of_active_heliostats, 4, -1, -1)
        
        optimizer = torch.optim.Adam(
            nurbs_surfaces.parameters(), lr=initial_learning_rate
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        
        # Start the optimization.
        log_step = max_epoch // num_log
        while loss > tolerance and epoch <= max_epoch:
            optimizer.zero_grad()

            (
                self.heliostat_group.active_surface_points, 
                self.heliostat_group.active_surface_normals
            ) = nurbs_surfaces.calculate_surface_points_and_normals(
                evaluation_points=evaluation_points, 
                device=device
            )

            self.heliostat_group.active_surface_points = self.heliostat_group.active_surface_points.reshape(self.heliostat_group.active_surface_points.shape[0], -1, 4)
            self.heliostat_group.active_surface_normals = self.heliostat_group.active_surface_normals.reshape(self.heliostat_group.active_surface_normals.shape[0], -1, 4)

            # Align heliostats.
            self.heliostat_group.align_surfaces_with_incident_ray_directions(
                aim_points=self.scenario.target_areas.centers[
                    target_area_mask
                ],
                incident_ray_directions=incident_ray_directions,
                active_heliostats_mask=active_heliostats_mask,
                device=device,
            )

            # Create a ray tracer.
            ray_tracer = HeliostatRayTracer(
                scenario=self.scenario,
                heliostat_group=self.heliostat_group,
                batch_size=self.heliostat_group.number_of_active_heliostats,
            )

            # Perform heliostat-based ray tracing.
            flux_distributions = ray_tracer.trace_rays(
                incident_ray_directions=incident_ray_directions,
                active_heliostats_mask=active_heliostats_mask,
                target_area_mask=target_area_mask,
                device=device,
            )

            loss = (flux_distributions - flux_distributions_measured).abs().mean()
            loss.backward()
            total_grad = sum(p.grad.abs().sum() for p in nurbs_surfaces.parameters() if p.grad is not None)
            print(f"Total grad sum: {total_grad.item()}")
            for name, param in nurbs_surfaces.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"{name} has NaNs or Infs!")

            optimizer.step()
            scheduler.step()

            if epoch % log_step == 0 and rank == 0:
                log.info(
                    f"Epoch: {epoch}, Loss: {loss}, LR: {optimizer.param_groups[0]['lr']}",
                )

            epoch += 1

        if rank == 0:
            log.info("Surfaces reconstructed.")
        
