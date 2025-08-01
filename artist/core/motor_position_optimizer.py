import logging

import torch

from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.scenario.scenario import Scenario
from artist.util.environment_setup import get_device

log = logging.getLogger(__name__)
"""A logger for the motor positions optimizer."""

class MotorPositionsOptimizer:
    """
    An optimizer used to find optimal motor positions for the heliostats.

    Attributes
    ----------
    scenario : Scenario
        The scenario.
    num_log : int
        The number of log statements during optimization.
    initial_learning_rate : float
        The initial learning rate for the optimizer (default is 0.0004).
    tolerance : float
        The optimizer tolerance.
    max_epoch : int
        The maximum number of optimization epochs.
    optimizer : Optimizer
        The optimizer.

    Methods
    -------
    optimize()
        Optimize the motor positions.
    """
    
    def __init__(
        self,
        scenario: Scenario,
        initial_learning_rate: float = 0.0004,
        tolerance: float = 0.035,
        max_epoch: int = 600,
        num_log: int = 3,
    ) -> None:
        """
        Initialize the motor positions optimizer.

        Parameters
        ----------
        scenario : Scenario
            The scenario.
        initial_learning_rate : float
            The initial learning rate for the optimizer (default is 0.0004).
        tolerance : float
            The tolerance during optimization (default is 0.035).
        max_epoch : int
            The maximum optimization epoch (default is 600).
        num_log : int
            The number of log statements during optimization (default is 3).
        """
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        if rank == 0:
            log.info("Create a motor positions optimizer.")


        self.scenario=scenario
        
        self.num_log = num_log

        self.initial_learning_rate = initial_learning_rate
        self.tolerance = tolerance
        self.max_epoch = max_epoch


    def optimze(
        self,
        ddp_setup: tuple[torch.device, bool, bool, int, int, torch.distributed.ProcessGroup | None, dict[int, list[int]], int, int],
    ) -> torch.Tensor:
        """
        Optimize the motor positions.

        Parameters
        ----------
        ddp_set_up : tuple[torch.device, bool, bool, int, int, torch.distributed.ProcessGroup | None, dict[int, list[int]], int, int],
        """
        device = get_device(device=ddp_setup["device"])

        rank = ddp_setup["rank"]
        if rank == 0:
            log.info("Start the motor positions optimization.")


        # get motor positions or can they be zero in the beginning?
        
        optimizable_parameters = [group.active_motor_positions for group in self.scenario.heliostat_field.heliostat_groups]

        self.optimizer = torch.optim.Adam(
            optimizable_parameters,
            lr=self.initial_learning_rate,
        )

        loss = torch.inf
        epoch = 0
        log_step = self.max_epoch // self.num_log
        while loss > self.tolerance and epoch <= self.max_epoch:
            self.optimizer.zero_grad()

            bitmap_resolution = torch.tensor([256, 256])

            combined_bitmaps_per_target = torch.zeros(
                (
                    self.scenario.target_areas.number_of_target_areas,
                    bitmap_resolution[0],
                    bitmap_resolution[1],
                ),
                device=device,
            )

            for group_index in ddp_setup["groups_to_ranks_mapping"][rank]:
                heliostat_group = self.scenario.heliostat_field.heliostat_groups[group_index]

                # If no mapping from heliostats to target areas to incident ray direction is provided, the scenario.index_mapping() method
                # activates all heliostats. It is possible to then provide a default target area index and a default incident ray direction
                # if those are not specified either all heliostats are assigned to the first target area found in the scenario with an
                # incident ray direction "north" (meaning the light source position is directly in the south) for all heliostats.
                (
                    active_heliostats_mask,
                    target_area_mask,
                    incident_ray_directions,
                ) = self.scenario.index_mapping(
                    heliostat_group=heliostat_group,
                    single_incident_ray_direction=torch.tensor([0.0, 1.0, 0.0, 0.0], device=device), #TODO
                    single_target_area_index=1, #TODO == receiver
                    device=device,
                )

                heliostat_group.activate_heliostats(
                    active_heliostats_mask=active_heliostats_mask, device=device
                )

                # TODO align with motor positions
                # Align heliostats.
                heliostat_group.align_surfaces_with_incident_ray_directions(
                    aim_points=self.scenario.target_areas.centers[target_area_mask],
                    incident_ray_directions=incident_ray_directions,
                    active_heliostats_mask=active_heliostats_mask,
                    device=device,
                )

                # Create a ray tracer.
                ray_tracer = HeliostatRayTracer(
                    scenario=self.scenario,
                    heliostat_group=heliostat_group,
                    world_size=ddp_setup["heliostat_group_world_size"],
                    rank=ddp_setup["heliostat_group_rank"],
                    batch_size=heliostat_group.number_of_active_heliostats,
                    random_seed=ddp_setup["heliostat_group_rank"],
                    bitmap_resolution=bitmap_resolution,
                )

                # Perform heliostat-based ray tracing.
                bitmaps_per_heliostat = ray_tracer.trace_rays(
                    incident_ray_directions=incident_ray_directions,
                    active_heliostats_mask=active_heliostats_mask,
                    target_area_mask=target_area_mask,
                    device=device,
                )

                bitmaps_per_target = ray_tracer.get_bitmaps_per_target(
                    bitmaps_per_heliostat=bitmaps_per_heliostat,
                    target_area_mask=target_area_mask,
                    device=device,
                )

                combined_bitmaps_per_target = combined_bitmaps_per_target + bitmaps_per_target

            if ddp_setup["is_nested"]:
                torch.distributed.all_reduce(
                    combined_bitmaps_per_target,
                    op=torch.distributed.ReduceOp.SUM,
                    group=ddp_setup["process_subgroup"],
                )

            if ddp_setup["is_distributed"]:
                torch.distributed.all_reduce(
                    combined_bitmaps_per_target, op=torch.distributed.ReduceOp.SUM
                )


