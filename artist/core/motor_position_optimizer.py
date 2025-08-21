import logging
from typing import Callable

import torch

from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.scenario.scenario import Scenario
from artist.util.environment_setup import DistributedEnvironmentTypedDict, get_device

log = logging.getLogger(__name__)
"""A logger for the motor positions optimizer."""


class MotorPositionsOptimizer:
    """
    An optimizer used to find optimal motor positions for the heliostats.

    Attributes
    ----------
    ddp_setup : DistributedEnvironmentTypedDict
        Information about the distributed environment, process_groups, devices, ranks, world_Size, heliostat group to ranks mapping.
    scenario : Scenario
        The scenario.
    incident_ray_direction : torch.Tensor
        The incident ray direction during the optimization.
        Tensor of shape [4].
    target_area_index : int
        The index of the target used for the optimization.
    optimization_goal : torch.Tensor
        The desired focal spot or distribution.
        Tensor of shape [4] or tensor of shape [bitmap_resolution_e, bitmap_resolution_u].
    bitmap_resolution : torch.Tensor
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
    optimize()
        Optimize the motor positions.
    """

    def __init__(
        self,
        ddp_setup: DistributedEnvironmentTypedDict,
        scenario: Scenario,
        incident_ray_direction: torch.Tensor,
        target_area_index: int,
        optimization_goal: torch.Tensor,
        bitmap_resolution: torch.Tensor = torch.tensor([256, 256]),
        initial_learning_rate: float = 0.0004,
        tolerance: float = 1e-5,
        max_epoch: int = 600,
        num_log: int = 3,
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize the motor positions optimizer.

        Parameters
        ----------
        ddp_setup : DistributedEnvironmentTypedDict
            Information about the distributed environment, process_groups, devices, ranks, world_Size, heliostat group to ranks mapping.
        scenario : Scenario
            The scenario.
        incident_ray_direction : torch.Tensor
            The incident ray direction during the optimization.
            Tensor of shape [4].
        target_area_index : int
            The index of the target used for the optimization.
        optimization_goal : torch.Tensor
            The desired focal spot or distribution.
            Tensor of shape [4] or tensor of shape [bitmap_resolution_e, bitmap_resolution_u].
        bitmap_resolution : torch.Tensor
            The resolution of all bitmaps during optimization (default is torch.tensor([256,256])).
            Tensor of shape [2].
        initial_learning_rate : float
            The initial learning rate for the optimizer (default is 0.0004).
        tolerance : float
            The tolerance during optimization (default is 0.035).
        max_epoch : int
            The maximum optimization epoch (default is 600).
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
            log.info("Create a motor positions optimizer.")

        self.ddp_setup = ddp_setup
        self.scenario = scenario
        self.incident_ray_direction = incident_ray_direction
        self.target_area_index = target_area_index
        self.optimization_goal = optimization_goal
        self.bitmap_resolution = bitmap_resolution.to(device)
        self.initial_learning_rate = initial_learning_rate
        self.tolerance = tolerance
        self.max_epoch = max_epoch
        self.num_log = num_log

    def optimize(
        self,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: torch.device | None = None,
    ) -> None:
        """
        Optimize the motor positions.

        Parameters
        ----------
        loss_function : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            A callable function that computes the loss. It accepts predictions and targets
            and optionally other keyword arguments and return a tensor with loss values.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        device = get_device(device)

        rank = self.ddp_setup["rank"]

        if rank == 0:
            log.info("Start the motor positions optimization.")

        for heliostat_group_index in self.ddp_setup["groups_to_ranks_mapping"][rank]:
            heliostat_group = self.scenario.heliostat_field.heliostat_groups[
                heliostat_group_index
            ]

            # Align all heliostats once, to the given incident ray direction and target, to set initial
            # motor positions. The motor positions are set automatically within the align from incident ray method.
            (
                active_heliostats_mask,
                target_area_mask,
                incident_ray_directions,
            ) = self.scenario.index_mapping(
                heliostat_group=heliostat_group,
                single_incident_ray_direction=self.incident_ray_direction,
                single_target_area_index=self.target_area_index,
                device=device,
            )

            # Activate heliostats.
            heliostat_group.activate_heliostats(
                active_heliostats_mask=active_heliostats_mask, device=device
            )

            # Align heliostats.
            heliostat_group.align_surfaces_with_incident_ray_directions(
                aim_points=self.scenario.target_areas.centers[target_area_mask],
                incident_ray_directions=incident_ray_directions,
                active_heliostats_mask=active_heliostats_mask,
                device=device,
            )

            # The motor positions are optimized through a reparameterization to ensure stable training 
            # across different heliostats with widely varying initial motor positions and ranges. Motor
            # positions can range from 0 to up to ~80000. Instead of directly optimizing the absolute 
            # motor positions, which can differ in magnitudes, an unconstrained parameter is optimized.
            # Directly optimizing the absolute motor positions, would have very different effects depending 
            # on the scale of the motors. For small initial motor positions (e.g. ~100), a gradient update 
            # of size 10 may cause a ~10% relative change, drastically altering the motor positions of this
            # heliostat. For large initial motor positions (e.g. ~50,000), the same optimizer step would 
            # correspond to only a 0.02% relative change in motor positions, effectively freezing the 
            # optimization of this heliostat. This mismatch makes it impossible to choose a single learning 
            # rate that works robustly across all heliostats.
            # The reparametrization of the optimizable parameter (motor positions) defines the optimizable 
            # parameter as: 
            # motor_positions_optimized = tanh(torch.nn.Parameter(optimizable_parameter))
            # The true motor positions can be reconstructed by:
            # motor_positions = initial_motor_positions + motor_positions_normalized * scale
            # where scale defines the range (e.g. up to ~80,000) for adjustments.
            # By optimizing as explained above instead of raw motor positions, every heliostat sees updates 
            # of comparable relative magnitude, regardless of the absolute size of its motors positions.
            initial_motor_positions = heliostat_group.kinematic.active_motor_positions.detach().clone()
            motor_positions_minimum = heliostat_group.kinematic.actuators.actuator_parameters[:, 2]
            motor_positions_maximum = heliostat_group.kinematic.actuators.actuator_parameters[:, 3]
            lower_margin = initial_motor_positions - motor_positions_minimum
            upper_margin = motor_positions_maximum - initial_motor_positions
            scale = torch.minimum(lower_margin, upper_margin).clamp(min=1.0)

            # Create the optimizer.
            optimizable_parameter = torch.nn.Parameter(torch.zeros_like(initial_motor_positions, device=device))
            optimizer = torch.optim.Adam([optimizable_parameter], lr=self.initial_learning_rate)

            # Start the optimization.
            loss = torch.inf
            epoch = 0
            log_step = self.max_epoch // self.num_log
            while loss > self.tolerance and epoch <= self.max_epoch:
                optimizer.zero_grad()

                # Reconstruct true motor positions from reparameterized version.
                motor_positions_normalized = torch.tanh(optimizable_parameter)
                heliostat_group.kinematic.motor_positions = initial_motor_positions + motor_positions_normalized * scale
     
                # Activate heliostats.
                heliostat_group.activate_heliostats(
                    active_heliostats_mask=active_heliostats_mask, device=device
                )

                # Align heliostats.
                heliostat_group.align_surfaces_with_motor_positions(
                    motor_positions=heliostat_group.kinematic.active_motor_positions,
                    active_heliostats_mask=active_heliostats_mask,
                    device=device,
                )

                # Create a ray tracer.
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

                if self.ddp_setup["is_nested"]:
                    flux_distributions = torch.distributed.nn.functional.all_reduce(
                        flux_distributions,
                        group=self.ddp_setup["process_subgroup"],
                        op=torch.distributed.ReduceOp.SUM,
                    )

                flux_distribution_on_target = ray_tracer.get_bitmaps_per_target(
                    bitmaps_per_heliostat=flux_distributions,
                    target_area_mask=target_area_mask,
                    device=device,
                )[self.target_area_index]

                loss = loss_function(
                    predictions=flux_distribution_on_target.unsqueeze(0),
                    targets=self.optimization_goal.unsqueeze(0),
                    scenario=self.scenario,
                    target_area_index=self.target_area_index,
                    device=device
                )

                loss.backward()

                if self.ddp_setup["is_nested"]:
                    # Reduce gradients within each heliostat group.
                    for param_group in optimizer.param_groups:
                        for param in param_group["params"]:
                            if param.grad is not None:
                                param.grad = torch.distributed.nn.functional.all_reduce(
                                    param.grad,
                                    op=torch.distributed.ReduceOp.SUM,
                                    group=self.ddp_setup["process_subgroup"],
                                )
                
                optimizer.step()

                if epoch % log_step == 0 and rank == 0:
                    log.info(
                        f"Epoch: {epoch}, Loss: {loss.item()}, LR: {optimizer.param_groups[0]['lr']}",
                    )

                epoch += 1

            log.info(f"Rank: {rank}, motor positions optimized.")

        if self.ddp_setup["is_distributed"]:
            for index, heliostat_group in enumerate(self.scenario.heliostat_field.heliostat_groups):
                source = self.ddp_setup['ranks_to_groups_mapping'][index]
                torch.distributed.broadcast(heliostat_group.kinematic.motor_positions, src=source[0])

            log.info(f"Rank: {rank}, synchronised after motor positions optimization.")
