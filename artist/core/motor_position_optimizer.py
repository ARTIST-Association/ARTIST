import logging

import torch

from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, utils
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
    method : str
        The method used for optimization. The motor positions can be optimized to aim at a
        specific coordinate or to match a specific distribution.
    optimization_goal : torch.Tensor
        The desired focal spot or distribution.
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
        method: str,
        optimization_goal: torch.Tensor,
        bitmap_resolution: torch.Tensor = torch.tensor([256, 256]),
        initial_learning_rate: float = 0.0004,
        tolerance: float = 0.035,
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
        method : str
            The method used for optimization. The motor positions can be optimized to aim at a
            specific coordinate or to match a specific distribution.
        optimization_goal : torch.Tensor
            The desired focal spot or distribution.
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
        self.method = method
        self.optimization_goal = optimization_goal
        self.bitmap_resolution = bitmap_resolution.to(device)
        self.initial_learning_rate = initial_learning_rate
        self.tolerance = tolerance
        self.max_epoch = max_epoch
        self.num_log = num_log

    def optimize(
        self,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Optimize the motor positions.

        Parameters
        ----------
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

            heliostat_group.kinematic.motor_positions = (
                heliostat_group.kinematic.active_motor_positions.clone().detach()
            )

            optimizer = torch.optim.Adam(
                [heliostat_group.kinematic.motor_positions.requires_grad_()],
                lr=self.initial_learning_rate,
            )

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

                # Align heliostats.
                heliostat_group.align_surfaces_with_motor_positions(
                    motor_positions=heliostat_group.kinematic.motor_positions,
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
                bitmaps_per_heliostat = ray_tracer.trace_rays(
                    incident_ray_directions=incident_ray_directions,
                    active_heliostats_mask=active_heliostats_mask,
                    target_area_mask=target_area_mask,
                    device=device,
                )

                flux_distribution_on_target = ray_tracer.get_bitmaps_per_target(
                    bitmaps_per_heliostat=bitmaps_per_heliostat,
                    target_area_mask=target_area_mask,
                    device=device,
                )[self.target_area_index]

                if self.ddp_setup["is_nested"]:
                    torch.distributed.all_reduce(
                        flux_distribution_on_target,
                        op=torch.distributed.ReduceOp.SUM,
                        group=self.ddp_setup["process_subgroup"],
                    )

                if self.method == config_dictionary.optimization_to_focal_spot:
                    # Determine the focal spots of all flux density distributions
                    focal_spot = utils.get_center_of_mass(
                        bitmaps=flux_distribution_on_target.unsqueeze(0),
                        target_centers=self.scenario.target_areas.centers[
                            self.target_area_index
                        ],
                        target_widths=self.scenario.target_areas.dimensions[
                            self.target_area_index
                        ][0],
                        target_heights=self.scenario.target_areas.dimensions[
                            self.target_area_index
                        ][1],
                        device=device,
                    )
                    
                    loss = loss_function(
                        focal_spot,
                        self.optimization_goal,
                    )


                if self.method == config_dictionary.optimization_to_distribution:
                    target_distribution = (self.optimization_goal / (self.optimization_goal.sum() + 1e-12))
                    flux_shifted = flux_distribution_on_target - flux_distribution_on_target.min()
                    current_distribution = flux_shifted / flux_shifted.sum()
                    log_target_distribution = torch.log(target_distribution + 1e-12)
                    log_current_distribution = torch.log(current_distribution + 1e-12)
                    
                    loss = torch.nn.functional.kl_div(input=log_current_distribution, target=log_target_distribution, reduction='sum', log_target=True)
                    # loss = torch.nn.functional.kl_div(log_current_distribution, target_distribution, reduction='sum', log_target=False)
                    loss = utils.kl_divergence(p=target_distribution, q=current_distribution)
                    # P represents the data/measured distribution (P=target)
                    # Q represents an approximation of P (Q=prediction)

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

                optimizer.step()

                if epoch % log_step == 0 and rank == 0:
                    log.info(
                        f"Epoch: {epoch}, Loss: {loss}, LR: {optimizer.param_groups[0]['lr']}",
                    )

                epoch += 1

            log.info(f"Rank: {rank}, motor positions optimized.")
            import matplotlib.pyplot as plt

            plt.imshow(flux_distribution_on_target.cpu().detach(), cmap="gray")
            plt.savefig(f"result_{heliostat_group_index}.png")

        if self.ddp_setup["is_distributed"]:
            for heliostat_group in self.scenario.heliostat_field.heliostat_groups:
                torch.distributed.nn.functional.all_reduce(
                    heliostat_group.kinematic.motor_positions,
                    op=torch.distributed.ReduceOp.SUM,
                )

    @staticmethod
    def trapezoid_1d(
        total_width: torch.Tensor,
        slope_width: torch.Tensor,
        plateau_width: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Create a one dimensional trapezoid distribution.

        Parameter
        ---------
        total_width : torch.Tensor
            The total width of the trapezoid.
        slope_width : torch.Tensor
            The width of the slope of the trapezoid.
        plateau_width : torch.Tensor
            The width of the plateau.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The one dimensional trapezoid distribution.
        """
        device = get_device(device)

        indices = torch.arange(total_width, device=device)
        center = (total_width - 1) / 2
        half_plateau = plateau_width / 2

        # Distances from the plateau edge.
        distances = torch.abs(indices - center) - half_plateau

        trapezoid = 1 - (distances / slope_width).clamp(min=0, max=1)

        return trapezoid
