import logging
from typing import Literal

import torch
from torch.optim import Optimizer

from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.field.heliostat_group import HeliostatGroup
from artist.scenario.scenario import Scenario
from artist.util import raytracing_utils, utils
from artist.util.environment_setup import get_device

log = logging.getLogger(__name__)
"""A logger for the kinematic optimizer."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AngleLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(AngleLoss, self).__init__()
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.reduction = reduction

    def forward(self, input, target):

        input = input[:,:3]
        target = target[:,:3]
        # Ensure input and target have the same shape
        if input.shape != target.shape:
            raise ValueError("Input and target must have the same shape")

        # Normalize the input and target to unit vectors
        input_norm = F.normalize(input, p=2, dim=-1)
        target_norm = F.normalize(target, p=2, dim=-1)

        # Compute cosine similarity
        cos_sim = (input_norm * target_norm).sum(dim=-1).clamp(-1.0, 1.0)  # clamp for numerical stability

        # Compute angle in radians
        angle = torch.acos(cos_sim)

        if self.reduction == 'none':
            return angle
        elif self.reduction == 'mean':
            return angle.mean()
        elif self.reduction == 'sum':
            return angle.sum()




class KinematicOptimizer:
    """
    An optimizer used to find optimal kinematic parameters.

    The kinematic optimizer optimizes kinematic parameters.
    These parameters are specific to a certain kinematic type
    and can for example include the 18 kinematic deviations parameters as well as five actuator
    parameters for each actuator for a rigid body kinematic.

    Attributes
    ----------
    scenario : Scenario
        The scenario.
    heliostat_group : HeliostatGroup
        The heliostat group to be calibrated.
    optimizer : Optimizer
        The optimizer.

    Methods
    -------
    optimize()
        Optimize the kinematic parameters.
    """

    def __init__(
        self,
        scenario: Scenario,
        heliostat_group: HeliostatGroup,
        optimizer: Optimizer,
    ) -> None:
        """
        Initialize the kinematic optimizer.

        Parameters
        ----------
        scenario : Scenario
            The scenario.
        heliostat_group : HeliostatGroup
            The heliostat group to be calibrated.
        optimizer : Optimizer
            The optimizer.
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
        self.optimizer = optimizer

    def optimize(
        self,
        focal_spots_calibration: torch.Tensor,
        incident_ray_directions: torch.Tensor,
        active_heliostats_mask: torch.Tensor,
        target_area_mask_calibration: torch.Tensor | None = None,
        motor_positions_calibration: torch.Tensor | None = None,
        tolerance: float = 5e-5,
        max_epoch: int = 10000,
        num_log: int = 3,
        loss_type: str = "l1",
        loss_reduction: str = "mean",
        loss_return_value: str = "mean",
        device: torch.device | None = None,
    ) -> None:
        """
        Optimize the kinematic parameters.

        Parameters
        ----------
        focal_spots_calibration : torch.Tensor
            The center coordinates of the calibration flux densities.
        incident_ray_directions : torch.Tensor
            The incident ray directions specified in the calibrations.
        active_heliostats_mask : torch.Tensor
            A mask for the selected heliostats for calibration.
        target_area_mask_calibration : torch.Tensor | None
            The indices of the target area for each calibration (default is None).
        motor_positions_calibration : torch.Tensor | None
            The motor positions specified in the calibration files (default is None).
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
            log.info("Start the kinematic calibration.")

        if motor_positions_calibration is not None:
            losses = self._optimize_kinematic_parameters_with_motor_positions(
                focal_spots_calibration=focal_spots_calibration,
                incident_ray_directions=incident_ray_directions,
                active_heliostats_mask=active_heliostats_mask,
                motor_positions_calibration=motor_positions_calibration,
                tolerance=tolerance,
                max_epoch=max_epoch,
                num_log=num_log,
                loss_type = loss_type,
                loss_reduction = loss_reduction,
                loss_return_value = loss_return_value,
                device=device,
            )

        else:
            losses = self._optimize_kinematic_parameters_with_raytracing(
                focal_spots_calibration=focal_spots_calibration,
                incident_ray_directions=incident_ray_directions,
                active_heliostats_mask=active_heliostats_mask,
                target_area_mask_calibration=target_area_mask_calibration,
                tolerance=tolerance,
                max_epoch=max_epoch,
                num_log=num_log,
                loss_type = loss_type,
                reduction = loss_reduction,
                device=device,
            )
        if rank == 0:
            log.info("Kinematic parameters optimized.")
        return losses
        

    def _optimize_kinematic_parameters_with_motor_positions(
        self,
        focal_spots_calibration: torch.Tensor,
        incident_ray_directions: torch.Tensor,
        active_heliostats_mask: torch.Tensor,
        motor_positions_calibration: torch.Tensor,
        tolerance: float,
        max_epoch: int,
        num_log: int = 3,
        loss_type: Literal["l1", "l2", "angular"] = "l1",
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        loss_return_value: Literal["l1", "l2", "angular"] = "l1",
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Optimize the kinematic parameters using the motor positions.

        This optimizer method optimizes the kinematic parameters by extracting the motor positions
        and incident ray directions from calibration data and using the scene's geometry.

        Parameters
        ----------
        focal_spots_calibration : torch.Tensor
            The center coordinates of the calibration flux densities.
        incident_ray_directions : torch.Tensor
            The incident ray directions specified in the calibration data.
        active_heliostats_mask : torch.Tensor
            A mask where 0 indicates a deactivated heliostat and 1 an activated one.
            An integer greater than 1 indicates that this heliostat is regarded multiple times.
        motor_positions_calibration : torch.Tensor
            The motor positions specified in the calibration data.
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
            log.info("Kinematic calibration with motor positions.")

        loss = torch.inf
        epoch = 0

        preferred_reflection_directions_calibration = torch.nn.functional.normalize(
            (
                focal_spots_calibration
                - self.heliostat_group.positions.repeat_interleave(
                    active_heliostats_mask, dim=0
                )
            ),
            p=2,
            dim=1,
        )

        log_step = max_epoch // num_log

        loss_fns = {
            "l1": torch.nn.L1Loss(reduction="none"),
            "l2": torch.nn.MSELoss(reduction="none"),
            "angular": AngleLoss(reduction="none"),
        }

        if loss_type not in loss_fns or loss_return_value not in loss_fns:
            raise ValueError(f"Unsupported loss_type or loss_return_value")

        loss_fn = loss_fns[loss_type]
        return_loss_fn = loss_fns[loss_return_value]

        last_valid_state = {}  # To hold tensors right before NaN happens
        previous_loss_value = None
        torch.autograd.set_detect_anomaly(False)
        while loss > tolerance and epoch <= max_epoch: #TODO Loss überschreiben
            self.optimizer.zero_grad()

            self.heliostat_group.activate_heliostats(
                active_heliostats_mask=active_heliostats_mask, device=device
            )

            orientations = self.heliostat_group.kinematic.motor_positions_to_orientations(
                motor_positions=motor_positions_calibration,
                device=device,
            )

            preferred_reflection_directions = raytracing_utils.reflect(
                incident_ray_directions=incident_ray_directions,
                reflection_surface_normals=orientations[:, 0:4, 2],
            )

            unreduced_loss = loss_fn(
                preferred_reflection_directions,
                preferred_reflection_directions_calibration
            )

            # Reduce loss
            if loss_reduction == "mean":
                loss_value = unreduced_loss.mean()
            elif loss_reduction == "sum":
                loss_value = unreduced_loss.sum()
            elif loss_reduction == "none":
                loss_value = unreduced_loss.mean()
            else:
                raise ValueError(f"Unsupported reduction: {loss_reduction}")

            last_valid_state = {
                "epoch": epoch,
                "loss_value": loss_value.detach().cpu(),
                "orientations": orientations.detach().cpu(),
                "preferred_reflection_directions": preferred_reflection_directions.detach().cpu(),
            }
            previous_loss_value = loss_value.item()

            loss_value.backward()
            self.optimizer.step()


            return_loss_unreduced = return_loss_fn(
            preferred_reflection_directions,
            preferred_reflection_directions_calibration)

            if epoch % log_step == 0 and rank == 0:
                log.info(
                    f"Epoch: {epoch}, Loss: {loss_value.item()}, Loss(rad): {return_loss_unreduced.mean().item()}, LR: {self.optimizer.param_groups[0]['lr']}",
                )
            epoch += 1




        return return_loss_unreduced

    def _optimize_kinematic_parameters_with_raytracing(
        self,
        focal_spots_calibration: torch.Tensor,
        incident_ray_directions: torch.Tensor,
        active_heliostats_mask: torch.Tensor,
        target_area_mask_calibration: torch.Tensor,
        tolerance: float,
        max_epoch: int,
        num_log: int = 3,
        loss_type: Literal["l1", "l2"] = "l1",
        reduction: Literal["mean", "sum", "none"] = "mean",
        device: torch.device | None = None,
    ) -> None:
        """
        Optimize the kinematic parameters using ray tracing.

        This optimizer method optimizes the kinematic parameters by extracting the focus points
        of calibration images and using heliostat-tracing.

        Parameters
        ----------
        focal_spots_calibration : torch.Tensor
            The center coordinates of the calibration flux densities.
        incident_ray_directions : torch.Tensor
            The incident ray directions specified in the calibration data.
        active_heliostats_mask : torch.Tensor
            A mask where 0 indicates a deactivated heliostat and 1 an activated one.
            An integer greater than 1 indicates that this heliostat is regarded multiple times.
        target_area_mask_calibration : torch.Tensor
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
            log.info("Kinematic optimization with ray tracing.")

        loss = torch.inf
        epoch = 0

        # Start the optimization.
        log_step = max_epoch // num_log

        if loss_type == "l1":
            loss_fn = torch.nn.L1Loss(reduction="none")
        elif loss_type == "l2":
            loss_fn = torch.nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        final_loss_tensor = None  # for returning after optimization

        while loss > tolerance and epoch <= max_epoch:
            self.optimizer.zero_grad()

            # Align heliostats.
            self.heliostat_group.align_surfaces_with_incident_ray_directions(
                aim_points=self.scenario.target_areas.centers[
                    target_area_mask_calibration
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
                target_area_mask=target_area_mask_calibration,
                device=device,
            )

            # Determine the focal spots of all flux density distributions
            focal_spots = utils.get_center_of_mass(
                bitmaps=flux_distributions,
                target_centers=self.scenario.target_areas.centers[
                    target_area_mask_calibration
                ],
                target_widths=self.scenario.target_areas.dimensions[
                    target_area_mask_calibration
                ][:, 0],
                target_heights=self.scenario.target_areas.dimensions[
                    target_area_mask_calibration
                ][:, 1],
                device=device,
            )

            unreduced_loss = loss_fn(
                        focal_spots,
                        focal_spots_calibration)

             # Calculate scalar loss for backward pass
            if reduction == "mean":
                loss_value = unreduced_loss.mean()
            elif reduction == "sum":
                loss_value = unreduced_loss.sum()
            elif reduction == "none":
                loss_value = unreduced_loss.mean()
            else:
                raise ValueError(f"Unsupported reduction: {reduction}")
            loss_value.backward()

            self.optimizer.step()

            if epoch % log_step == 0 and rank == 0:
                log.info(
                    f"Epoch: {epoch}, Loss: {loss_value}, LR: {self.optimizer.param_groups[0]['lr']}",
                )
            epoch += 1

        final_loss_tensor = unreduced_loss if reduction == "none" else loss_value
        return final_loss_tensor
