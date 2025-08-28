import logging
import pathlib
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as functional

from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.data_loader import paint_loader
from artist.field.heliostat_group import HeliostatGroup
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, raytracing_utils, utils
from artist.util.environment_setup import get_device

log = logging.getLogger(__name__)  # Logger for the kinematic optimizer.


class AngleLoss(nn.Module):
    """Compute the angular difference (in radians) between 3D vectors.

    Parameters
    ----------
    reduction : {"none", "mean", "sum"}, optional
        Reduction applied across the batch dimension. Default is "mean".

    Raises
    ------
    ValueError
        If an invalid reduction mode is provided.
    """

    def __init__(self, reduction: Literal["none", "mean", "sum"] = "mean") -> None:
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute per-sample angular error.

        Parameters
        ----------
        input : torch.Tensor
            Predicted vectors of shape (..., 3+) where the first three channels are xyz.
        target : torch.Tensor
            Target vectors of shape (..., 3+) where the first three channels are xyz.

        Returns
        -------
        torch.Tensor
            Angle in radians, reduced according to `self.reduction`.
        """
        input = input[:, :3]
        target = target[:, :3]
        # Ensure input and target have the same shape.
        if input.shape != target.shape:
            raise ValueError("Input and target must have the same shape")

        # Normalize the input and target to unit vectors.
        input_norm = functional.normalize(input, p=2, dim=-1)
        target_norm = functional.normalize(target, p=2, dim=-1)

        # Compute cosine similarity.
        cos_sim = (input_norm * target_norm).sum(dim=-1).clamp(-1.0, 1.0)

        # Compute angle in radians.
        angle = torch.acos(cos_sim)

        if self.reduction == "none":
            return angle
        elif self.reduction == "mean":
            return angle.mean()
        elif self.reduction == "sum":
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
    calibration_method : str
        The calibration method. Either using ray tracing or motor positions (default is ray_tracing).
    focal_spots_measured : torch.Tensor
        The center coordinates of the calibration flux densities.
        Tensor of shape [number_of_calibration_data_points, 4].
    incident_ray_directions : torch.Tensor
        The incident ray directions specified in the calibrations.
        Tensor of shape [number_of_calibration_data_points, 4].
    motor_positions : torch.Tensor | None
        The motor positions specified in the calibration files or None for ray tracing.
        Tensor of shape [number_of_calibration_data_points, 2].
    heliostats_mask : torch.Tensor
        A mask for the selected heliostats for calibration.
        Tensor of shape [number_of_heliostats].
    target_area_mask : torch.Tensor
        The indices of the target area for each calibration.
        Tensor of shape [number_of_active_heliostats].
    num_log_epochs : int
        The number of log statements during optimization.
    initial_learning_rate : float
        The initial learning rate for the optimizer (default is 0.0004).
    tolerance : float
        The optimizer tolerance.
    max_epoch : int
        The maximum number of optimization epochs.
    optimizer : torch.optim.Optimizer
        The optimizer.

    Methods
    -------
    optimize()
        Optimize the kinematic parameters and return the final per-sample loss.
    """

    def __init__(
        self,
        scenario: Scenario,
        heliostat_group: HeliostatGroup,
        heliostat_data_mapping: list[
            tuple[str, list[pathlib.Path], list[pathlib.Path]]
        ],
        calibration_method: str = config_dictionary.kinematic_calibration_raytracing,
        initial_learning_rate: float = 0.0004,
        tolerance: float = 0.035,
        max_epoch: int = 600,
        num_log_epochs: int = 3,
        loss_type: Literal["l1", "l2", "angle"] = "l1",
        loss_reduction: Literal["none", "mean", "sum"] = "mean",
        loss_return_value: Literal["none", "mean", "sum"] = "mean",
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize the kinematic optimizer.

        Parameters
        ----------
        scenario : Scenario
            The scenario.
        heliostat_group : HeliostatGroup
            The heliostat group to be calibrated.
        heliostat_data_mapping : list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
            The mapping of heliostat name to lists of calibration properties paths.
        calibration_method : str
            The calibration method. Either "ray_tracing" or "motor_positions" (default is "ray_tracing").
        initial_learning_rate : float
            The initial learning rate for the optimizer (default is 0.0004).
        tolerance : float
            The tolerance during optimization (default is 0.035).
        max_epoch : int
            The maximum optimization epoch (default is 600).
        num_log_epochs : int
            The number of log statements during optimization (default is 3).
        loss_type : {"l1", "l2", "angle"}, optional
            The loss function type to use during optimization. Default is "l1".
        loss_reduction : {"none", "mean", "sum"}, optional
            The reduction mode applied across the batch. Default is "mean".
        loss_return_value : {"none", "mean", "sum"}, optional
            The reduction applied across feature dimensions for l1/l2. Default is "mean".
        device : torch.device | None, optional
            The device on which to perform computations. If None, the device is auto-selected.
        """
        device = get_device(device=device)

        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        if rank == 0:
            log.info("Create a kinematic optimizer.")

        # Validate configuration options early.
        if loss_type not in {"l1", "l2", "angle"}:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        if loss_reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unsupported loss_reduction: {loss_reduction}")
        if loss_return_value not in {"none", "mean", "sum"}:
            raise ValueError(f"Unsupported loss_return_value: {loss_return_value}")

        self.scenario = scenario
        self.heliostat_group = heliostat_group
        self.calibration_method = calibration_method

        heliostat_calibration_mapping = [
            (heliostat_name, calibration_properties_paths)
            for heliostat_name, calibration_properties_paths, _ in heliostat_data_mapping
            if heliostat_name in self.heliostat_group.names
        ]

        # Load the calibration data.
        (
            self.focal_spots_measured,
            self.incident_ray_directions,
            self.motor_positions,
            self.heliostats_mask,
            self.target_area_mask,
        ) = paint_loader.extract_paint_calibration_properties_data(
            heliostat_calibration_mapping=heliostat_calibration_mapping,
            heliostat_names=heliostat_group.names,
            target_area_names=scenario.target_areas.names,
            power_plant_position=scenario.power_plant_position,
            device=device,
        )

        if (
            self.calibration_method
            == config_dictionary.kinematic_calibration_raytracing
        ):
            self.motor_positions = None

        self.num_log = num_log_epochs

        self.loss_type = loss_type
        self.loss_reduction = loss_reduction
        self.loss_return_value = loss_return_value

        # Create the optimizer.
        self.initial_learning_rate = initial_learning_rate
        self.tolerance = tolerance
        self.max_epoch = max_epoch

        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            [
                self.heliostat_group.kinematic.deviation_parameters.requires_grad_(),
                self.heliostat_group.kinematic.actuators.actuator_parameters.requires_grad_(),
            ],
            lr=self.initial_learning_rate,
        )

    # Reusable helpers.
    def _compute_unreduced_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: Literal["direction", "point"],
    ) -> torch.Tensor:
        """
        Compute per-sample loss (unreduced) based on configuration.

        Parameters
        ----------
        prediction : torch.Tensor
            Predicted values.
        target : torch.Tensor
            Target values.
        context : {"direction", "point"}
            Determines allowed loss types and feature handling.

        Returns
        -------
        torch.Tensor
            Per-sample loss values (shape: [batch]).
        """
        if context not in ("direction", "point"):
            raise ValueError(f"Unsupported context: {context}")

        if self.loss_type == "angle":
            if context != "direction":
                raise ValueError(
                    "loss_type 'angle' is only supported for directional data."
                )
            loss_fn = AngleLoss(reduction="none")
            return loss_fn(prediction, target)

        if self.loss_type in ("l1", "l2"):
            # Optionally limit to xyz for directions.
            pred_feat = prediction[:, :3] if context == "direction" else prediction
            tgt_feat = target[:, :3] if context == "direction" else target
            diff = pred_feat - tgt_feat
            elem = diff.abs() if self.loss_type == "l1" else diff.pow(2)
            # Reduce across feature dimension to a per-sample scalar.
            if self.loss_return_value == "sum":
                return elem.sum(dim=-1)
            else:  # "mean" or "none" default to mean across features.
                return elem.mean(dim=-1)

        raise ValueError(f"Unsupported loss_type: {self.loss_type}")

    def _reduce_batch_loss(self, per_sample_loss: torch.Tensor) -> torch.Tensor:
        """
        Reduce per-sample losses across the batch to a scalar for optimization.

        Parameters
        ----------
        per_sample_loss : torch.Tensor
            Loss values per sample.

        Returns
        -------
        torch.Tensor
            Reduced scalar loss.
        """
        if self.loss_reduction == "mean":
            return per_sample_loss.mean()
        elif self.loss_reduction == "sum":
            return per_sample_loss.sum()
        elif self.loss_reduction == "none":
            # Default to mean to ensure a scalar for backward().
            return per_sample_loss.mean()
        else:
            raise ValueError(f"Unsupported reduction: {self.loss_reduction}")

    def optimize(self, device: torch.device | None = None) -> torch.Tensor | None:
        """
        Optimize the kinematic parameters.

        Parameters
        ----------
        device : torch.device | None, optional
            The device on which to perform computations. If None, the device is auto-selected.

        Returns
        -------
        torch.Tensor | None
            Unreduced per-sample loss values from the final epoch, or None if no heliostats are selected.
        """
        device = get_device(device=device)

        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        if rank == 0:
            log.info("Start the kinematic calibration.")

        if self.heliostats_mask.sum() > 0:
            if (
                self.calibration_method
                == config_dictionary.kinematic_calibration_motor_positions
            ):
                losses = self._optimize_kinematic_parameters_with_motor_positions(
                    device=device,
                )
            elif (
                self.calibration_method
                == config_dictionary.kinematic_calibration_raytracing
            ):
                losses = self._optimize_kinematic_parameters_with_raytracing(
                    device=device,
                )
            else:
                raise ValueError(
                    f"Unsupported calibration method: {self.calibration_method}",
                )
            return losses
        return None

    def _optimize_kinematic_parameters_with_motor_positions(
        self,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Optimize the kinematic parameters using the motor positions.

        This optimizer method optimizes the kinematic parameters by extracting the motor positions
        and incident ray directions from calibration data and using the scene's geometry.

        Parameters
        ----------
        device : torch.device | None, optional
            The device on which to perform computations. If None, the device is auto-selected.

        Returns
        -------
        torch.Tensor
            Unreduced per-sample loss values from the final epoch.
        """
        device = get_device(device=device)

        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        if rank == 0:
            log.info("Kinematic calibration with motor positions.")

        preferred_reflection_directions_measured = torch.nn.functional.normalize(
            (
                self.focal_spots_measured
                - self.heliostat_group.positions.repeat_interleave(
                    self.heliostats_mask, dim=0
                )
            ),
            p=2,
            dim=1,
        )

        loss = torch.inf
        epoch = 0
        log_step = max(1, self.max_epoch // max(1, self.num_log))
        while loss > self.tolerance and epoch <= self.max_epoch:
            self.optimizer.zero_grad()

            self.heliostat_group.activate_heliostats(
                active_heliostats_mask=self.heliostats_mask, device=device
            )

            # Retrieve the orientation of the heliostats for given motor positions.
            orientations = (
                self.heliostat_group.kinematic.motor_positions_to_orientations(
                    motor_positions=self.motor_positions,
                    device=device,
                )
            )

            preferred_reflection_directions = raytracing_utils.reflect(
                incident_ray_directions=self.incident_ray_directions,
                reflection_surface_normals=orientations[:, 0:4, 2],
            )

            unreduced_loss = self._compute_unreduced_loss(
                prediction=preferred_reflection_directions,
                target=preferred_reflection_directions_measured,
                context="direction",
            )
            loss = self._reduce_batch_loss(unreduced_loss)

            loss.backward()
            self.optimizer.step()

            if epoch % log_step == 0:
                log.info(
                    f"Rank: {rank}, Epoch: {epoch}, Loss: {loss}, LR: {self.optimizer.param_groups[0]['lr']}",
                )
            epoch += 1

        log.info(f"Kinematic parameters of group {rank} optimized.")
        return unreduced_loss

    def _optimize_kinematic_parameters_with_raytracing(
        self,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Optimize the kinematic parameters using ray tracing.

        This optimizer method optimizes the kinematic parameters by extracting the focus points
        of calibration images and using heliostat-tracing.

        Parameters
        ----------
        device : torch.device | None, optional
            The device on which to perform computations. If None, the device is auto-selected.

        Returns
        -------
        torch.Tensor
            Unreduced per-sample loss values from the final epoch.
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
        log_step = max(1, self.max_epoch // max(1, self.num_log))
        while loss > self.tolerance and epoch <= self.max_epoch:
            self.optimizer.zero_grad()

            self.heliostat_group.activate_heliostats(
                active_heliostats_mask=self.heliostats_mask, device=device
            )

            # Align heliostats.
            self.heliostat_group.align_surfaces_with_incident_ray_directions(
                aim_points=self.scenario.target_areas.centers[self.target_area_mask],
                incident_ray_directions=self.incident_ray_directions,
                active_heliostats_mask=self.heliostats_mask,
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
                incident_ray_directions=self.incident_ray_directions,
                active_heliostats_mask=self.heliostats_mask,
                target_area_mask=self.target_area_mask,
                device=device,
            )

            # Determine the focal spots of all flux density distributions.
            focal_spots = utils.get_center_of_mass(
                bitmaps=flux_distributions,
                target_centers=self.scenario.target_areas.centers[
                    self.target_area_mask
                ],
                target_widths=self.scenario.target_areas.dimensions[
                    self.target_area_mask
                ][:, 0],
                target_heights=self.scenario.target_areas.dimensions[
                    self.target_area_mask
                ][:, 1],
                device=device,
            )

            # Use reusable helpers.
            unreduced_loss = self._compute_unreduced_loss(
                prediction=focal_spots,
                target=self.focal_spots_measured,
                context="point",
            )
            loss = self._reduce_batch_loss(unreduced_loss)

            loss.backward()
            self.optimizer.step()

            if epoch % log_step == 0:
                log.info(
                    f"Rank: {rank}, Epoch: {epoch}, Loss: {loss}, LR: {self.optimizer.param_groups[0]['lr']}",
                )
            epoch += 1

        log.info(f"Kinematic parameters of group {rank} optimized.")
        return unreduced_loss
