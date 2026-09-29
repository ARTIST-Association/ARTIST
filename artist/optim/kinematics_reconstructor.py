from datetime import datetime
import logging
import os
import pathlib
from functools import partial
from typing import Any, Callable, cast

from matplotlib import pyplot as plt
import torch
from torch.optim.lr_scheduler import LRScheduler

from artist.field.heliostat_group import HeliostatGroup
from artist.flux import bitmap
from artist.geometry import coordinates
from artist.io.calibration_parser import CalibrationDataParser
from artist.optim import training
from artist.optim.loss import (
    AngleLoss,
    FocalSpotLoss,
    Loss,
    reduce_loss_per_sample,
)
from artist.raytracing.geometry import reflect
from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer
from artist.scenario.scenario import Scenario
from artist.util import constants, indices
from artist.util.env import DdpSetup, get_device

log = logging.getLogger(__name__)
"""A logger for the kinematic reconstructor."""


class KinematicsReconstructor:
    """
    An optimizer used to reconstruct real-world kinematics deviation parameters.

    The kinematics reconstructor learns kinematics parameters. These parameters are
    specific to a certain kinematics type and can, for example, include the four kinematics
    rotation deviation parameters as well as the two initial actuator parameters
    for each actuator of a rigid-body kinematics.

    Attributes
    ----------
    ddp_setup : DdpSetup
        Information about the distributed environment, process groups, devices, ranks, world size, and
        heliostat-group-to-ranks mapping.
    scenario : Scenario
        The scenario.
    data : dict[str, CalibrationDataParser | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]]
        The data parser and the mapping of heliostat name and calibration data.
    optimizer_dict : dict[str, Any]
        The parameters for the optimization.
    scheduler_dict : dict[str, Any]
        The parameters for the scheduler.
    dni : float
        Direct normal irradiance in W/m^2.
    reconstruction_method : str
        The reconstruction method. Currently, only reconstruction via ray tracing is implemented.
    validation_loss_focal_spot : FocalSpotLoss
        Flux loss used for validation.
    validation_loss_pixel : PixelLoss
        Pixel loss used for validation.
    validation_loss_kl_div : KLDivergenceLoss
        Kullback-Leibler divergence loss used for validation.

    Note
    ----
    Each heliostat selected for reconstruction needs to have the same number of samples as all others.

    Methods
    -------
    reconstruct_kinematics()
        Reconstruct the kinematics parameters.
    """

    def __init__(
        self,
        ddp_setup: DdpSetup,
        scenario: Scenario,
        data: dict[
            str,
            CalibrationDataParser
            | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
        ],
        optimization_configuration: dict[str, Any],
        case: str,
        dni: float | None = None,
        reconstruction_method: str = constants.kinematics_reconstruction_raytracing,
        bitmap_resolution: torch.Tensor = torch.tensor([256, 256]),
    ) -> None:
        """
        Initialize the kinematics optimizer.

        Parameters
        ----------
        ddp_setup : DdpSetup
            Information about the distributed environment, process groups, devices, ranks, world size, and
            heliostat-group-to-ranks mapping.
        scenario : Scenario
            The scenario.
        data : dict[str, CalibrationDataParser | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]]
            The data parser and the mapping of heliostat name and calibration data.
        optimization_configuration : dict[str, Any]
            Parameters for the optimizer, learning rate scheduler, regularizers, and early stopping.
        dni : float | None
            Direct normal irradiance in W/m^2 (default is None which leads to a ray magnitude of 1.0).
        reconstruction_method : str
            The reconstruction method. Currently, only reconstruction via ray tracing is implemented.
        bitmap_resolution : torch.Tensor
            The resolution of all bitmaps during reconstruction (default is ``torch.tensor([256, 256])``).
            Shape is ``[2]``.
        """
        device = ddp_setup["device"]
        rank = ddp_setup["rank"]
        if rank == 0:
            log.info("Create a kinematics reconstructor.")

        self.ddp_setup = ddp_setup
        self.scenario = scenario
        self.data = data
        self.optimizer_dict = optimization_configuration[constants.optimization]
        self.scheduler_dict = optimization_configuration[constants.scheduler]
        self.dni = dni
        self.bitmap_resolution = bitmap_resolution.to(device)

        self.case=case
        self.batch_size_outer = self.optimizer_dict[constants.batch_size_outer]
        self.validation_loss_focal_spot = FocalSpotLoss(scenario=self.scenario)

        if reconstruction_method in [
            constants.kinematics_reconstruction_raytracing,
            constants.kinematics_reconstruction_alignment,
        ]:
            self.reconstruction_method = reconstruction_method
        else:
            raise ValueError(
                f"The kinematics reconstruction method '{reconstruction_method}' is unknown. "
                f"Please select another reconstruction method and try again!"
            )
        

    def reconstruct_kinematics(
        self,
        loss_definition: Loss,
        device: torch.device | None = None,
    ) -> tuple[
        torch.Tensor, list[list[dict[str, list[float] | dict[str, torch.Tensor]]]]
    ]:
        """
        Reconstruct the kinematic parameters.

        Parameters
        ----------
        loss_definition : Loss
            The definition of the loss function and pre-processing of the prediction.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The final loss of the kinematics reconstruction for each heliostat in each group.
            Shape is ``[total_number_of_heliostats_in_scenario]``.
        list[list[dict[str, list[float] | dict[str, torch.Tensor]]]]
            Loss histories over epochs grouped by rank.
            Outer list: one entry per rank.
            Inner list: one entry per heliostat group processed on that rank.
            Each group entry is a dict with key ``"total_loss"`` mapping to a list
            of per-epoch scalar loss values.
            In non-distributed mode, this is a single-rank container: ``[local_group_histories]``.
        """
        device = get_device(device=device)

        if self.reconstruction_method == constants.kinematics_reconstruction_raytracing:
            loss, loss_history = self._reconstruct_kinematics_flux_driven(
                loss_definition=loss_definition,
                device=device,
            )
        elif (
            self.reconstruction_method == constants.kinematics_reconstruction_alignment
        ):
            loss, loss_history = self._reconstruct_kinematics_alignment_driven(
                loss_definition=loss_definition,
                device=device,
            )

        return loss, loss_history

    def _validate(
        self,
        heliostat_group: HeliostatGroup,
        data_split: training.TrainTestSplit,
        reduction: Callable[..., Any],
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Validate the kinematic reconstruction for a specified heliostat group on the test data.

        Parameters
        ----------
        heliostat_group : HeliostatGroup
            Heliostat group to validate.
        data_split : training.TrainTestSplit
            Train/test split containing all test tensors and metadata.
        reduction : Callable[..., Any]
            Reduction function applied across the sample dimension for each heliostat.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        device = get_device(device=device)

        heliostat_group.activate_heliostats(
            active_heliostats_mask=data_split.active_heliostats_mask_test,
            device=device,
        )

        heliostat_group.align_surfaces_with_motor_positions(
            motor_positions=data_split.motor_positions_test,
            active_heliostats_mask=data_split.active_heliostats_mask_test,
            device=device,
        )

        ray_tracer = HeliostatRayTracer(
            scenario=self.scenario,
            heliostat_group=heliostat_group,
            blocking_active=False,
            batch_size=self.optimizer_dict[constants.batch_size],
            dni=self.dni,
            bitmap_resolution=self.bitmap_resolution,
        )

        flux_prediction, _, _, _ = ray_tracer.trace_rays(
            incident_ray_directions=data_split.incident_ray_directions_test,
            active_heliostats_mask=data_split.active_heliostats_mask_test,
            target_area_indices=data_split.target_area_indices_test,
            device=device,
        )

        indices_for_local_rank = ray_tracer.get_sampler_indices()

        loss_focal_spot_per_sample = self.validation_loss_focal_spot(
            prediction=flux_prediction,
            ground_truth=data_split.flux_measured_test[indices_for_local_rank],
            target_area_indices=data_split.target_area_indices_test[
                indices_for_local_rank
            ],
            device=device,
        )

        heliostat_group.activate_heliostats(
            active_heliostats_mask=data_split.active_heliostats_mask_test,
            device=device,
        )
        orientations = heliostat_group.kinematics.motor_positions_to_orientations(
            motor_positions=data_split.motor_positions_test,
            device=device,
        )
        normals_predicted = orientations @ torch.tensor(
            [0.0, 0.0, 1.0, 0.0], device=device
        )
        normals_measured = self._compute_measured_normals(
            heliostat_group=heliostat_group,
            focal_spots_measured=data_split.focal_spots_measured_test,
            incident_ray_directions=data_split.incident_ray_directions_test,
            active_heliostats_mask=data_split.active_heliostats_mask_test,
            device=device
        )
        loss_definition = AngleLoss()
        angle_loss_normal = loss_definition(
            prediction=normals_predicted,
            ground_truth=normals_measured,
        )

        test_loss_focal_spot = reduce_loss_per_sample(
            loss_per_sample=loss_focal_spot_per_sample,
            number_of_samples_per_heliostat=data_split.number_of_test_samples,
            reduction=reduction,
        )
        test_loss_tracking = reduce_loss_per_sample(
            loss_per_sample=angle_loss_normal,
            number_of_samples_per_heliostat=data_split.number_of_test_samples,
            reduction=reduction,
        )

        print(
            f"validation loss focal spot: {torch.mean(test_loss_focal_spot).item():.5f} meters",
        )
        print(
            f"validation tracking errors: {(torch.mean(test_loss_tracking).item() * 1000.0):.5f} mrad",
        )
        
        return loss_focal_spot_per_sample, flux_prediction


    def _initialize_reconstruction_bookkeeping(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize the per-heliostat loss container and group index offsets.

        Parameters
        ----------
        device : torch.device
            The device on which to perform computations or load tensors and models.

        Returns
        -------
        torch.Tensor
            Final loss per heliostat over all groups, initialized with positive infinity.
            Shape is ``[total_number_of_heliostats_in_scenario]``.
        torch.Tensor
            Prefix sums mapping group-local heliostat indices to global heliostat indices.
            Shape is ``[number_of_heliostat_groups + 1]``.
        """
        final_loss_per_heliostat = torch.full(
            (self.scenario.heliostat_field.number_of_heliostats_per_group.sum(),),
            torch.inf,
            device=device,
        )
        final_loss_start_indices = torch.cat(
            [
                torch.tensor([0], device=device),
                self.scenario.heliostat_field.number_of_heliostats_per_group.cumsum(
                    indices.heliostat_dimension
                ),
            ]
        )
        return final_loss_per_heliostat, final_loss_start_indices

    def _parse_group_calibration_data(
        self, batch_data, heliostat_group: HeliostatGroup, device: torch.device
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Load and parse the calibration data for a single heliostat group.

        Parameters
        ----------
        heliostat_group : HeliostatGroup
            The heliostat group whose calibration data is parsed.
        device : torch.device
            The device on which to perform computations or load tensors and models.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            The measured flux, measured focal spots, incident ray directions, motor positions,
            active heliostats mask, and target area indices.
        """
        parser = cast(CalibrationDataParser, batch_data[constants.data_parser])
        heliostat_mapping = cast(
            list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
            batch_data[constants.heliostat_data_mapping],
        )
        return parser.parse_data_for_reconstruction(
            heliostat_data_mapping=heliostat_mapping,
            heliostat_group=heliostat_group,
            scenario=self.scenario,
            bitmap_resolution=self.bitmap_resolution,
            device=device,
        )

    def _setup_optimizer_scheduler_early_stopping(
        self, heliostat_group: HeliostatGroup
    ) -> tuple[torch.optim.Optimizer, LRScheduler, training.EarlyStopping]:
        """
        Create the optimizer, learning rate scheduler, and early stopping for a group.

        The optimizer learns the rotation deviation parameters of the group's kinematics.

        Parameters
        ----------
        heliostat_group : HeliostatGroup
            The heliostat group whose kinematics parameters are optimized.

        Returns
        -------
        torch.optim.Optimizer
            The Adam optimizer over the rotation deviation parameters.
        LRScheduler
            The learning rate scheduler.
        training.EarlyStopping
            The early stopping monitor.
        """
        # optimizer_params = [
        #     {
        #         "params": heliostat_group.kinematics.rotation_deviation_parameters.requires_grad_(),
        #         "lr": self.optimizer_dict[
        #             constants.initial_learning_rate_rotation_deviation
        #         ],
        #     }
        # ]
        optimizer_params = [
            {
                "params": heliostat_group.kinematics.rotation_deviation_parameters.requires_grad_(),
                "lr": 4e-5
            }
        ]

        optimizer = torch.optim.Adam(optimizer_params)

        scheduler_fn = getattr(
            training,
            self.scheduler_dict[constants.scheduler_type],
        )
        scheduler: LRScheduler = scheduler_fn(
            optimizer=optimizer, parameters=self.scheduler_dict
        )

        early_stopper = training.EarlyStopping(
            window_size=self.optimizer_dict[constants.early_stopping_window],
            patience=self.optimizer_dict[constants.early_stopping_patience],
            min_improvement=self.optimizer_dict[constants.early_stopping_delta],
            relative=True,
        )

        return optimizer, scheduler, early_stopper

    def _compute_measured_normals(
        self,
        heliostat_group: HeliostatGroup,
        focal_spots_measured: torch.Tensor,
        incident_ray_directions: torch.Tensor,
        active_heliostats_mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute the measured surface normals from the measured focal spots.

        The preferred reflection directions are derived from the measured focal spots and the
        heliostat positions and combined with the incident ray directions to obtain the normals.

        Parameters
        ----------
        heliostat_group : HeliostatGroup
            The heliostat group whose normals are computed.
        focal_spots_measured : torch.Tensor
            The measured focal spots.
        incident_ray_directions : torch.Tensor
            The incident ray directions.
        active_heliostats_mask : torch.Tensor
            Mask for active samples available per heliostat.
        device : torch.device
            The device on which to perform computations or load tensors and models.

        Returns
        -------
        torch.Tensor
            The measured normals in 4D format.
        """
        preferred_reflection_directions_measured = torch.nn.functional.normalize(
            (
                focal_spots_measured[:, :3]
                - heliostat_group.positions.repeat_interleave(
                    active_heliostats_mask, dim=0
                )[:, :3]
            ),
            p=2,
            dim=1,
        )
        return coordinates.convert_3d_directions_to_4d_format(
            torch.nn.functional.normalize(
                preferred_reflection_directions_measured
                - incident_ray_directions[:, :3],
                dim=-1,
            ),
            device=device,
        )

    def _compute_alignment_loss(
        self,
        heliostat_group: HeliostatGroup,
        data_split: training.TrainTestSplit,
        loss_definition: Loss,
        normals_measured: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the alignment reconstruction loss for a single epoch.

        The heliostats are activated and their predicted normals are obtained from the motor
        positions and compared against the measured normals.

        Parameters
        ----------
        heliostat_group : HeliostatGroup
            The heliostat group to align.
        data_split : training.TrainTestSplit
            Train/test split containing all training tensors and metadata.
        loss_definition : Loss
            The definition of the loss function and pre-processing of the prediction.
        normals_measured : torch.Tensor
            The measured normals in 4D format.
        device : torch.device
            The device on which to perform computations or load tensors and models.

        Returns
        -------
        torch.Tensor
            The mean loss over all heliostats.
        torch.Tensor
            The loss per heliostat.
        """
        heliostat_group.activate_heliostats(
            active_heliostats_mask=data_split.active_heliostats_mask_train,
            device=device,
        )

        orientations = heliostat_group.kinematics.motor_positions_to_orientations(
            motor_positions=data_split.motor_positions_train,
            device=device,
        )

        normals_predicted = orientations @ torch.tensor(
            [0.0, 0.0, 1.0, 0.0], device=device
        )

        loss_per_sample = loss_definition(
            prediction=normals_predicted,
            ground_truth=normals_measured[data_split.train_indices],
        )

        loss_per_heliostat = reduce_loss_per_sample(
            loss_per_sample=loss_per_sample,
            number_of_samples_per_heliostat=data_split.number_of_train_samples,
            reduction=partial(torch.mean, dim=-1),
        )

        loss = torch.mean(loss_per_heliostat)

        return loss, loss_per_heliostat, loss_per_sample

    def _compute_raytracing_loss(
        self,
        heliostat_group: HeliostatGroup,
        data_split: training.TrainTestSplit,
        loss_definition: Loss,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the ray-tracing reconstruction loss for a single epoch.

        The heliostats are activated, aligned, and ray traced to predict flux distributions
        which are compared against the measured flux.

        Parameters
        ----------
        heliostat_group : HeliostatGroup
            The heliostat group to trace.
        data_split : training.TrainTestSplit
            Train/test split containing all training tensors and metadata.
        loss_definition : Loss
            The definition of the loss function and pre-processing of the prediction.
        device : torch.device
            The device on which to perform computations or load tensors and models.

        Returns
        -------
        torch.Tensor
            The mean loss over all heliostats.
        torch.Tensor
            The loss per heliostat.
        torch.Tensor
            The sample indices processed on the local rank.
        """
        heliostat_group.activate_heliostats(
            active_heliostats_mask=data_split.active_heliostats_mask_train,
            device=device,
        )

        heliostat_group.align_surfaces_with_motor_positions(
            motor_positions=data_split.motor_positions_train,
            active_heliostats_mask=data_split.active_heliostats_mask_train,
            device=device,
        )

        # Create a parallelized ray tracer. Blocking is always deactivated for this reconstruction.
        ray_tracer = HeliostatRayTracer(
            scenario=self.scenario,
            heliostat_group=heliostat_group,
            blocking_active=False,
            world_size=self.ddp_setup["heliostat_group_world_size"],
            rank=self.ddp_setup["heliostat_group_rank"],
            batch_size=self.optimizer_dict[constants.batch_size],
            random_seed=self.ddp_setup["heliostat_group_rank"],
            dni=self.dni,
            bitmap_resolution=self.bitmap_resolution,
        )

        flux_prediction_train, _, _, _ = ray_tracer.trace_rays(
            incident_ray_directions=data_split.incident_ray_directions_train,
            active_heliostats_mask=data_split.active_heliostats_mask_train,
            target_area_indices=data_split.target_area_indices_train,
            device=device,
        )

        sample_indices_for_local_rank = ray_tracer.get_sampler_indices()

        loss_per_sample = loss_definition(
            prediction=flux_prediction_train,
            ground_truth=data_split.flux_measured_train[sample_indices_for_local_rank],
            target_area_indices=data_split.target_area_indices_train[
                sample_indices_for_local_rank
            ],
            reduction_dimensions=(
                indices.batched_bitmap_e,
                indices.batched_bitmap_u,
            ),
            device=device,
        )

        loss_per_heliostat = reduce_loss_per_sample(
            loss_per_sample=loss_per_sample,
            number_of_samples_per_heliostat=data_split.number_of_train_samples,
            reduction=partial(torch.mean, dim=-1),
        )

        loss = torch.mean(loss_per_heliostat)

        return loss, loss_per_heliostat, sample_indices_for_local_rank, loss_per_sample

    def _synchronize_gradients_nested_ddp(
        self, optimizer: torch.optim.Optimizer
    ) -> None:
        """
        Synchronize gradients within a nested heliostat-group subgroup.

        In nested distributed data parallel mode the gradients are summed across the ranks that
        process the same heliostat group and then averaged. This is a no-op when not nested.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer whose parameter gradients are synchronized.
        """
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
                        param.grad /= self.ddp_setup["heliostat_group_world_size"]

    def _synchronize_reconstruction_across_ranks(
        self,
        final_loss_per_heliostat: torch.Tensor,
    ) -> list[list[dict[str, list[float] | dict[str, torch.Tensor]]]]:
        """
        Synchronize the reconstruction results across all distributed ranks.

        Broadcasts the reconstructed kinematics parameters, reduces the final loss to its
        minimum across ranks, and gathers the loss histories of all ranks.

        Parameters
        ----------
        final_loss_per_heliostat : torch.Tensor
            The final loss per heliostat on the local rank.
            Shape is ``[total_number_of_heliostats_in_scenario]``.
        """
        rank = self.ddp_setup["rank"]

        if self.ddp_setup["is_distributed"]:
            for index, heliostat_group in enumerate(
                self.scenario.heliostat_field.heliostat_groups
            ):
                source = self.ddp_setup["ranks_to_groups_mapping"][index]
                torch.distributed.broadcast(
                    heliostat_group.kinematics.rotation_deviation_parameters,
                    src=source[indices.first_rank_from_group],
                )
                torch.distributed.broadcast(
                    heliostat_group.kinematics.actuators.optimizable_parameters,
                    src=source[indices.first_rank_from_group],
                )
            torch.distributed.all_reduce(
                final_loss_per_heliostat, op=torch.distributed.ReduceOp.MIN
            )

            log.info(f"Rank: {rank}, synchronized after kinematics reconstruction.")


    def _reconstruct_kinematics_alignment_driven(
        self,
        loss_definition: Loss,
        device: torch.device | None = None,
    ) -> tuple[
        torch.Tensor, list[list[dict[str, list[float] | dict[str, torch.Tensor]]]]
    ]:
        """
        Reconstruct the kinematics parameters using alignment and geometry data.

        This reconstruction method optimizes the kinematics parameters by iteratively
        aligning heliostats to reach a defined flux focal spot.

        Parameters
        ----------
        loss_definition : Loss
            Definition of the loss function and pre-processing of the prediction.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The final loss of the kinematics reconstruction for each heliostat in each group.
            Shape is ``[total_number_of_heliostats_in_scenario]``.
        list[list[dict[str, list[float] | dict[str, torch.Tensor]]]]
            Loss histories over epochs grouped by rank.
            Outer list: one entry per rank.
            Inner list: one entry per heliostat group processed on that rank.
            Each group entry is a dict with key ``"total_loss"`` mapping to a list
            of per-epoch scalar loss values.
            In non-distributed mode, this is a single-rank container: ``[local_group_histories]``.
        """
        device = get_device(device=device)
        rank = self.ddp_setup["rank"]

        if rank == 0:
            log.info("Beginning kinematics reconstruction with alignment.")

        final_loss_per_heliostat, final_loss_start_indices = (
            self._initialize_reconstruction_bookkeeping(device=device)
        )
    
        data_mappings = self.data[
            constants.heliostat_data_mapping
        ]

        for i in range(0, len(data_mappings), self.batch_size_outer):
            batch_data = {
                constants.data_parser: self.data[constants.data_parser],
                constants.heliostat_data_mapping: data_mappings[i : i + self.batch_size_outer],
                constants.validation_sample_fraction: self.data[constants.validation_sample_fraction]
            }
            print(i)

            # Process only groups assigned to this rank.
            for heliostat_group_index in self.ddp_setup["groups_to_ranks_mapping"][rank]:
                heliostat_group: HeliostatGroup = (
                    self.scenario.heliostat_field.heliostat_groups[heliostat_group_index]
                )

                (
                    flux_measured,
                    focal_spots_measured,
                    incident_ray_directions,
                    motor_positions,
                    active_heliostats_mask,
                    target_area_indices,
                ) = self._parse_group_calibration_data(
                    batch_data=batch_data, heliostat_group=heliostat_group, device=device
                )

                # Skip groups with no active heliostats.
                if active_heliostats_mask.sum() > 0:

                    if (len(data_mappings) != len(heliostat_group.names)):
                        log.warning("Not all heliostats in this group are being reconstructed!")

                    data_split: training.TrainTestSplit = training.train_test_split(
                        active_heliostats_mask=active_heliostats_mask,
                        flux_measured=flux_measured,
                        focal_spots_measured=focal_spots_measured,
                        incident_ray_directions=incident_ray_directions,
                        motor_positions=motor_positions,
                        target_area_indices=target_area_indices,
                        test_fraction=self.data[constants.validation_sample_fraction],
                        device=device,
                    )
                    # Calculate focal spot from measured flux.
                    normals_measured = self._compute_measured_normals(
                        heliostat_group=heliostat_group,
                        focal_spots_measured=focal_spots_measured,
                        incident_ray_directions=incident_ray_directions,
                        active_heliostats_mask=active_heliostats_mask,
                        device=device,
                    )

                    # Set up optimizer, scheduler, and early stopping.
                    optimizer, _, early_stopper = (
                        self._setup_optimizer_scheduler_early_stopping(
                            heliostat_group=heliostat_group
                        )
                    )

                    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer,
                        max_lr=4e-3,
                        total_steps=self.optimizer_dict[constants.max_epoch] + 1,
                        pct_start=0.15,
                        anneal_strategy="cos",
                        div_factor=100,
                        final_div_factor=10,
                    )

                    # Start the optimization.
                    loss = torch.inf
                    epoch = 0
                    log_step = (
                        self.optimizer_dict[constants.max_epoch]
                        if self.optimizer_dict[constants.log_step] == 0
                        else self.optimizer_dict[constants.log_step]
                    )
                    while (
                        loss > float(self.optimizer_dict[constants.tolerance])
                        and epoch <= self.optimizer_dict[constants.max_epoch]
                    ):
                        optimizer.zero_grad()

                        loss, loss_per_heliostat, loss_per_sample = self._compute_alignment_loss(
                            heliostat_group=heliostat_group,
                            data_split=data_split,
                            loss_definition=loss_definition,
                            normals_measured=normals_measured,
                            device=device,
                        )

                        loss.backward()

                        optimizer.step()
                        # if isinstance(
                        #     scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                        # ):
                        #     scheduler.step(loss.detach())
                        # else:
                        #     scheduler.step()
                        if epoch < 3000:
                            scheduler.step()
                        else:
                            # Freeze at whatever LR you choose
                            for group in optimizer.param_groups:
                                group["lr"] = 0.0018
                        

                        stop = early_stopper.step(loss.item())

                        if epoch % log_step == 0 or stop:
                            log.info(
                                f"Rank: {rank}, Epoch: {epoch}, Loss: {loss}, LR: {optimizer.param_groups[0]['lr']}",
                            )

                            # with torch.no_grad():
                            #     test_loss, flux_test = self._validate(
                            #         heliostat_group=heliostat_group,
                            #         data_split=data_split,
                            #         reduction=partial(torch.mean, dim=-1),
                            #         device=device,
                            #     )
                        
                        # Early stopping when loss did not improve for a predefined number of epochs.
                        if stop:
                            log.info(f"Early stopping at epoch {epoch}.")
                            break

                        if epoch == 0:
                            save_dir = pathlib.Path(
                                f"/workVERLEIHNIX/mb/ARTIST/dissertation/kinematics_data/{self.case}/reconstruction"
                            )
                            os.makedirs(save_dir, exist_ok=True)
                            results = {
                                "deviation_params": torch.zeros_like(heliostat_group.kinematics.rotation_deviation_parameters.detach().cpu()),
                                "training_loss": [],
                                "target_area_indices": []
                            }
                            for e in range(0, self.optimizer_dict[constants.max_epoch]+1, 1000):
                                file_path = save_dir / f"results_alignment_{e}_{data_split.number_of_train_samples}.pt"
                                if not file_path.exists():
                                    torch.save(results, file_path)

                        if epoch % 1000 == 0:
                            results = torch.load(f"/workVERLEIHNIX/mb/ARTIST/dissertation/kinematics_data/{self.case}/reconstruction/results_alignment_{epoch}_{data_split.number_of_train_samples}.pt")
                            results["deviation_params"][active_heliostats_mask!=0] = heliostat_group.kinematics.rotation_deviation_parameters[active_heliostats_mask!=0].detach().cpu()
                            results["training_loss"].append(loss_per_sample.detach().cpu().tolist())
                            results["target_area_indices"].append(data_split.target_area_indices_train.detach().cpu().tolist())
                            torch.save(results, f"/workVERLEIHNIX/mb/ARTIST/dissertation/kinematics_data/{self.case}/reconstruction/results_alignment_{epoch}_{data_split.number_of_train_samples}.pt")     

                        epoch += 1

                active_indices_group = torch.nonzero(
                    active_heliostats_mask != 0, as_tuple=True
                )[0]

                final_indices = (
                    active_indices_group
                    + final_loss_start_indices[heliostat_group_index]
                )

                final_loss_per_heliostat[final_indices] = loss_per_heliostat

                log.info(f"Rank: {rank}, Kinematics reconstructed.")

        for heliostat_group in self.scenario.heliostat_field.heliostat_groups:
            heliostat_group.kinematics.rotation_deviation_parameters = (
                heliostat_group.kinematics.rotation_deviation_parameters.detach()
            )

        return final_loss_per_heliostat.detach().cpu(), None

    def _reconstruct_kinematics_flux_driven(
        self,
        loss_definition: Loss,
        device: torch.device | None = None,
    ) -> tuple[
        torch.Tensor, list[list[dict[str, list[float] | dict[str, torch.Tensor]]]]
    ]:
        """
        Reconstruct the kinematics parameters using ray tracing and comparing fluxes.

        This reconstruction method optimizes the kinematics parameters by extracting the focal points
        of calibration images and using heliostat-tracing.

        Parameters
        ----------
        loss_definition : Loss
            Definition of the loss function and pre-processing of the prediction.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The final loss of the kinematics reconstruction for each heliostat in each group.
            Shape is ``[total_number_of_heliostats_in_scenario]``.
        list[list[dict[str, list[float] | dict[str, torch.Tensor]]]]
            Loss histories over epochs grouped by rank.
            Outer list: one entry per rank.
            Inner list: one entry per heliostat group processed on that rank.
            Each group entry is a dict with key ``"total_loss"`` mapping to a list
            of per-epoch scalar loss values.
            In non-distributed mode, this is a single-rank container: ``[local_group_histories]``.
        """
        device = get_device(device=device)
        rank = self.ddp_setup["rank"]

        if rank == 0:
            log.info("Beginning kinematics reconstruction with ray tracing.")

        final_loss_per_heliostat, final_loss_start_indices = (
            self._initialize_reconstruction_bookkeeping(device=device)
        )

        data_mappings = self.data[
            constants.heliostat_data_mapping
        ]

        for i in range(0, len(data_mappings), self.batch_size_outer):
            batch_data = {
                constants.data_parser: self.data[constants.data_parser],
                constants.heliostat_data_mapping: data_mappings[i : i + self.batch_size_outer],
                constants.validation_sample_fraction: self.data[constants.validation_sample_fraction]
            }
            print(i)

            # Process only groups assigned to this rank.
            for heliostat_group_index in self.ddp_setup[constants.groups_to_ranks_mapping][rank]:
                heliostat_group: HeliostatGroup = (
                    self.scenario.heliostat_field.heliostat_groups[heliostat_group_index]
                )
                (
                    flux_measured,
                    focal_spots_measured,
                    incident_ray_directions,
                    motor_positions,
                    active_heliostats_mask,
                    target_area_indices,
                ) = self._parse_group_calibration_data(
                    batch_data=batch_data, heliostat_group=heliostat_group, device=device
                )

                # Skip groups with no active heliostats.
                if active_heliostats_mask.sum() > 0:
        
                    if (len(data_mappings) != len(heliostat_group.names)):
                        log.warning("Not all heliostats in this group are being reconstructed!")

                    data_split: training.TrainTestSplit = training.train_test_split(
                        active_heliostats_mask=active_heliostats_mask,
                        flux_measured=flux_measured,
                        focal_spots_measured=focal_spots_measured,
                        incident_ray_directions=incident_ray_directions,
                        motor_positions=motor_positions,
                        target_area_indices=target_area_indices,
                        test_fraction=self.data[constants.validation_sample_fraction],
                        device=device,
                    )

                    # Set up optimizer, scheduler, and early stopping.
                    optimizer, _, early_stopper = (
                        self._setup_optimizer_scheduler_early_stopping(
                            heliostat_group=heliostat_group
                        )
                    )

                    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer,
                        max_lr=1.2e-3,
                        total_steps=self.optimizer_dict[constants.max_epoch]+1,
                        pct_start=0.20,
                        anneal_strategy="cos",
                        div_factor=1.0e2,
                        final_div_factor=1.0e3,
                    )

                    # Start the optimization.
                    loss = torch.inf
                    epoch = 0
                    log_step = (
                        self.optimizer_dict[constants.max_epoch]
                        if self.optimizer_dict[constants.log_step] == 0
                        else self.optimizer_dict[constants.log_step]
                    )
                    while (
                        loss > float(self.optimizer_dict[constants.tolerance])
                        and epoch <= self.optimizer_dict[constants.max_epoch]
                    ):
                        optimizer.zero_grad()

                        loss, loss_per_heliostat, sample_indices_for_local_rank, loss_per_sample = (
                            self._compute_raytracing_loss(
                                heliostat_group=heliostat_group,
                                data_split=data_split,
                                loss_definition=loss_definition,
                                device=device,
                            )
                        )

                        loss.backward()
                        self._synchronize_gradients_nested_ddp(optimizer=optimizer)

                        optimizer.step()
                        if isinstance(
                            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                        ):
                            scheduler.step(loss.detach())
                        else:
                            scheduler.step()

                        is_last_epoch = (
                            epoch == self.optimizer_dict[constants.max_epoch] - 1
                        )
                        stop = early_stopper.step(loss.item())

                        if epoch % log_step == 0 or is_last_epoch or stop:
                            log.info(
                                f"Rank: {rank}, Epoch: {epoch}, Loss: {loss}",
                            )

                            # with torch.no_grad():
                            #     test_loss, flux_test = self._validate(
                            #         heliostat_group=heliostat_group,
                            #         data_split=data_split,
                            #         reduction=partial(torch.mean, dim=1),
                            #         device=device,
                            #     )

                        # Early stopping when loss did not improve for a predefined number of epochs.
                        if stop:
                            log.info(f"Early stopping at epoch {epoch}.")
                            break
                        
                        if epoch == 0:
                            save_dir = pathlib.Path(
                                f"/workVERLEIHNIX/mb/ARTIST/dissertation/kinematics_data/{self.case}/reconstruction"
                            )
                            os.makedirs(save_dir, exist_ok=True)
                            results = {
                                "deviation_params": torch.zeros_like(heliostat_group.kinematics.rotation_deviation_parameters.detach().cpu()),
                                "training_loss": [],
                                "target_area_indices": []
                            }
                            for e in range(0, self.optimizer_dict[constants.max_epoch]+1, 100):
                                file_path = save_dir / f"results_flux_{e}_{data_split.number_of_train_samples}.pt"
                                if not file_path.exists():
                                    torch.save(results, file_path)

                        if epoch % 100 == 0:
                            results = torch.load(f"/workVERLEIHNIX/mb/ARTIST/dissertation/kinematics_data/{self.case}/reconstruction/results_flux_{epoch}_{data_split.number_of_train_samples}.pt")
                            results["deviation_params"][active_heliostats_mask!=0] = heliostat_group.kinematics.rotation_deviation_parameters[active_heliostats_mask!=0].detach().cpu()
                            results["training_loss"].append(loss_per_sample.detach().cpu().tolist())
                            results["target_area_indices"].append(data_split.target_area_indices_train.detach().cpu().tolist())
                            torch.save(results, f"/workVERLEIHNIX/mb/ARTIST/dissertation/kinematics_data/{self.case}/reconstruction/results_flux_{epoch}_{data_split.number_of_train_samples}.pt")     

                        epoch += 1
                    
                local_indices = (
                    sample_indices_for_local_rank[:: data_split.number_of_train_samples]
                    // data_split.number_of_train_samples
                )

                global_active_indices = torch.nonzero(
                    active_heliostats_mask != 0, as_tuple=True
                )[0]

                rank_active_indices_global = global_active_indices[local_indices]

                final_indices = (
                    rank_active_indices_global
                    + final_loss_start_indices[heliostat_group_index]
                )

                final_loss_per_heliostat[final_indices] = loss_per_heliostat

                log.info(f"Rank: {rank}, Kinematics reconstructed.")

        self._synchronize_reconstruction_across_ranks(
            final_loss_per_heliostat=final_loss_per_heliostat,
        )

        for heliostat_group in self.scenario.heliostat_field.heliostat_groups:
            heliostat_group.kinematics.rotation_deviation_parameters = (
                heliostat_group.kinematics.rotation_deviation_parameters.detach()
            )

        return final_loss_per_heliostat.detach().cpu(), None
