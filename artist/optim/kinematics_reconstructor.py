from dataclasses import dataclass
from functools import partial
import logging
import pathlib
from typing import Any, cast

from matplotlib import pyplot as plt
import torch
from torch.optim.lr_scheduler import LRScheduler

from artist.field.heliostat_group import HeliostatGroup
from artist.flux import bitmap
from artist.geometry import coordinates
from artist.io.calibration_parser import CalibrationDataParser
from artist.optim import training
from artist.optim.loss import FocalSpotLoss, KLDivergenceLoss, Loss, PixelLoss, reduce_loss_per_sample
from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer
from artist.scenario.scenario import Scenario
from artist.util import constants, indices
from artist.util.env import DdpSetup, get_device

log = logging.getLogger(__name__)
"""A logger for the kinematic reconstructor."""

@dataclass
class TrainTestSplit:
    """
    Container holding the train/test split for heliostat reconstruction data.

    Attributes
    ----------
    flux_measured_train : torch.Tensor
        Measured flux distributions for the training set.
    focal_spots_measured_train : torch.Tensor
        Measured focal spot coordinates for the training set.
    incident_ray_directions_train : torch.Tensor
        Incident ray directions for the training set.
    motor_positions_train : torch.Tensor
        Motor positions for the training set.
    target_area_indices_train : torch.Tensor
        Target area indices for the training set.
    flux_measured_test : torch.Tensor
        Measured flux distributions for the test set.
    focal_spots_measured_test : torch.Tensor
        Measured focal spot coordinates for the test set.
    incident_ray_directions_test : torch.Tensor
        Incident ray directions for the test set.
    motor_positions_test : torch.Tensor
        Motor positions for the test set.
    target_area_indices_test : torch.Tensor
        Target area indices for the test set.
    active_heliostats_mask_train : torch.Tensor
        Number of active training samples per heliostat.
    active_heliostats_mask_test : torch.Tensor
        Number of active test samples per heliostat.
    train_indices : torch.Tensor
        Flattened indices selecting training samples from the original dataset.
    test_indices : torch.Tensor
        Flattened indices selecting test samples from the original dataset.
    number_of_train_samples : int
        Number of training samples per heliostat.
    number_of_test_samples : int
        Number of test samples per heliostat.
    number_of_samples_per_heliostat : int
        Total number of samples available per heliostat.
    """
    flux_measured_train: torch.Tensor
    focal_spots_measured_train: torch.Tensor
    incident_ray_directions_train: torch.Tensor
    motor_positions_train: torch.Tensor
    target_area_indices_train: torch.Tensor

    flux_measured_test: torch.Tensor
    focal_spots_measured_test: torch.Tensor
    incident_ray_directions_test: torch.Tensor
    motor_positions_test: torch.Tensor
    target_area_indices_test: torch.Tensor

    active_heliostats_mask_train: torch.Tensor
    active_heliostats_mask_test: torch.Tensor

    train_indices: torch.Tensor
    test_indices: torch.Tensor

    number_of_train_samples: int
    number_of_test_samples: int
    number_of_samples_per_heliostat: int

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
        rank = ddp_setup["rank"]
        if rank == 0:
            log.info("Create a kinematics reconstructor.")

        self.ddp_setup = ddp_setup
        self.scenario = scenario
        self.data = data
        self.optimizer_dict = optimization_configuration[constants.optimization]
        self.scheduler_dict = optimization_configuration[constants.scheduler]
        self.dni = dni
        self.bitmap_resolution = bitmap_resolution

        self.loss_focal_spot = FocalSpotLoss(scenario=self.scenario)
        self.loss_pixel = PixelLoss()
        self.loss_kl_div = KLDivergenceLoss()

        if reconstruction_method in [
            constants.kinematics_reconstruction_raytracing,
            constants.kinematics_reconstruction_geometry
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
    ) -> tuple[torch.Tensor, list[Any]]:
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
        list[list[dict[str, list[float]]]]
            Loss histories over epochs grouped by rank.
            Outer list: one entry per rank.
            Inner list: one entry per heliostat group processed on that rank.
            Each group entry is a dict with key ``"total_loss"`` mapping to a list
            of per-epoch scalar loss values.
            In non-distributed mode, this is a single-rank container: ``[local_group_histories]``.
        """
        device = get_device(device=device)

        if self.reconstruction_method == constants.kinematics_reconstruction_raytracing:
            loss, loss_history = (
                self._reconstruct_kinematics_parameters_with_raytracing(
                    loss_definition=loss_definition,
                    device=device,
                )
            )
        elif (
            self.reconstruction_method == constants.kinematics_reconstruction_geometry
        ):
            loss, loss_history = (
                self._reconstruct_kinematics_parameters_with_geometry(
                    loss_definition=loss_definition,
                    device=device,
                )
            )

        else:
            raise ValueError(
                f"The kinematics reconstruction method '{self.reconstruction_method}' is unknown. "
                f"Please select another reconstruction method and try again!"
            )

        return loss, loss_history


    def train_test_split(
        self,
        active_heliostats_mask: torch.Tensor,
        flux_measured: torch.Tensor,
        focal_spots_measured: torch.Tensor,
        incident_ray_directions: torch.Tensor,
        motor_positions: torch.Tensor,
        target_area_indices: torch.Tensor,
        test_fraction: int = 0.25,
        device: torch.device | None = None
    ) -> TrainTestSplit:
        device = get_device(device=device)
       
        total_samples = int(active_heliostats_mask.sum().item())
        number_of_heliostats = int((active_heliostats_mask > 0).sum().item())
        number_of_samples_per_heliostat = int(
            total_samples / number_of_heliostats
        )
        number_of_test_samples = max(1, int(number_of_samples_per_heliostat * test_fraction))
        number_of_train_samples = number_of_samples_per_heliostat - number_of_test_samples
        starts = torch.arange(number_of_heliostats) * number_of_samples_per_heliostat
        offsets = torch.arange(number_of_train_samples, number_of_samples_per_heliostat)
        train_indices = (
            torch.arange(0, total_samples, number_of_samples_per_heliostat, device=device)[:, None]
            + torch.arange(number_of_train_samples, device=device)
        ).reshape(-1)
        test_indices = (starts[:, None] + offsets).reshape(-1)

        active_heliostats_mask_train = torch.clamp(active_heliostats_mask - number_of_test_samples, min=0)
        active_heliostats_mask_test = torch.clamp(active_heliostats_mask - number_of_train_samples, min=0)

        return TrainTestSplit(
            flux_measured_train=flux_measured[train_indices],
            focal_spots_measured_train=focal_spots_measured[train_indices],
            incident_ray_directions_train=incident_ray_directions[train_indices],
            motor_positions_train=motor_positions[train_indices],
            target_area_indices_train=target_area_indices[train_indices],

            flux_measured_test=flux_measured[test_indices],
            focal_spots_measured_test=focal_spots_measured[test_indices],
            incident_ray_directions_test=incident_ray_directions[test_indices],
            motor_positions_test=motor_positions[test_indices],
            target_area_indices_test=target_area_indices[test_indices],

            active_heliostats_mask_train=active_heliostats_mask_train,
            active_heliostats_mask_test=active_heliostats_mask_test,

            train_indices=train_indices,
            test_indices=test_indices,

            number_of_train_samples=number_of_train_samples,
            number_of_test_samples=number_of_test_samples,
            number_of_samples_per_heliostat=number_of_samples_per_heliostat,
        )
    

    def validate(
        self,
        heliostat_group: HeliostatGroup,
        data_split: TrainTestSplit,
        device: torch.device | None = None,
    ):
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
            world_size=self.ddp_setup["heliostat_group_world_size"],
            rank=self.ddp_setup["heliostat_group_rank"],
            batch_size=self.optimizer_dict[constants.batch_size],
            random_seed=self.ddp_setup["heliostat_group_rank"],
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

        loss_focal_spot_per_sample = self.loss_focal_spot(
            prediction=flux_prediction,
            ground_truth=data_split.flux_measured_test[
                indices_for_local_rank
            ],
            target_area_indices=data_split.target_area_indices_test[
                indices_for_local_rank
            ],
            device=device
        )
        loss_pixel_per_sample = self.loss_pixel(
            prediction=flux_prediction,
            ground_truth=data_split.flux_measured_test[
                indices_for_local_rank
            ],
            reduction_dimensions=(1,2,),
        )
        loss_kl_div_per_sample = self.loss_kl_div(
            prediction=flux_prediction,
            ground_truth=data_split.flux_measured_test[
                indices_for_local_rank
            ],
            reduction_dimensions=(1,2,),
        )

        test_loss_focal_spot = reduce_loss_per_sample(
            loss_per_sample=loss_focal_spot_per_sample,
            number_of_samples_per_heliostat=data_split.number_of_test_samples,
            reduction=partial(torch.median, dim=1),
        )                   
        test_loss_pixel = reduce_loss_per_sample(
            loss_per_sample=loss_pixel_per_sample,
            number_of_samples_per_heliostat=data_split.number_of_test_samples,
            reduction=partial(torch.median, dim=1),
        )                        
        test_loss_kl_div = reduce_loss_per_sample(
            loss_per_sample=loss_kl_div_per_sample,
            number_of_samples_per_heliostat=data_split.number_of_test_samples,
            reduction=partial(torch.median, dim=1),
        )

        print(f"test loss focal spot, mean: {torch.mean(test_loss_focal_spot)}")
        print(f"test loss pixel, mean: {torch.mean(test_loss_pixel)}")
        print(f"test loss kl div, mean: {torch.mean(test_loss_kl_div)}")

        return flux_prediction

    def plot_fluxes(
        self,
        flux_measured: torch.Tensor,
        flux_prediction_train: torch.Tensor,
        flux_prediction_test: torch.Tensor,
        data_split: TrainTestSplit,
        plot_name: str,
    ) -> None:
        device=torch.device("cpu")
        flux_predicted = torch.zeros_like(flux_measured, device=device)
        flux_predicted[data_split.train_indices] = flux_prediction_train
        flux_predicted[data_split.test_indices] = flux_prediction_test
        samples_per_heliostat = data_split.number_of_samples_per_heliostat
        total_samples = flux_measured.shape[0]
        
        centers_of_mass_predicted = bitmap.get_center_of_mass(flux_predicted, device=device)
        centers_of_mass_measured = bitmap.get_center_of_mass(flux_measured, device=device)

        for heliostat_start_index in range(0, total_samples, samples_per_heliostat):
            fig, axes = plt.subplots(
                samples_per_heliostat, 2, figsize=(8, samples_per_heliostat * 4),
            )
            heliostat_index = heliostat_start_index // samples_per_heliostat

            for sample_offset in range(samples_per_heliostat):
                sample_index = heliostat_start_index + sample_offset

                axes[sample_offset, 0].imshow(flux_predicted[sample_index])
                axes[sample_offset, 0].set_title(f"Predicted Flux - Heliostat {heliostat_index}")

                axes[sample_offset, 1].imshow(flux_measured[sample_index])
                axes[sample_offset, 1].set_title(f"Measured Flux - Heliostat {heliostat_index}")

                for ax in axes[sample_offset, :]:
                    ax.scatter(
                        centers_of_mass_measured[sample_index, 0],
                        centers_of_mass_measured[sample_index, 1],
                        c="black",
                        s=30,
                        marker="o",
                    )
                    ax.scatter(
                        centers_of_mass_predicted[sample_index, 0],
                        centers_of_mass_predicted[sample_index, 1],
                        c="red",
                        s=30,
                        marker="x",
                    )   

            plt.tight_layout()
            plt.savefig(f"./ignored/new/heliostat_{heliostat_start_index // samples_per_heliostat}_{plot_name}")
            plt.close(fig)


    def _reconstruct_kinematics_parameters_with_geometry(
        self,
        loss_definition: Loss,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, list[list[dict[str, list[float]]]]]:
        """
        Reconstruct the kinematics parameters using ray tracing.

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
        list[list[dict[str, list[float]]]]
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

        # Initialize final loss per heliostat, group offset table into global heliostat index space, and
        # per-group loss curves for this rank.
        final_loss_per_heliostat = torch.full(
            (self.scenario.heliostat_field.number_of_heliostats_per_group.sum(),),
            torch.inf,
            device=device,
        )
        loss_history: list[dict[str, list[float]]] = []

        # Iterate heliostat groups assigned to this rank.
        for heliostat_group_index in self.ddp_setup["groups_to_ranks_mapping"][rank]:
            # Parse calibration inputs for current group to obtain measured flux, incident ray directions, mask of
            # active heliostats, and target area indices.
            heliostat_group: HeliostatGroup = (
                self.scenario.heliostat_field.heliostat_groups[heliostat_group_index]
            )
            parser = cast(CalibrationDataParser, self.data[constants.data_parser])
            heliostat_mapping = cast(
                list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
                self.data[constants.heliostat_data_mapping],
            )
            (
                flux_measured,
                focal_spots_measured,
                incident_ray_directions,
                motor_positions,
                active_heliostats_mask,
                target_area_indices,
            ) = parser.parse_data_for_reconstruction(
                heliostat_data_mapping=heliostat_mapping,
                heliostat_group=heliostat_group,
                scenario=self.scenario,
                bitmap_resolution=self.bitmap_resolution,
                device=device,
            )

            if active_heliostats_mask.sum() > 0:

                data_split: TrainTestSplit = self.train_test_split(
                    active_heliostats_mask=active_heliostats_mask,
                    flux_measured=flux_measured,
                    focal_spots_measured=focal_spots_measured,
                    incident_ray_directions=incident_ray_directions,
                    motor_positions=motor_positions,
                    target_area_indices=target_area_indices,
                    test_fraction=0.25,
                    device=device
                )

                preferred_reflection_directions_measured = (
                    torch.nn.functional.normalize(
                        (
                            focal_spots_measured[:, :3]
                            - heliostat_group.positions.repeat_interleave(
                                active_heliostats_mask, dim=0
                            )[:, :3]
                        ),
                        p=2,
                        dim=1,
                    )
                )

                normals_measured = coordinates.convert_3d_directions_to_4d_format(torch.nn.functional.normalize(preferred_reflection_directions_measured - incident_ray_directions[:, :3], dim=-1), device=device)

                # Set up optimizer, scheduler, and early stopping.
                optimizer_params = [
                    {
                        "params": heliostat_group.kinematics.rotation_deviation_parameters.requires_grad_(),
                        "lr": self.optimizer_dict[
                            constants.initial_learning_rate_rotation_deviation
                        ],
                    }
                ]

                optimizer = torch.optim.Adam(optimizer_params)

                # Create a learning rate scheduler.
                scheduler_fn = getattr(
                    training,
                    self.scheduler_dict[constants.scheduler_type],
                )
                scheduler: LRScheduler = scheduler_fn(
                    optimizer=optimizer, parameters=self.scheduler_dict
                )

                # Set up early stopping.
                early_stopper = training.EarlyStopping(
                    window_size=self.optimizer_dict[constants.early_stopping_window],
                    patience=self.optimizer_dict[constants.early_stopping_patience],
                    min_improvement=self.optimizer_dict[constants.early_stopping_delta],
                    relative=True,
                )

                loss_history_list = []

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

                    # Activate heliostats.
                    heliostat_group.activate_heliostats(
                        active_heliostats_mask=data_split.active_heliostats_mask_train, device=device
                    )

                    orientations = heliostat_group.kinematics.motor_positions_to_orientations(
                        motor_positions=data_split.motor_positions_train,
                        device=device,
                    )

                    normals_predicted = orientations @ torch.tensor(
                        [0.0, 0.0, 1.0, 0.0], device=device
                    )

                    # Compute loss from prediction vs. measured normals.
                    loss_per_sample = loss_definition(
                        prediction=normals_predicted,
                        ground_truth=normals_measured[data_split.train_indices],
                    )

                    loss_per_heliostat = reduce_loss_per_sample(
                        loss_per_sample=loss_per_sample,
                        number_of_samples_per_heliostat=data_split.number_of_train_samples,
                        reduction=partial(torch.mean)
                    )

                    loss = torch.mean(loss_per_heliostat)

                    loss.backward()
                    
                    optimizer.param_groups[0]["params"][0].grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

                    if self.ddp_setup["is_nested"]:
                        # Reduce gradients within each heliostat group.
                        for param_group in optimizer.param_groups:
                            for param in param_group["params"]:
                                if param.grad is not None:
                                    param.grad = (
                                        torch.distributed.nn.functional.all_reduce(
                                            param.grad,
                                            op=torch.distributed.ReduceOp.SUM,
                                            group=self.ddp_setup["process_subgroup"],
                                        )
                                    )
                                    param.grad /= self.ddp_setup[
                                        "heliostat_group_world_size"
                                    ]

                    optimizer.step()

                    with torch.no_grad():
                        heliostat_group.activate_heliostats(
                            active_heliostats_mask=data_split.active_heliostats_mask_train, device=device
                        )

                        heliostat_group.align_surfaces_with_motor_positions(
                            motor_positions=data_split.motor_positions_train,
                            active_heliostats_mask=data_split.active_heliostats_mask_train,
                            device=device
                        )

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

                        flux_prediction_test=self.validate(
                            heliostat_group=heliostat_group,
                            data_split=data_split,
                            device=device
                        )

                    if isinstance(
                        scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        scheduler.step(loss.detach())
                    else:
                        scheduler.step()

                    if epoch % log_step == 0:
                        log.info(
                            f"Rank: {rank}, Epoch: {epoch}, Loss: {loss},",
                        )

                    loss_history_list.append(loss.detach().cpu().item())

                    # Early stopping when loss did not improve for a predefined number of epochs.
                    stop = early_stopper.step(loss.item())

                    if stop:
                        log.info(f"Early stopping at epoch {epoch}.")
                        break
                    
                    if epoch == 300:
                        self.plot_fluxes(
                            flux_measured=flux_measured.cpu().detach(),
                            flux_prediction_train=flux_prediction_train.cpu().detach(),
                            flux_prediction_test=flux_prediction_test.cpu().detach(),
                            data_split=data_split,
                            plot_name=f"geometry_{epoch}"
                        )

                    epoch += 1

                loss_history.append(
                    {
                        "total_loss": loss_history_list,
                    }
                )

                final_loss_per_heliostat[active_heliostats_mask!=0] = loss_per_heliostat

                log.info(f"Rank: {rank}, Kinematics reconstructed.")

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

            final_loss_history_all_groups: list[list[dict[str, list[float]]]] = [
                [] for _ in range(self.ddp_setup["world_size"])
            ]
            torch.distributed.all_gather_object(
                final_loss_history_all_groups, loss_history
            )

            log.info(f"Rank: {rank}, synchronized after kinematics reconstruction.")

        else:
            final_loss_history_all_groups = [loss_history]

        for heliostat_group in self.scenario.heliostat_field.heliostat_groups:
            heliostat_group.kinematics.rotation_deviation_parameters = heliostat_group.kinematics.rotation_deviation_parameters.detach()

        return final_loss_per_heliostat.detach().cpu(), final_loss_history_all_groups


    def _reconstruct_kinematics_parameters_with_raytracing(
        self,
        loss_definition: Loss,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, list[list[dict[str, list[float]]]]]:
        """
        Reconstruct the kinematics parameters using ray tracing.

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
        list[list[dict[str, list[float]]]]
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

        # Initialize final loss per heliostat, group offset table into global heliostat index space, and
        # per-group loss curves for this rank.
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
        loss_history: list[dict[str, list[float]]] = []

        # Iterate heliostat groups assigned to this rank.
        for heliostat_group_index in self.ddp_setup["groups_to_ranks_mapping"][rank]:
            # Parse calibration inputs for current group to obtain measured flux, incident ray directions, mask of
            # active heliostats, and target area indices.
            heliostat_group: HeliostatGroup = (
                self.scenario.heliostat_field.heliostat_groups[heliostat_group_index]
            )
            parser = cast(CalibrationDataParser, self.data[constants.data_parser])
            heliostat_mapping = cast(
                list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
                self.data[constants.heliostat_data_mapping],
            )
            (
                flux_measured,
                focal_spots_measured,
                incident_ray_directions,
                motor_positions,
                active_heliostats_mask,
                target_area_indices,
            ) = parser.parse_data_for_reconstruction(
                heliostat_data_mapping=heliostat_mapping,
                heliostat_group=heliostat_group,
                scenario=self.scenario,
                bitmap_resolution=self.bitmap_resolution,
                device=device,
            )

            if active_heliostats_mask.sum() > 0:
                data_split: TrainTestSplit = self.train_test_split(
                    active_heliostats_mask=active_heliostats_mask,
                    flux_measured=flux_measured,
                    focal_spots_measured=focal_spots_measured,
                    incident_ray_directions=incident_ray_directions,
                    motor_positions=motor_positions,
                    target_area_indices=target_area_indices,
                    test_fraction=0.25,
                    device=device
                )

                # Set up optimizer, scheduler, and early stopping.
                optimizer_params = [
                    {
                        "params": heliostat_group.kinematics.rotation_deviation_parameters.requires_grad_(),
                        "lr": self.optimizer_dict[
                            constants.initial_learning_rate_rotation_deviation
                        ],
                    }
                ]

                optimizer = torch.optim.Adam(optimizer_params)

                # Create a learning rate scheduler.
                scheduler_fn = getattr(
                    training,
                    self.scheduler_dict[constants.scheduler_type],
                )
                scheduler: LRScheduler = scheduler_fn(
                    optimizer=optimizer, parameters=self.scheduler_dict
                )

                # Set up early stopping.
                early_stopper = training.EarlyStopping(
                    window_size=self.optimizer_dict[constants.early_stopping_window],
                    patience=self.optimizer_dict[constants.early_stopping_patience],
                    min_improvement=self.optimizer_dict[constants.early_stopping_delta],
                    relative=True,
                )

                loss_history_list = []

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

                    # Activate heliostats.
                    heliostat_group.activate_heliostats(
                        active_heliostats_mask=data_split.active_heliostats_mask_train, device=device
                    )

                    # Align heliostats.
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

                    # Perform heliostat-based ray tracing.
                    flux_prediction_train, _, _, _ = ray_tracer.trace_rays(
                        incident_ray_directions=data_split.incident_ray_directions_train,
                        active_heliostats_mask=data_split.active_heliostats_mask_train,
                        target_area_indices=data_split.target_area_indices_train,
                        device=device,
                    )

                    sample_indices_for_local_rank = ray_tracer.get_sampler_indices()

                    # Compute loss from prediction vs. measured flux.
                    loss_per_sample = loss_definition(
                        prediction=flux_prediction_train,
                        ground_truth=data_split.flux_measured_train[
                            sample_indices_for_local_rank
                        ],
                        target_area_indices=data_split.target_area_indices_train[
                            sample_indices_for_local_rank
                        ],
                        reduction_dimensions=(1,2),
                        device=device,
                    )

                    loss_per_heliostat = reduce_loss_per_sample(
                        loss_per_sample=loss_per_sample,
                        number_of_samples_per_heliostat=data_split.number_of_train_samples,
                        reduction=partial(torch.median, dim=1)
                    )

                    loss = torch.mean(loss_per_heliostat)

                    loss.backward()

                    if self.ddp_setup["is_nested"]:
                        # Reduce gradients within each heliostat group.
                        for param_group in optimizer.param_groups:
                            for param in param_group["params"]:
                                if param.grad is not None:
                                    param.grad = (
                                        torch.distributed.nn.functional.all_reduce(
                                            param.grad,
                                            op=torch.distributed.ReduceOp.SUM,
                                            group=self.ddp_setup["process_subgroup"],
                                        )
                                    )
                                    param.grad /= self.ddp_setup[
                                        "heliostat_group_world_size"
                                    ]

                    optimizer.step()

                    with torch.no_grad():
                        flux_prediction_test=self.validate(
                            heliostat_group=heliostat_group,
                            data_split=data_split,
                            device=device
                        )

                    if isinstance(
                        scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        scheduler.step(loss.detach())
                    else:
                        scheduler.step()

                    if epoch % log_step == 0:
                        log.info(
                            f"Rank: {rank}, Epoch: {epoch}, Loss: {loss},",
                        )

                    loss_history_list.append(loss.detach().cpu().item())

                    # Early stopping when loss did not improve for a predefined number of epochs.
                    stop = early_stopper.step(loss.item())

                    if stop:
                        log.info(f"Early stopping at epoch {epoch}.")
                        break
                    
                    if epoch == 300:
                        self.plot_fluxes(
                            flux_measured=flux_measured.cpu().detach(),
                            flux_prediction_train=flux_prediction_train.cpu().detach(),
                            flux_prediction_test=flux_prediction_test.cpu().detach(),
                            data_split=data_split,
                            plot_name=f"raytracing_{epoch}"
                        )

                    epoch += 1

                loss_history.append(
                    {
                        "total_loss": loss_history_list,
                    }
                )

                local_indices = (
                    sample_indices_for_local_rank[::data_split.number_of_train_samples]
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

            final_loss_history_all_groups: list[list[dict[str, list[float]]]] = [
                [] for _ in range(self.ddp_setup["world_size"])
            ]
            torch.distributed.all_gather_object(
                final_loss_history_all_groups, loss_history
            )

            log.info(f"Rank: {rank}, synchronized after kinematics reconstruction.")

        else:
            final_loss_history_all_groups = [loss_history]

        for heliostat_group in self.scenario.heliostat_field.heliostat_groups:
            heliostat_group.kinematics.rotation_deviation_parameters = heliostat_group.kinematics.rotation_deviation_parameters.detach()

        return final_loss_per_heliostat.detach().cpu(), final_loss_history_all_groups
