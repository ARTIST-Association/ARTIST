import logging
import pathlib
from typing import Any, cast

import torch
from torch.optim.lr_scheduler import LRScheduler

from artist.core import core_utils, learning_rate_schedulers
from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.core.loss_functions import Loss
from artist.data_parser.calibration_data_parser import CalibrationDataParser
from artist.field.heliostat_group import HeliostatGroup
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, index_mapping, utils
from artist.util.environment_setup import DdpSetup, get_device

log = logging.getLogger(__name__)
"""A logger for the kinematic reconstructor."""


class KinematicsReconstructor:
    """
    An optimizer used to reconstruct real-world kinematics deviation parameters.

    The kinematics reconstructor learns kinematics parameters. These parameters are
    specific to a certain kinematics type and can for example include the four
    kinematics rotation deviation parameters as well as the two initial actuator parameters
    for each actuator of a rigid body kinematics.

    Attributes
    ----------
    ddp_setup : DdpSetup
        Information about the distributed environment, process_groups, devices, ranks, world_size, heliostat group to ranks mapping.
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
        The reconstruction method. Currently only reconstruction via ray tracing is available.

    Note
    ----
    Each heliostat selected for reconstruction needs to have the same amount of samples as all others.

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
        reconstruction_method: str = config_dictionary.kinematics_reconstruction_raytracing,
    ) -> None:
        """
        Initialize the kinematics optimizer.

        Parameters
        ----------
        ddp_setup : DdpSetup
            Information about the distributed environment, process_groups, devices, ranks, world_size, heliostat group to ranks mapping.
        scenario : Scenario
            The scenario.
        data : dict[str, CalibrationDataParser | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]]
            Data parser and the mapping of heliostat name and calibration data.
        optimization_configuration : dict[str, Any]
            Parameters for the optimizer, learning rate scheduler, regularizers and early stopping.
        dni : float | None
            Direct normal irradiance in W/m^2 (default is None which leads to a ray magnitude of 1.0).
        reconstruction_method : str
            Reconstruction method. Currently only reconstruction via ray tracing is available (default is ray_tracing).
        """
        rank = ddp_setup[config_dictionary.rank]  # type:ignore
        if rank == 0:
            log.info("Create a kinematics reconstructor.")

        self.ddp_setup = ddp_setup
        self.scenario = scenario
        self.data = data
        self.optimizer_dict = optimization_configuration[config_dictionary.optimization]
        self.scheduler_dict = optimization_configuration[config_dictionary.scheduler]
        self.dni = dni

        if reconstruction_method in [
            config_dictionary.kinematics_reconstruction_raytracing,
            config_dictionary.kinematics_reconstruction_motor_positions,
        ]:
            self.reconstruction_method = reconstruction_method
        else:
            raise ValueError(
                f"The kinematics reconstruction method {reconstruction_method} is not recognized. Please select another reconstruction method and try again!"
            )

    def reconstruct_kinematics(
        self,
        loss_definition: Loss,
        device: torch.device | None = None,
    ) -> torch.Tensor:
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
            Tensor of shape [total_number_of_heliostats_in_scenario].
        """
        device = get_device(device=device)

        if (
            self.reconstruction_method
            == config_dictionary.kinematics_reconstruction_raytracing
        ):
            loss = self._reconstruct_kinematics_parameters_with_raytracing(
                loss_definition=loss_definition,
                device=device,
            )

        return loss

    def _reconstruct_kinematics_parameters_with_raytracing(
        self,
        loss_definition: Loss,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Reconstruct the kinematics parameters using ray tracing.

        This reconstruction method optimizes the kinematics parameters by extracting the focal points
        of calibration images and using heliostat-tracing.

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
            Tensor of shape [total_number_of_heliostats_in_scenario].
        """
        device = get_device(device=device)

        rank = self.ddp_setup[config_dictionary.rank]  # type: ignore

        if rank == 0:
            log.info("Beginning kinematics reconstruction with ray tracing.")

        final_loss_per_heliostat = torch.full(
            (self.scenario.heliostat_field.number_of_heliostats_per_group.sum(),),
            torch.inf,
            device=device,
        )
        final_loss_start_indices = torch.cat(
            [
                torch.tensor([0], device=device),
                self.scenario.heliostat_field.number_of_heliostats_per_group.cumsum(
                    index_mapping.heliostat_dimension
                ),
            ]
        )

        for heliostat_group_index in self.ddp_setup[
            config_dictionary.groups_to_ranks_mapping  # type: ignore
        ][rank]:
            heliostat_group: HeliostatGroup = (
                self.scenario.heliostat_field.heliostat_groups[heliostat_group_index]
            )
            parser = cast(
                CalibrationDataParser, self.data[config_dictionary.data_parser]
            )
            heliostat_mapping = cast(
                list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
                self.data[config_dictionary.heliostat_data_mapping],
            )
            (
                flux_measured,
                _,
                incident_ray_directions,
                _,
                active_heliostats_mask,
                target_area_indices,
            ) = parser.parse_data_for_reconstruction(
                heliostat_data_mapping=heliostat_mapping,
                heliostat_group=heliostat_group,
                scenario=self.scenario,
                device=device,
            )

            if active_heliostats_mask.sum() > 0:
                # Calculate focal spot from measured flux.
                focal_spots_bitmap_coordinates = utils.get_center_of_mass(
                    bitmaps=flux_measured, device=device
                )
                focal_spots_measured = utils.bitmap_coordinates_to_target_coordinates(
                    bitmap_coordinates=focal_spots_bitmap_coordinates,
                    bitmap_resolution=torch.tensor([256, 256], device=device),
                    solar_tower=self.scenario.solar_tower,
                    target_area_indices=target_area_indices,
                    device=device,
                )

                # Reparametrize optimizable actuator parameters.
                initial_actuator_params = (
                    heliostat_group.kinematics.actuators.optimizable_parameters.detach()
                )

                angle_mean = initial_actuator_params[
                    :, index_mapping.actuator_params_initial_angle
                ].mean()
                angle_std = (
                    initial_actuator_params[
                        :, index_mapping.actuator_params_initial_angle
                    ]
                    .std()
                    .clamp(min=1e-3)
                )

                stroke_mean = initial_actuator_params[
                    :, index_mapping.actuator_params_initial_stroke_length
                ].mean()
                stroke_std = (
                    initial_actuator_params[
                        :, index_mapping.actuator_params_initial_stroke_length
                    ]
                    .std()
                    .clamp(min=1e-3)
                )

                angle_normalized = (
                    initial_actuator_params[
                        :, index_mapping.actuator_params_initial_angle
                    ]
                    - angle_mean
                ) / angle_std
                stroke_length_normalized = (
                    initial_actuator_params[
                        :, index_mapping.actuator_params_initial_stroke_length
                    ]
                    - stroke_mean
                ) / stroke_std

                delta_angle = torch.zeros_like(angle_normalized, requires_grad=True)
                delta_stroke = torch.zeros_like(
                    stroke_length_normalized, requires_grad=True
                )

                optimizer = torch.optim.Adam(
                    [
                        {
                            "params": heliostat_group.kinematics.rotation_deviation_parameters.requires_grad_(),
                            "lr": self.optimizer_dict[
                                config_dictionary.initial_learning_rate_rotation_deviation
                            ],
                        },
                        {
                            "params": delta_angle,
                            "lr": self.optimizer_dict[
                                config_dictionary.initial_learning_rate_initial_angles
                            ],
                        },
                        {
                            "params": delta_stroke,
                            "lr": self.optimizer_dict[
                                config_dictionary.initial_learning_rate_initial_stroke_length
                            ],
                        },
                    ]
                )

                # Create a learning rate scheduler.
                scheduler_fn = getattr(
                    learning_rate_schedulers,
                    self.scheduler_dict[config_dictionary.scheduler_type],
                )
                scheduler: LRScheduler = scheduler_fn(
                    optimizer=optimizer, parameters=self.scheduler_dict
                )

                # Set up early stopping.
                early_stopper = learning_rate_schedulers.EarlyStopping(
                    window_size=self.optimizer_dict[
                        config_dictionary.early_stopping_window
                    ],
                    patience=self.optimizer_dict[
                        config_dictionary.early_stopping_patience
                    ],
                    min_improvement=self.optimizer_dict[
                        config_dictionary.early_stopping_delta
                    ],
                    relative=True,
                )

                loss_history_list = []

                # Start the optimization.
                loss = torch.inf
                epoch = 0
                log_step = (
                    self.optimizer_dict[config_dictionary.max_epoch]
                    if self.optimizer_dict[config_dictionary.log_step] == 0
                    else self.optimizer_dict[config_dictionary.log_step]
                )
                while (
                    loss > float(self.optimizer_dict[config_dictionary.tolerance])
                    and epoch <= self.optimizer_dict[config_dictionary.max_epoch]
                ):
                    optimizer.zero_grad()

                    # Get actuator parameters from reparametrized version.
                    actuator_params = torch.cat(
                        [
                            ((angle_normalized + delta_angle) * angle_std + angle_mean)[
                                :, None, :
                            ],
                            (
                                (stroke_length_normalized + delta_stroke) * stroke_std
                                + stroke_mean
                            )[:, None, :],
                        ],
                        dim=-1,
                    ).view_as(
                        heliostat_group.kinematics.actuators.optimizable_parameters
                    )
                    heliostat_group.kinematics.actuators.optimizable_parameters = (
                        actuator_params
                    )

                    # Activate heliostats.
                    heliostat_group.activate_heliostats(
                        active_heliostats_mask=active_heliostats_mask, device=device
                    )

                    # Align heliostats.
                    heliostat_group.align_surfaces_with_incident_ray_directions(
                        aim_points=self.scenario.solar_tower.get_centers_of_target_areas(
                            target_area_indices=target_area_indices, device=device
                        ),
                        incident_ray_directions=incident_ray_directions,
                        active_heliostats_mask=active_heliostats_mask,
                        device=device,
                    )

                    # Create a parallelized ray tracer. Blocking is always deactivated for this reconstruction.
                    ray_tracer = HeliostatRayTracer(
                        scenario=self.scenario,
                        heliostat_group=heliostat_group,
                        blocking_active=False,
                        world_size=self.ddp_setup[
                            config_dictionary.heliostat_group_world_size  # type: ignore
                        ],
                        rank=self.ddp_setup[config_dictionary.heliostat_group_rank],  # type: ignore
                        batch_size=self.optimizer_dict[config_dictionary.batch_size],
                        random_seed=self.ddp_setup[
                            config_dictionary.heliostat_group_rank  # type: ignore
                        ],
                        dni=self.dni,
                    )

                    # Perform heliostat-based ray tracing.
                    flux_distributions, _, _, _ = ray_tracer.trace_rays(
                        incident_ray_directions=incident_ray_directions,
                        active_heliostats_mask=active_heliostats_mask,
                        target_area_indices=target_area_indices,
                        device=device,
                    )

                    sample_indices_for_local_rank = ray_tracer.get_sampler_indices()

                    loss_per_sample = loss_definition(
                        prediction=flux_distributions,
                        ground_truth=focal_spots_measured[
                            sample_indices_for_local_rank
                        ],
                        target_area_indices=target_area_indices[
                            sample_indices_for_local_rank
                        ],
                        reduction_dimensions=(index_mapping.focal_spots,),
                        device=device,
                    )

                    number_of_samples_per_heliostat = int(
                        heliostat_group.active_heliostats_mask.sum()
                        / (heliostat_group.active_heliostats_mask > 0).sum()
                    )

                    loss_per_heliostat = core_utils.mean_loss_per_heliostat(
                        loss_per_sample=loss_per_sample,
                        number_of_samples_per_heliostat=number_of_samples_per_heliostat,
                    )

                    loss = loss_per_heliostat.mean()

                    loss.backward()

                    if self.ddp_setup[config_dictionary.is_nested]:  # type: ignore
                        # Reduce gradients within each heliostat group.
                        for param_group in optimizer.param_groups:
                            for param in param_group["params"]:
                                if param.grad is not None:
                                    param.grad = (
                                        torch.distributed.nn.functional.all_reduce(
                                            param.grad,
                                            op=torch.distributed.ReduceOp.SUM,
                                            group=self.ddp_setup[
                                                config_dictionary.process_subgroup  # type: ignore
                                            ],
                                        )
                                    )
                                    param.grad /= self.ddp_setup[
                                        config_dictionary.heliostat_group_world_size  # type: ignore
                                    ]

                    optimizer.step()
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
                    stop = early_stopper.step(loss)

                    if stop:
                        log.info(f"Early stopping at epoch {epoch}.")
                        break

                    epoch += 1

                loss_history = {
                    "total_loss": loss_history_list,
                }

                local_indices = (
                    sample_indices_for_local_rank[::number_of_samples_per_heliostat]
                    // number_of_samples_per_heliostat
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

        if self.ddp_setup[config_dictionary.is_distributed]:  # type: ignore
            for index, heliostat_group in enumerate(
                self.scenario.heliostat_field.heliostat_groups
            ):
                source = self.ddp_setup[config_dictionary.ranks_to_groups_mapping][  # type: ignore
                    index
                ]
                torch.distributed.broadcast(
                    heliostat_group.kinematics.rotation_deviation_parameters,
                    src=source[index_mapping.first_rank_from_group],
                )
                torch.distributed.broadcast(
                    heliostat_group.kinematics.actuators.optimizable_parameters,
                    src=source[index_mapping.first_rank_from_group],
                )
            torch.distributed.all_reduce(
                final_loss_per_heliostat, op=torch.distributed.ReduceOp.MIN
            )

            log.info(f"Rank: {rank}, synchronized after kinematics reconstruction.")

        return final_loss_per_heliostat.detach().cpu(), loss_history
