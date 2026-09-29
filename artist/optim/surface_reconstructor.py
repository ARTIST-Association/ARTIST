import logging
import pathlib
from functools import partial
from typing import Any, cast

import torch
from torch.optim.lr_scheduler import LRScheduler

from artist.field.heliostat_group import HeliostatGroup
from artist.flux import bitmap
from artist.io.calibration_parser import CalibrationDataParser
from artist.nurbs.surfaces import NURBSSurfaces
from artist.nurbs.utils import create_nurbs_evaluation_grid
from artist.optim import training
from artist.optim.loss import KLDivergenceLoss, Loss, PixelLoss, reduce_loss_per_sample
from artist.optim.regularizers import IdealSurfaceRegularizer, SmoothnessRegularizer
from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer
from artist.scenario.scenario import Scenario
from artist.util import (
    constants,
    indices,
)
from artist.util.env import DdpSetup, get_device

log = logging.getLogger(__name__)
"""A logger for the surface reconstructor."""


class SurfaceReconstructor:
    """
    An optimizer used to reconstruct surfaces using NURBS and measured flux distributions.

    The surface reconstructor learns a surface representation from measured flux density
    distributions. The optimizable parameters for this optimization process are the
    NURBS control points.
    The reconstruction loss is defined by the loss between the flux density predictions and measurements.
    Further, the reconstruction is constrained by flux integral constraints to preserve energy in the reconstructed
    surfaces. There are also optional regularizers to keep the NURBS control points close to the ideal
    surface and smooth.

    Attributes
    ----------
    ddp_setup : DdpSetup
        Information about the distributed environment, process groups, devices, ranks, world size,
        and heliostat-group-to-ranks mapping.
    scenario : Scenario
        The scenario.
    data : dict[str, CalibrationDataParser | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]]
        The data parser and the mapping of heliostat name and calibration data.
    optimizer_dict : dict[str, Any]
        The parameters for the optimization.
    scheduler_dict : dict[str, Any]
        The parameters for the scheduler.
    constraint_dict : dict[str, Any]
        The parameters for the constraints.
    number_of_surface_points : torch.Tensor
        The number of surface points of the reconstructed surfaces.
        Shape is ``[2]``.
    dni : float | None
        Direct normal irradiance in W/m² used to scale the ray-traced flux. If None, the
        ``HeliostatRayTracer`` uses its own default.
    bitmap_resolution : torch.Tensor
        The resolution of all bitmaps during reconstruction.
        Shape is ``[2]``.
    epsilon : float | None
        Small numerical offset used to avoid division by zero in the energy constraint.
    plot_results : bool
        Create flux plots in the last epoch of training for each heliostat and all its samples (slow!).
    validation_loss_pixel : PixelLoss
        Pixel loss used for validation.
    validation_loss_kl_div : KLDivergenceLoss
        Kullback-Leibler divergence loss used for validation.

    Note
    ----
    Each heliostat selected for reconstruction needs to have the same number of samples as all others.

    Methods
    -------
    reconstruct_surfaces()
        Reconstruct NURBS surfaces from bitmaps.
    lock_control_points_on_outer_edges()
        Lock the u and v values of the control points on the outer edges of each facet.
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
        number_of_surface_points: torch.Tensor = torch.tensor([50, 50]),
        bitmap_resolution: torch.Tensor = torch.tensor([256, 256]),
        epsilon: float | None = 1e-12,
        plot_results: bool = False,
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize the surface reconstructor.

        Parameters
        ----------
        ddp_setup : DdpSetup
            Information about the distributed environment, process groups, devices, ranks, world size,
            and heliostat-group-to-ranks mapping.
        scenario : Scenario
            The scenario.
        data : dict[str, CalibrationDataParser | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]]
            The data parser and the mapping of heliostat name and calibration data.
        optimization_configuration : dict[str, Any]
            The parameters for the optimizer, learning rate scheduler, early stopping, and constraints.
        dni : float | None
            Direct normal irradiance in W/m² used to scale the ray-traced flux (default is None).
            If None, the ``HeliostatRayTracer`` uses its own default.
        number_of_surface_points : torch.Tensor
            The number of surface points of the reconstructed surfaces (default is ``torch.tensor([50, 50])``).
            Shape is ``[2]``.
        bitmap_resolution : torch.Tensor
            The resolution of all bitmaps during reconstruction (default is ``torch.tensor([256, 256])``).
            Shape is ``[2]``.
        epsilon : float | None
            Small numerical offset used to avoid division by zero in the energy constraint (default is 1e-12).
        plot_results : bool
            Create flux plots in the last epoch of training for each heliostat and all its samples (slow!, default is ``False``).
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        device = get_device(device=device)

        rank = ddp_setup["rank"]

        if rank == 0:
            log.info("Create a surface reconstructor.")

        self.ddp_setup = ddp_setup
        self.scenario = scenario
        self.data = data
        self.optimizer_dict = optimization_configuration[constants.optimization]
        self.scheduler_dict = optimization_configuration[constants.scheduler]
        self.constraint_dict = optimization_configuration[constants.constraints]
        self.number_of_surface_points = number_of_surface_points.to(device)
        self.dni = dni
        self.bitmap_resolution = bitmap_resolution.to(device)
        self.epsilon = epsilon
        self.plot_results = plot_results

        self.validation_loss_pixel = PixelLoss()
        self.validation_loss_kl_div = KLDivergenceLoss()

    def _validate(
        self,
        heliostat_group: HeliostatGroup,
        data_split: training.TrainTestSplit,
        evaluation_points: torch.Tensor,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Validate the surface reconstruction for a specified heliostat group on the test data.

        This validates a surface reconstruction (NURBS-based mirror surface shape + alignment) for a given heliostat group on the test set:

        - It builds NURBS surfaces from control points.
        - Samples surface points and normals.
        - Aligns those surfaces to incident rays and aim points.
        - Runs ray tracing to predict flux on the receiver.
        - Compares predicted vs measured flux for the test samples and computes several test losses.

        It returns the predicted flux images (for the local rank’s test samples) and a dict with per‑heliostat test losses (pixel loss and KL divergence).

        Parameters
        ----------
        heliostat_group : HeliostatGroup
            Heliostat group to validate.
        data_split : training.TrainTestSplit
            Train/test split containing all test tensors and metadata.
        evaluation_points : torch.Tensor
            Evaluation points for the NURBS surface sampling.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        dict[str, torch.Tensor]
            Test losses per sample.
        """
        device = get_device(device=device)

        heliostat_group.activate_heliostats(
            active_heliostats_mask=data_split.active_heliostats_mask_test,
            device=device,
        )

        nurbs_surfaces = NURBSSurfaces(
            degrees=heliostat_group.nurbs_degrees,
            control_points=heliostat_group.active_nurbs_control_points,
            device=device,
        )

        (
            new_surface_points,
            new_surface_normals,
        ) = nurbs_surfaces.calculate_surface_points_and_normals(
            evaluation_points=evaluation_points[data_split.test_indices],
            canting=heliostat_group.active_canting,
            facet_translations=heliostat_group.active_facet_translations,
            device=device,
        )

        heliostat_group.active_surface_points = new_surface_points.reshape(
            heliostat_group.active_surface_points.shape[indices.heliostat_dimension],
            -1,
            4,
        )
        heliostat_group.active_surface_normals = new_surface_normals.reshape(
            heliostat_group.active_surface_normals.shape[indices.heliostat_dimension],
            -1,
            4,
        )

        heliostat_group.align_surfaces_with_incident_ray_directions(
            aim_points=self.scenario.solar_tower.get_centers_of_target_areas(
                target_area_indices=data_split.target_area_indices_test, device=device
            ),
            incident_ray_directions=data_split.incident_ray_directions_test,
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

        cropped_flux_distributions = bitmap.crop_flux_distributions_around_center(
            flux_distributions=flux_prediction,
            solar_tower=self.scenario.solar_tower,
            target_area_indices=data_split.target_area_indices_test,
            device=device,
        )

        indices_for_local_rank = ray_tracer.get_sampler_indices()

        loss_pixel_per_sample = self.validation_loss_pixel(
            prediction=cropped_flux_distributions,
            ground_truth=data_split.flux_measured_test[indices_for_local_rank],
            reduction_dimensions=(
                1,
                2,
            ),
        )
        loss_kl_div_per_sample = self.validation_loss_kl_div(
            prediction=cropped_flux_distributions,
            ground_truth=data_split.flux_measured_test[indices_for_local_rank],
            reduction_dimensions=(
                1,
                2,
            ),
        )

        test_loss_pixel = reduce_loss_per_sample(
            loss_per_sample=loss_pixel_per_sample,
            number_of_samples_per_heliostat=data_split.number_of_test_samples,
            reduction=partial(torch.mean, dim=-1),
        )
        test_loss_kl_div = reduce_loss_per_sample(
            loss_per_sample=loss_kl_div_per_sample,
            number_of_samples_per_heliostat=data_split.number_of_test_samples,
            reduction=partial(torch.mean, dim=-1),
        )

        log.info(
            "pixel mean: %.5f, kl-div mean: %.5f",
            torch.mean(test_loss_pixel).item(),
            torch.mean(test_loss_kl_div).item(),
        )

        return {
            "pixel_loss": test_loss_pixel,
            "kl_div": test_loss_kl_div,
        }

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
        # Final per-heliostat loss container (global over all groups), initialized with + inf.
        final_loss_per_heliostat = torch.full(
            (self.scenario.heliostat_field.number_of_heliostats_per_group.sum(),),
            torch.inf,
            device=device,
        )

        # Prefix sums to map group-local heliostat indices to global heliostat indices.
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
        self, heliostat_group: HeliostatGroup, device: torch.device
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
        parser = cast(CalibrationDataParser, self.data[constants.data_parser])
        heliostat_mapping = cast(
            list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
            self.data[constants.heliostat_data_mapping],
        )
        return parser.parse_data_for_reconstruction(
            heliostat_data_mapping=heliostat_mapping,
            heliostat_group=heliostat_group,
            scenario=self.scenario,
            bitmap_resolution=self.bitmap_resolution,
            device=device,
        )

    def _create_evaluation_grid_and_reference_points(
        self,
        heliostat_group: HeliostatGroup,
        active_heliostats_mask: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create the NURBS evaluation grid and a frozen copy of the control points.

        Parameters
        ----------
        heliostat_group : HeliostatGroup
            The heliostat group whose surfaces are reconstructed.
        active_heliostats_mask : torch.Tensor
            Mask for active samples available per heliostat.
        device : torch.device
            The device on which to perform computations or load tensors and models.

        Returns
        -------
        torch.Tensor
            The evaluation points for the NURBS surface sampling.
        torch.Tensor
            A frozen copy of the original control points used by the regularizers.
        """
        evaluation_points = (
            create_nurbs_evaluation_grid(
                number_of_evaluation_points=self.number_of_surface_points,
                device=device,
            )
            .unsqueeze(indices.heliostat_dimension)
            .unsqueeze(indices.facet_index_unbatched)
            .expand(
                int(active_heliostats_mask.sum()),
                heliostat_group.number_of_facets_per_heliostat,
                -1,
                -1,
            )
        )
        # Keep a frozen copy of original control points for regularization terms.
        with torch.no_grad():
            original_control_points = heliostat_group.nurbs_control_points[
                active_heliostats_mask > 0
            ].clone()

        return evaluation_points, original_control_points

    def _setup_optimizer_scheduler_early_stopping(
        self, heliostat_group: HeliostatGroup
    ) -> tuple[torch.optim.Optimizer, LRScheduler, training.EarlyStopping]:
        """
        Create the optimizer, learning rate scheduler, and early stopping for a group.

        The optimizer learns the NURBS control points of the group's surfaces.

        Parameters
        ----------
        heliostat_group : HeliostatGroup
            The heliostat group whose surfaces are reconstructed.

        Returns
        -------
        torch.optim.Optimizer
            The Adam optimizer over the NURBS control points.
        LRScheduler
            The learning rate scheduler.
        training.EarlyStopping
            The early stopping monitor.
        """
        # Create the optimizer.
        optimizer = torch.optim.Adam(
            [heliostat_group.nurbs_control_points.requires_grad_()],
            lr=float(self.optimizer_dict[constants.initial_learning_rate]),
        )

        # Create a learning rate scheduler.
        scheduler_fn = getattr(
            training,
            self.scheduler_dict[constants.scheduler_type],
        )
        scheduler: LRScheduler = scheduler_fn(
            optimizer=optimizer, parameters=self.scheduler_dict
        )

        # Set up early stopping on stagnating loss.
        early_stopper = training.EarlyStopping(
            window_size=self.optimizer_dict[constants.early_stopping_window],
            patience=self.optimizer_dict[constants.early_stopping_patience],
            min_improvement=self.optimizer_dict[constants.early_stopping_delta],
            relative=True,
        )

        return optimizer, scheduler, early_stopper

    def _predict_flux(
        self,
        heliostat_group: HeliostatGroup,
        evaluation_points: torch.Tensor,
        data_split: training.TrainTestSplit,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict the cropped flux distributions for the training samples of one epoch.

        The current NURBS surfaces are built and sampled, aligned toward the targets, ray
        traced, and the resulting flux distributions are cropped around their center.

        Parameters
        ----------
        heliostat_group : HeliostatGroup
            The heliostat group to reconstruct.
        evaluation_points : torch.Tensor
            Evaluation points for the NURBS surface sampling.
        data_split : training.TrainTestSplit
            Train/test split containing all training tensors and metadata.
        device : torch.device
            The device on which to perform computations or load tensors and models.

        Returns
        -------
        torch.Tensor
            The cropped predicted flux distributions of the training samples.
        torch.Tensor
            The sample indices processed on the local rank.
        torch.Tensor
            The local heliostat indices processed on the local rank.
        """
        # Activate heliostats.
        heliostat_group.activate_heliostats(
            active_heliostats_mask=data_split.active_heliostats_mask_train,
            device=device,
        )

        # Build NURBS surface from current control points.
        nurbs_surfaces = NURBSSurfaces(
            degrees=heliostat_group.nurbs_degrees,
            control_points=heliostat_group.active_nurbs_control_points,
            device=device,
        )

        # Calculate surface points and normals.
        (
            new_surface_points,
            new_surface_normals,
        ) = nurbs_surfaces.calculate_surface_points_and_normals(
            evaluation_points=evaluation_points[data_split.train_indices],
            canting=heliostat_group.active_canting,
            facet_translations=heliostat_group.active_facet_translations,
            device=device,
        )

        # Flatten faceted tensors to the shape expected by alignment module and ray tracer.
        heliostat_group.active_surface_points = new_surface_points.reshape(
            heliostat_group.active_surface_points.shape[indices.heliostat_dimension],
            -1,
            4,
        )
        heliostat_group.active_surface_normals = new_surface_normals.reshape(
            heliostat_group.active_surface_normals.shape[indices.heliostat_dimension],
            -1,
            4,
        )

        # Align heliostat surfaces toward target under current incident ray directions.
        heliostat_group.align_surfaces_with_incident_ray_directions(
            aim_points=self.scenario.solar_tower.get_centers_of_target_areas(
                target_area_indices=data_split.target_area_indices_train,
                device=device,
            ),
            incident_ray_directions=data_split.incident_ray_directions_train,
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
            bitmap_resolution=self.bitmap_resolution,
            dni=self.dni,
        )

        # Perform heliostat-based ray tracing to obtain simulated flux from current reconstructed surfaces.
        flux_prediction_train, _, _, _ = ray_tracer.trace_rays(
            incident_ray_directions=data_split.incident_ray_directions_train,
            active_heliostats_mask=data_split.active_heliostats_mask_train,
            target_area_indices=data_split.target_area_indices_train,
            device=device,
        )

        # Crop predictions around center before comparing to measurements.
        cropped_flux_predictions = bitmap.crop_flux_distributions_around_center(
            flux_distributions=flux_prediction_train,
            solar_tower=self.scenario.solar_tower,
            target_area_indices=data_split.target_area_indices_train,
            device=device,
        )

        sample_indices_for_local_rank = ray_tracer.get_sampler_indices()
        local_indices = (
            sample_indices_for_local_rank[:: data_split.number_of_train_samples]
            // data_split.number_of_train_samples
        )

        return cropped_flux_predictions, sample_indices_for_local_rank, local_indices

    def _compute_flux_integral_constraint(
        self,
        cropped_flux_predictions: torch.Tensor,
        flux_integrals_reference: torch.Tensor,
        lambda_flux_integral: torch.Tensor | float,
        rho_flux_integral: float,
        energy_tolerance: float,
        data_split: training.TrainTestSplit,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the Augmented-Lagrangian flux integral constraint.

        The constraint ensures that the flux integral is conserved, i.e., intensity does not
        get lost relative to the reference captured in the first epoch.

        Parameters
        ----------
        cropped_flux_predictions : torch.Tensor
            The cropped predicted flux distributions of the training samples.
        flux_integrals_reference : torch.Tensor
            The reference flux integrals captured in the first epoch.
        lambda_flux_integral : torch.Tensor | float
            The current Augmented-Lagrangian multiplier.
        rho_flux_integral : float
            The penalty parameter of the constraint.
        energy_tolerance : float
            The tolerance below which the flux integral may drop without penalty.
        data_split : training.TrainTestSplit
            Train/test split containing all training tensors and metadata.

        Returns
        -------
        torch.Tensor
            The flux integral constraint per heliostat.
        torch.Tensor
            The relative differences of the flux integrals per sample.
        torch.Tensor
            The clamped flux constraint per heliostat.
        """
        flux_integrals_relative_differences = (
            cropped_flux_predictions.sum(
                dim=(indices.batched_bitmap_e, indices.batched_bitmap_u)
            )
            - flux_integrals_reference
        ) / (flux_integrals_reference + torch.tensor(self.epsilon))
        flux_constraint_per_sample = torch.clamp(
            -energy_tolerance - flux_integrals_relative_differences, min=0.0
        )
        flux_constraint_per_heliostat = reduce_loss_per_sample(
            loss_per_sample=flux_constraint_per_sample,
            number_of_samples_per_heliostat=data_split.number_of_train_samples,
            reduction=partial(torch.mean, dim=-1),
        )
        flux_integrals_constraint = (
            lambda_flux_integral * flux_constraint_per_heliostat
            + 0.5 * rho_flux_integral * flux_constraint_per_heliostat**2
        )
        return (
            flux_integrals_constraint,
            flux_integrals_relative_differences,
            flux_constraint_per_heliostat,
        )

    def _compute_regularization_terms(
        self,
        heliostat_group: HeliostatGroup,
        original_control_points: torch.Tensor,
        local_indices: torch.Tensor,
        data_split: training.TrainTestSplit,
        flux_loss_per_heliostat: torch.Tensor,
        smoothness_regularizer: SmoothnessRegularizer,
        ideal_surface_regularizer: IdealSurfaceRegularizer,
        weight_smoothness: float,
        weight_ideal_surface: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the smoothness and ideal-surface regularization terms.

        The regularizers keep the reconstructed surface smooth and close to the ideal/original
        surface. Their magnitudes are dynamically balanced relative to the data term.

        Parameters
        ----------
        heliostat_group : HeliostatGroup
            The heliostat group whose surfaces are reconstructed.
        original_control_points : torch.Tensor
            A frozen copy of the original control points.
        local_indices : torch.Tensor
            The local heliostat indices processed on the local rank.
        data_split : training.TrainTestSplit
            Train/test split containing all training tensors and metadata.
        flux_loss_per_heliostat : torch.Tensor
            The flux loss per heliostat used for the dynamic balancing.
        smoothness_regularizer : SmoothnessRegularizer
            The smoothness regularizer.
        ideal_surface_regularizer : IdealSurfaceRegularizer
            The ideal-surface regularizer.
        weight_smoothness : float
            The weight of the smoothness regularizer.
        weight_ideal_surface : float
            The weight of the ideal-surface regularizer.
        device : torch.device
            The device on which to perform computations or load tensors and models.

        Returns
        -------
        torch.Tensor
            The dynamic balancing factor ``alpha`` for the smoothness term.
        torch.Tensor
            The smoothness loss per heliostat.
        torch.Tensor
            The dynamic balancing factor ``beta`` for the ideal-surface term.
        torch.Tensor
            The ideal-surface loss per heliostat.
        """
        smoothness_loss_per_heliostat = torch.zeros_like(
            flux_loss_per_heliostat, device=device
        )
        ideal_surface_loss_per_heliostat = torch.zeros_like(
            flux_loss_per_heliostat, device=device
        )
        if weight_smoothness > 0:
            smoothness_loss_per_heliostat = smoothness_regularizer(
                current_control_points=heliostat_group.active_nurbs_control_points[
                    :: data_split.number_of_train_samples
                ][local_indices],
                original_control_points=original_control_points[local_indices],
                device=device,
            )
        if weight_ideal_surface > 0:
            ideal_surface_loss_per_heliostat = ideal_surface_regularizer(
                current_control_points=heliostat_group.active_nurbs_control_points[
                    :: data_split.number_of_train_samples
                ][local_indices],
                original_control_points=original_control_points[local_indices],
                device=device,
            )
        # Dynamic balancing of regularization magnitudes relative to data term.
        alpha = (
            weight_smoothness
            * flux_loss_per_heliostat.mean()
            / (smoothness_loss_per_heliostat.mean() + torch.tensor(self.epsilon))
        )
        beta = (
            weight_ideal_surface
            * flux_loss_per_heliostat.mean()
            / (ideal_surface_loss_per_heliostat.mean() + torch.tensor(self.epsilon))
        )
        return (
            alpha,
            smoothness_loss_per_heliostat,
            beta,
            ideal_surface_loss_per_heliostat,
        )

    def _synchronize_and_lock_gradients(
        self, optimizer: torch.optim.Optimizer, device: torch.device
    ) -> None:
        """
        Synchronize gradients in nested-DDP mode and lock outer-edge control points.

        In nested distributed data parallel mode the gradients are averaged across the ranks
        that process the same heliostat group. The gradients of the outer-edge control points
        are then zeroed to preserve the rectangular surface shape.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer whose parameter gradients are synchronized and locked.
        device : torch.device
            The device on which to perform computations or load tensors and models.
        """
        # Nested-DDP gradient synchronization within heliostat-group subgroup.
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
                        param.grad /= self.ddp_setup["heliostat_group_world_size"]

        # Geometry-preserving constraint: Keep the surfaces in their original geometric shape by locking
        # the control points on the outer edges, i.e., zero/fix gradient on outer-edge control points.
        optimizer.param_groups[indices.optimizer_param_group_0]["params"][
            indices.optimizable_control_points
        ].grad = self.lock_control_points_on_outer_edges(
            gradients=optimizer.param_groups[indices.optimizer_param_group_0]["params"][
                indices.optimizable_control_points
            ].grad,
            device=device,
        )

    def _synchronize_reconstruction_across_ranks(
        self,
        final_loss_per_heliostat: torch.Tensor,
        loss_history: list[dict[str, list[float] | dict[str, torch.Tensor]]],
    ) -> list[list[dict[str, list[float] | dict[str, torch.Tensor]]]]:
        """
        Synchronize the reconstruction results across all distributed ranks.

        Broadcasts the reconstructed NURBS control points, reduces the final loss to its
        minimum across ranks, and gathers the loss histories of all ranks.

        Parameters
        ----------
        final_loss_per_heliostat : torch.Tensor
            The final loss per heliostat on the local rank.
            Shape is ``[total_number_of_heliostats_in_scenario]``.
        loss_history : list[dict[str, list[float] | dict[str, torch.Tensor]]]
            The local rank's loss histories per heliostat group.

        Returns
        -------
        list[list[dict[str, list[float] | dict[str, torch.Tensor]]]]
            Loss histories grouped by rank.
        """
        rank = self.ddp_setup["rank"]

        if self.ddp_setup["is_distributed"]:
            for index, heliostat_group in enumerate(
                self.scenario.heliostat_field.heliostat_groups
            ):
                source = self.ddp_setup["ranks_to_groups_mapping"][index]
                torch.distributed.broadcast(
                    heliostat_group.nurbs_control_points,
                    src=source[indices.first_rank_from_group],
                )
            torch.distributed.all_reduce(
                final_loss_per_heliostat, op=torch.distributed.ReduceOp.MIN
            )
            final_loss_history_all_groups: list[
                list[dict[str, list[float] | dict[str, torch.Tensor]]]
            ] = [[] for _ in range(self.ddp_setup["world_size"])]
            torch.distributed.all_gather_object(
                final_loss_history_all_groups, loss_history
            )

            log.info(f"Rank: {rank}, synchronized after surface reconstruction.")

        else:
            final_loss_history_all_groups = [loss_history]

        return final_loss_history_all_groups

    def reconstruct_surfaces(
        self,
        loss_definition: Loss,
        device: torch.device | None = None,
    ) -> tuple[
        torch.Tensor, list[list[dict[str, list[float] | dict[str, torch.Tensor]]]]
    ]:
        """
        Reconstruct NURBS surfaces from bitmaps.

        Parameters
        ----------
        loss_definition : Loss
            The definition of the loss function and pre-processing of the prediction.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The final reconstruction loss per heliostat, one entry per heliostat in the scenario.
            Shape is ``[total_number_of_heliostats_in_scenario]``.
        list[list[dict[str, list[float] | dict[str, torch.Tensor]]]]]
            Loss histories over epochs grouped by rank.

            - Outer list: one entry per rank.
            - Inner list: one entry per heliostat group processed on that rank.
            - Each group entry is a dict with keys:
              ``"total_loss"``, ``"flux_loss"``, ``"smoothness_regularizer"``,
              ``"ideal_regularizer"``, ``"flux_integral"``, and
              ``"flux_integral_constraint"``.
              Each value is a list of per-epoch scalar floats.

              In non-distributed mode, this is a single-rank container: ``[local_group_histories]``.
        """
        device = get_device(device=device)
        rank = self.ddp_setup["rank"]

        if rank == 0:
            log.info("Beginning surface reconstruction.")

        final_loss_per_heliostat, final_loss_start_indices = (
            self._initialize_reconstruction_bookkeeping(device=device)
        )

        # Rank-local history: one dict per processed heliostat group.
        loss_history: list[dict[str, list[float] | dict[str, torch.Tensor]]] = []

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
                heliostat_group=heliostat_group, device=device
            )

            # Skip groups with no active heliostats.
            if active_heliostats_mask.sum() > 0:
                data_split: training.TrainTestSplit = training.train_test_split(
                    active_heliostats_mask=active_heliostats_mask,
                    flux_measured=flux_measured,
                    focal_spots_measured=focal_spots_measured,
                    incident_ray_directions=incident_ray_directions,
                    motor_positions=motor_positions,
                    target_area_indices=target_area_indices,
                    device=device,
                )
                evaluation_points, original_control_points = (
                    self._create_evaluation_grid_and_reference_points(
                        heliostat_group=heliostat_group,
                        active_heliostats_mask=active_heliostats_mask,
                        device=device,
                    )
                )

                optimizer, scheduler, early_stopper = (
                    self._setup_optimizer_scheduler_early_stopping(
                        heliostat_group=heliostat_group
                    )
                )

                # Set up Augmented-Lagrangian constraint for energy conservation.
                flux_integrals_reference = torch.zeros_like(active_heliostats_mask)
                lambda_flux_integral = 0.0
                rho_flux_integral = self.constraint_dict[constants.rho_flux_integral]
                energy_tolerance = self.constraint_dict[constants.energy_tolerance]
                # Set up regularizers: Keep reconstructed surface smooth and close to ideal/original.
                ideal_surface_regularizer = IdealSurfaceRegularizer(
                    reduction_dimensions=(1,)
                )
                smoothness_regularizer = SmoothnessRegularizer(
                    reduction_dimensions=(1,)
                )
                weight_smoothness = self.constraint_dict[constants.weight_smoothness]
                weight_ideal_surface = self.constraint_dict[
                    constants.weight_ideal_surface
                ]

                # Set up per-epoch logging/history buffers.
                total_loss_history = []
                flux_loss_history = []
                flux_integral_history = []
                smoothness_history = []
                ideal_history = []
                flux_integral = []

                # Start the optimization.
                total_loss = torch.inf
                epoch = 0
                log_step = (
                    self.optimizer_dict[constants.max_epoch]
                    if self.optimizer_dict[constants.log_step] == 0
                    else self.optimizer_dict[constants.log_step]
                )
                while (
                    total_loss > float(self.optimizer_dict[constants.tolerance])
                    and epoch <= self.optimizer_dict[constants.max_epoch]
                ):
                    optimizer.zero_grad()

                    (
                        cropped_flux_predictions,
                        sample_indices_for_local_rank,
                        local_indices,
                    ) = self._predict_flux(
                        heliostat_group=heliostat_group,
                        evaluation_points=evaluation_points,
                        data_split=data_split,
                        device=device,
                    )

                    # Compute loss from prediction vs. measured flux.
                    flux_loss_per_sample = loss_definition(
                        prediction=cropped_flux_predictions,
                        ground_truth=data_split.flux_measured_train[
                            sample_indices_for_local_rank
                        ],
                        target_area_indices=data_split.target_area_indices_train[
                            sample_indices_for_local_rank
                        ],
                        reduction_dimensions=(
                            indices.batched_bitmap_e,
                            indices.batched_bitmap_u,
                        ),
                        device=device,
                    )

                    flux_loss_per_heliostat = reduce_loss_per_sample(
                        loss_per_sample=flux_loss_per_sample,
                        number_of_samples_per_heliostat=data_split.number_of_train_samples,
                        reduction=partial(torch.mean, dim=-1),
                    )

                    # Add Augmented-Lagrangian constraint to ensure that flux integral is conserved,
                    # i.e., intensity does not get lost.
                    if epoch == 0:
                        flux_integrals_reference = cropped_flux_predictions.sum(
                            dim=(indices.batched_bitmap_e, indices.batched_bitmap_u)
                        ).detach()
                    (
                        flux_integrals_constraint,
                        flux_integrals_relative_differences,
                        flux_constraint_per_heliostat,
                    ) = self._compute_flux_integral_constraint(
                        cropped_flux_predictions=cropped_flux_predictions,
                        flux_integrals_reference=flux_integrals_reference,
                        lambda_flux_integral=lambda_flux_integral,
                        rho_flux_integral=rho_flux_integral,
                        energy_tolerance=energy_tolerance,
                        data_split=data_split,
                    )

                    # Regularization terms.
                    (
                        alpha,
                        smoothness_loss_per_heliostat,
                        beta,
                        ideal_surface_loss_per_heliostat,
                    ) = self._compute_regularization_terms(
                        heliostat_group=heliostat_group,
                        original_control_points=original_control_points,
                        local_indices=local_indices,
                        data_split=data_split,
                        flux_loss_per_heliostat=flux_loss_per_heliostat,
                        smoothness_regularizer=smoothness_regularizer,
                        ideal_surface_regularizer=ideal_surface_regularizer,
                        weight_smoothness=weight_smoothness,
                        weight_ideal_surface=weight_ideal_surface,
                        device=device,
                    )

                    # Final per-heliostat loss
                    total_loss_per_heliostat = (
                        flux_loss_per_heliostat
                        + flux_integrals_constraint
                        + alpha * smoothness_loss_per_heliostat
                        + beta * ideal_surface_loss_per_heliostat
                    )

                    total_loss = torch.mean(total_loss_per_heliostat)

                    total_loss.backward()

                    # Update Augmented-Lagrangian multiplier.
                    with torch.no_grad():
                        lambda_flux_integral = torch.clamp(
                            lambda_flux_integral
                            + rho_flux_integral * flux_constraint_per_heliostat,
                            min=0.0,
                        )

                    self._synchronize_and_lock_gradients(
                        optimizer=optimizer, device=device
                    )

                    optimizer.step()
                    if isinstance(
                        scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        scheduler.step(total_loss.detach())
                    else:
                        scheduler.step()

                    is_last_epoch = (
                        epoch == self.optimizer_dict[constants.max_epoch] - 1
                    )
                    stop = early_stopper.step(total_loss.item())

                    if epoch % log_step == 0 or is_last_epoch or stop:
                        log.info(
                            f"Rank: {rank}, Epoch: {epoch}, Loss: {total_loss}",
                        )

                        with torch.no_grad():
                            test_loss = self._validate(
                                heliostat_group=heliostat_group,
                                data_split=data_split,
                                evaluation_points=evaluation_points,
                                device=device,
                            )

                    # Early stopping when loss did not improve for a predefined number of epochs.
                    if stop:
                        log.info(f"Early stopping at epoch {epoch}.")
                        break

                    total_loss_history.append(total_loss.detach().cpu().item())
                    flux_loss_history.append(
                        flux_loss_per_heliostat.mean().detach().cpu().item()
                    )
                    flux_integral.append(
                        flux_integrals_relative_differences.mean().detach().cpu().item()
                    )
                    smoothness_history.append(
                        (alpha * smoothness_loss_per_heliostat)
                        .mean()
                        .detach()
                        .cpu()
                        .item()
                    )
                    ideal_history.append(
                        (beta * ideal_surface_loss_per_heliostat)
                        .mean()
                        .detach()
                        .cpu()
                        .item()
                    )
                    flux_integral_history.append(
                        flux_integrals_constraint.mean().detach().cpu().item()
                    )

                    epoch += 1

                loss_history.append(
                    {
                        "total_loss": total_loss_history,
                        "flux_loss": flux_loss_history,
                        "smoothness_regularizer": smoothness_history,
                        "ideal_regularizer": ideal_history,
                        "flux_integral": flux_integral,
                        "flux_integral_constraint": flux_integral_history,
                        "test_loss": test_loss,
                    }
                )

                global_active_indices = torch.nonzero(
                    active_heliostats_mask != 0, as_tuple=True
                )[0]

                rank_active_indices_global = global_active_indices[local_indices]

                final_indices = (
                    rank_active_indices_global
                    + final_loss_start_indices[heliostat_group_index]
                )

                final_loss_per_heliostat[final_indices] = total_loss_per_heliostat

                log.info(f"Rank: {rank}, Surfaces reconstructed.")

        final_loss_history_all_groups = self._synchronize_reconstruction_across_ranks(
            final_loss_per_heliostat=final_loss_per_heliostat,
            loss_history=loss_history,
        )

        self.scenario.heliostat_field.update_surfaces(device=device)

        return final_loss_per_heliostat.detach().cpu(), final_loss_history_all_groups

    @staticmethod
    def lock_control_points_on_outer_edges(
        gradients: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Lock the u and v values of the control points on the outer edges of each facet.

        As the knots of the first and last knots on each facet have full multiplicity, the
        NURBS surfaces all start and end in control points. If the outer control points
        are not fixed in their u and v values, the reconstructed surfaces may not be
        rectangular anymore. To keep them rectangular, this function zeros the gradients
        of the u and v coordinates of all outer control points.

        Parameters
        ----------
        gradients : torch.Tensor
            The full control point gradient tensor for all active heliostats. Gradients on the
            outer edges will be zeroed; interior gradients are returned unchanged.
            Shape is ``[number_of_active_heliostats, number_of_facets_per_surface, number_of_control_points_u_direction, number_of_control_points_v_direction, 3]``.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The updated gradients.
            Shape is ``[number_of_active_heliostats, number_of_facets_per_surface, number_of_control_points_u_direction, number_of_control_points_v_direction, 3]``.
        """
        device = get_device(device=device)

        with torch.no_grad():
            fixed_gradients = gradients.clone()

            height = gradients.shape[indices.nurbs_control_points_u]
            width = gradients.shape[indices.nurbs_control_points_v]

            rows = (
                torch.arange(height, device=device)
                .unsqueeze(indices.unbatched_bitmap_u)
                .expand(height, width)
            )
            cols = (
                torch.arange(width, device=device)
                .unsqueeze(indices.unbatched_bitmap_e)
                .expand(height, width)
            )

            edge_mask = (
                (rows == 0) | (rows == height - 1) | (cols == 0) | (cols == width - 1)
            )

            fixed_gradients[:, :, :, :, : indices.z_coordinates] = torch.where(
                edge_mask.unsqueeze(indices.heliostat_dimension)
                .unsqueeze(indices.facet_index_unbatched)
                .unsqueeze(indices.nurbs_control_points),
                torch.tensor(0.0, device=device),
                gradients[:, :, :, :, : indices.z_coordinates],
            )

            return fixed_gradients
