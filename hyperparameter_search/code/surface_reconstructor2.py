import logging
import pathlib

import matplotlib.pyplot as plt
import torch

from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.data_loader import flux_distribution_loader, paint_loader
from artist.scenario.scenario import Scenario
from artist.util import utils
from artist.util.environment_setup import get_device
from artist.util.nurbs import NURBSSurfaces
from hyperparameter_search.code import helper

log = logging.getLogger(__name__)
"""A logger for the surface reconstructor."""


class SurfaceReconstructor2:
    """
    An optimizer used to reconstruct surfaces using NURBS and measured flux distributions.

    The surface reconstructor learns a surface representation from measured flux density
    distributions. The optimizable parameters for this optimization process are the
    NURBS control points.

    Attributes
    ----------
    scenario : Scenario
        The scenario.
    normalized_measured_flux_distributions : torch.Tensor
        The measured, normalized flux density distributions.
    incident_ray_directions : torch.Tensor
        The incident ray directions of the measured fluxes.
    heliostats_mask : torch.Tensor
        A mask for the selected heliostats for reconstruction.
    target_area_mask : torch.Tensor
        The indices of the target area for each reconstruction.
    number_of_surface_points : torch.Tensor
        The number of surface points of the reconstructed surfaces.
    resolution : torch.Tensor
        The resolution of all bitmaps during reconstruction.
    num_log : int
        The number of log statements during optimization.
    initial_learning_rate : float
        The initial learning rate for the optimizer (default is 0.0004).
    tolerance : float
        The optimizer tolerance.
    max_epoch : int
        The maximum number of optimization epochs.

    Methods
    -------
    reconstruct_surfaces()
        Reconstruct NURBS surfaces from bitmaps.
    """

    def __init__(
        self,
        scenario: Scenario,
        heliostat_data_mapping: list[
            tuple[str, list[pathlib.Path], list[pathlib.Path]]
        ],
        number_of_surface_points: torch.Tensor = torch.tensor([50, 50]),
        resolution: torch.Tensor = torch.tensor([256, 256]),
        initial_learning_rate: float = 1e-5,
        tolerance: float = 0.0005,
        max_epoch: int = 1000,
        num_log: int = 3,
        number_of_measurements: int = 4,
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize the surface reconstructor.

        Parameters
        ----------
        scenario : Scenario
            The scenario.
        heliostat_data_mapping : list[tuple[str, list[pathlib.Path, list[pathlib.Path]]]
            The mapping of heliostat and reconstruction data.
        number_of_surface_points : torch.Tensor
            The number of surface points of the reconstructed surfaces.
        resolution : torch.Tensor
            The resolution of all bitmaps during reconstruction.
        initial_learning_rate : float
            The initial learning rate for the optimizer (default is 1e-5).
        tolerance : float
            The tolerance during optimization (default is 0.0005).
        max_epoch : int
            The maximum optimization epoch (default is 1000).
        num_log : int
            The number of log statements during optimization (default is 3).
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
            log.info("Create a surface reconstructor.")

        self.scenario = scenario

        self.heliostat_data_mapping = heliostat_data_mapping
        self.number_of_measurements = number_of_measurements

        self.number_of_surface_points = number_of_surface_points.to(device)
        self.resolution = resolution.to(device)
        self.num_log = num_log

        # Create the optimizer.
        self.initial_learning_rate = initial_learning_rate
        self.tolerance = tolerance
        self.max_epoch = max_epoch

    def reconstruct_surfaces(
        self,
        ddp_setup,
        parameter_combination,
        device: torch.device | None = None,
    ) -> None:
        """
        Reconstruct NURBS surfaces from bitmaps.

        Parameters
        ----------
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        device = get_device(device=ddp_setup["device"])

        rank = ddp_setup["rank"]
        if rank == 0:
            log.info("Start the surface reconstruction.")
        
        for heliostat_group_index in ddp_setup["groups_to_ranks_mapping"][rank]:
            
            #if (rank % ddp_setup["heliostat_group_world_size"]) < self.scenario.heliostat_field.number_of_heliostat_groups:
            
            heliostat_group = self.scenario.heliostat_field.heliostat_groups[
                heliostat_group_index
            ]
            
            # Extract measured fluxes.
            heliostat_flux_path_mapping = [
                (heliostat_name, png_paths)
                for heliostat_name, _, png_paths in self.heliostat_data_mapping
                if heliostat_name in heliostat_group.names
            ]

            normalized_measured_flux_distributions = (
                flux_distribution_loader.load_flux_from_png(
                    heliostat_flux_path_mapping=heliostat_flux_path_mapping,
                    heliostat_names=heliostat_group.names,
                    resolution=self.resolution,
                    number_of_measurements=self.number_of_measurements,
                    device=device,
                )
            )

            # Extract environmental data for measured fluxes.
            heliostat_calibration_mapping = [
                (heliostat_name, calibration_properties_paths)
                for heliostat_name, calibration_properties_paths, _ in self.heliostat_data_mapping
                if heliostat_name in heliostat_group.names
            ]
            (
                _,
                incident_ray_directions,
                _,
                heliostats_mask,
                target_area_mask,
            ) = paint_loader.extract_paint_calibration_properties_data(
                heliostat_calibration_mapping=heliostat_calibration_mapping,
                heliostat_names=heliostat_group.names,
                target_area_names=self.scenario.target_areas.names,
                power_plant_position=self.scenario.power_plant_position,
                number_of_measurements=self.number_of_measurements,
                device=device,
            )
            
            if heliostats_mask.sum() > 0:
                evaluation_points = (
                    utils.create_nurbs_evaluation_grid(
                        number_of_evaluation_points=self.number_of_surface_points,
                        device=device,
                    )
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(
                        heliostats_mask.sum(),
                        heliostat_group.number_of_facets_per_heliostat,
                        -1,
                        -1,
                    )
                )

                # Activate heliostats.
                heliostat_group.activate_heliostats(
                    active_heliostats_mask=heliostats_mask, device=device
                )

                optimizer = torch.optim.Adam(
                    [heliostat_group.nurbs_control_points.requires_grad_()],
                    lr=self.initial_learning_rate,
                )

                # Start the optimization.
                loss = torch.inf
                epoch = 0
                log_step = self.max_epoch // self.num_log
                while loss > self.tolerance and epoch <= self.max_epoch:
                    optimizer.zero_grad()

                    # Activate heliostats.
                    heliostat_group.activate_heliostats(
                        active_heliostats_mask=heliostats_mask, device=device
                    )

                    # Create NURBS
                    nurbs_surfaces = NURBSSurfaces(
                        degrees=heliostat_group.nurbs_degrees,
                        control_points=heliostat_group.active_nurbs_control_points,
                        device=device,
                    )

                    # Calculate surface points and normals
                    (
                        heliostat_group.active_surface_points,
                        heliostat_group.active_surface_normals,
                    ) = nurbs_surfaces.calculate_surface_points_and_normals(
                        evaluation_points=evaluation_points, device=device
                    )

                    heliostat_group.active_surface_points = (
                        heliostat_group.active_surface_points.reshape(
                            heliostat_group.active_surface_points.shape[0], -1, 4
                        )
                    )
                    heliostat_group.active_surface_normals = (
                        heliostat_group.active_surface_normals.reshape(
                            heliostat_group.active_surface_normals.shape[0], -1, 4
                        )
                    )

                    # Align heliostats.
                    heliostat_group.align_surfaces_with_incident_ray_directions(
                        aim_points=self.scenario.target_areas.centers[
                            target_area_mask
                        ],
                        incident_ray_directions=incident_ray_directions,
                        active_heliostats_mask=heliostats_mask,
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
                        bitmap_resolution=self.resolution,
                    )

                    # Perform heliostat-based ray tracing.
                    flux_distributions = ray_tracer.trace_rays(
                        incident_ray_directions=incident_ray_directions,
                        active_heliostats_mask=heliostats_mask,
                        target_area_mask=target_area_mask,
                        device=device,
                    )

                    normalized_flux_distributions = utils.normalize_bitmaps(
                        flux_distributions=flux_distributions,
                        target_area_widths=ray_tracer.scenario.target_areas.dimensions[
                            target_area_mask
                        ][:, 0],
                        target_area_heights=ray_tracer.scenario.target_areas.dimensions[
                            target_area_mask
                        ][:, 1],
                        number_of_rays=torch.full((flux_distributions.shape[0],), ray_tracer.light_source.number_of_rays, device=device),
                        device=device
                    )

                    if ddp_setup["is_nested"]:
                        normalized_flux_distributions = torch.distributed.nn.functional.all_reduce(
                            normalized_flux_distributions,
                            group=ddp_setup["process_subgroup"],
                            op=torch.distributed.ReduceOp.SUM,
                        )

                    loss_function = torch.nn.MSELoss()
                    loss = loss_function(
                        normalized_flux_distributions,
                        normalized_measured_flux_distributions,
                    )
                    loss.backward()

                    if ddp_setup["is_nested"]:
                        for param_group in optimizer.param_groups:
                            for param in param_group["params"]:
                                if param.grad is not None:
                                    torch.distributed.all_reduce(
                                        param.grad,
                                        op=torch.distributed.ReduceOp.SUM,
                                        group=ddp_setup["process_subgroup"]
                                    )
                                    param.grad /= torch.distributed.get_world_size(ddp_setup["process_subgroup"])
                        
                    optimizer.step()

                    if epoch % log_step == 0 and rank == 0:
                        log.info(
                            f"Epoch: {epoch}, Loss: {loss}, LR: {optimizer.param_groups[0]['lr']}",
                        )

                    if epoch in [0, 10, 50, 100, 300, 500, 1000, 2000, 3000]:

                        sp = parameter_combination["points_and_rays"][0][0].item()
                        cp = heliostat_group.nurbs_control_points.shape[2]
                        rays = parameter_combination["points_and_rays"][1]
                        lr = parameter_combination["learning_rates"]
                        h = parameter_combination["scenario_paths_and_measurements"][1]

                        folder_name = f'sp_{sp}_rays_{rays}_cp_{cp}_lr{lr}_h_{h}'
                        
                        if rank == 0:
                            with open((pathlib.Path(f"{folder_name}") / "loss.txt" ), "a") as f:
                                f.write(f"epoch: \t {epoch}, \t loss: \t {loss} \n")

                        with torch.no_grad():
                            
                            for i in range (heliostat_group.number_of_heliostats):
                                temp_nurbs = NURBSSurfaces(
                                    degrees=heliostat_group.nurbs_degrees,
                                    control_points=heliostat_group.nurbs_control_points[i].unsqueeze(0),
                                    device=device,
                                )
                                
                                temp_points, temp_normals = temp_nurbs.calculate_surface_points_and_normals(
                                    evaluation_points=evaluation_points[0].unsqueeze(0),
                                    device=device,
                                )
                                
                                name1 = pathlib.Path("/workVERLEIHNIX/mb/ARTIST/hyperparameter_search/test/") / folder_name / f"points_and_normals_h_{i}_epoch_{epoch}_rank_{rank}"

                                helper.plot_normal_angle_map(
                                    surface_points=temp_points[0],
                                    surface_normals=temp_normals[0],
                                    reference_direction=torch.tensor([0.0, 0.0, 1.0, 0.0], device=device),
                                    name=name1
                                )

                            name2 = pathlib.Path("/workVERLEIHNIX/mb/ARTIST/hyperparameter_search/test/") / folder_name / f"points_and_normals_epoch_{epoch}_rank_{rank}"

                            helper.plot_multiple_fluxes(
                                reconstructed=normalized_flux_distributions,
                                references=normalized_measured_flux_distributions,
                                name=name2
                            )

                            # validation
                            # Extract measured fluxes.
                            validation_heliostat_flux_path_mapping = [
                                (heliostat_name, png_paths)
                                for heliostat_name, _, png_paths in self.heliostat_data_mapping
                                if heliostat_name in heliostat_group.names
                            ]

                            validation_normalized_measured_flux_distributions = (
                                flux_distribution_loader.load_flux_from_png(
                                    heliostat_flux_path_mapping=validation_heliostat_flux_path_mapping,
                                    heliostat_names=heliostat_group.names,
                                    resolution=self.resolution,
                                    number_of_measurements=6,
                                    device=device,
                                )
                            )

                            # Extract environmental data for measured fluxes.
                            validation_heliostat_calibration_mapping = [
                                (heliostat_name, calibration_properties_paths)
                                for heliostat_name, calibration_properties_paths, _ in self.heliostat_data_mapping
                                if heliostat_name in heliostat_group.names
                            ]
                            (
                                _,
                                validation_incident_ray_directions,
                                _,
                                validation_active_heliostats_mask,
                                validation_target_area_mask,
                            ) = paint_loader.extract_paint_calibration_properties_data(
                                heliostat_calibration_mapping=validation_heliostat_calibration_mapping,
                                heliostat_names=heliostat_group.names,
                                target_area_names=self.scenario.target_areas.names,
                                power_plant_position=self.scenario.power_plant_position,
                                number_of_measurements=6,
                                device=device,
                            )

                            # validation_heliostat_target_light_source_mapping = [
                            #     ("AA39", validation_targets[0], validation_incident_ray_directions[0]),
                            #     ("AA39", validation_targets[1], validation_incident_ray_directions[1]),
                            #     ("AA39", validation_targets[2], validation_incident_ray_directions[2]),
                            #     ("AA39", validation_targets[3], validation_incident_ray_directions[3]),
                            #     ("AA39", validation_targets[4], validation_incident_ray_directions[4]),
                            #     ("AA39", validation_targets[5], validation_incident_ray_directions[5]),
                            #     ("AA39", validation_targets[6], validation_incident_ray_directions[6]),
                            #     ("AA39", validation_targets[7], validation_incident_ray_directions[7]),
                            #     ("AA39", validation_targets[8], validation_incident_ray_directions[8]),
                            #     ("AA39", "receiver", torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)),
                            # ]

                            # (
                            #     validation_active_heliostats_mask,
                            #     validation_target_area_mask,
                            #     validation_incident_ray_directions,
                            # ) = self.scenario.index_mapping(
                            #     heliostat_group=heliostat_group,
                            #     string_mapping=validation_heliostat_target_light_source_mapping,
                            #     device=device,
                            # )

                            heliostat_group.activate_heliostats(
                                active_heliostats_mask=validation_active_heliostats_mask, device=device
                            )

                            validation_nurbs = NURBSSurfaces(
                                degrees=heliostat_group.nurbs_degrees,
                                control_points=heliostat_group.active_nurbs_control_points,
                                uniform=True,
                                device=device,
                            )

                            validation_evaluation_points = (
                                utils.create_nurbs_evaluation_grid(
                                    number_of_evaluation_points=self.number_of_surface_points,
                                    device=device,
                                )
                                .unsqueeze(0)
                                .unsqueeze(0)
                                .expand(
                                    validation_active_heliostats_mask.sum(),
                                    4,
                                    -1,
                                    -1,
                                )
                            )

                            validation_calc_points, validation_calc_normals = validation_nurbs.calculate_surface_points_and_normals(
                                evaluation_points=validation_evaluation_points, device=device
                            )

                            heliostat_group.active_surface_points = validation_calc_points.reshape(
                                validation_active_heliostats_mask.sum(), -1, 4
                            )
                            heliostat_group.active_surface_normals = validation_calc_normals.reshape(
                                validation_active_heliostats_mask.sum(), -1, 4
                            )

                            # Align heliostats.
                            heliostat_group.align_surfaces_with_incident_ray_directions(
                                aim_points=self.scenario.target_areas.centers[validation_target_area_mask],
                                incident_ray_directions=validation_incident_ray_directions,
                                active_heliostats_mask=validation_active_heliostats_mask,
                                device=device,
                            )

                            # Create a ray tracer.
                            validation_ray_tracer = HeliostatRayTracer(
                                scenario=self.scenario,
                                heliostat_group=heliostat_group,
                                world_size=ddp_setup["heliostat_group_world_size"],
                                rank=ddp_setup["heliostat_group_rank"],
                                batch_size=heliostat_group.number_of_active_heliostats,
                                random_seed=ddp_setup["heliostat_group_rank"],
                                bitmap_resolution=torch.tensor([256, 256], device=device),
                            )

                            # Perform heliostat-based ray tracing.
                            validation_bitmaps_per_heliostat = validation_ray_tracer.trace_rays(
                                incident_ray_directions=validation_incident_ray_directions,
                                active_heliostats_mask=validation_active_heliostats_mask,
                                target_area_mask=validation_target_area_mask,
                                device=device,
                            )

                            if ddp_setup["is_nested"]:
                                torch.distributed.all_reduce(
                                    validation_bitmaps_per_heliostat,
                                    op=torch.distributed.ReduceOp.SUM,
                                    group=ddp_setup["process_subgroup"],
                                )

                            references = torch.zeros_like(validation_bitmaps_per_heliostat)
                            references[:validation_normalized_measured_flux_distributions.shape[0]] = validation_normalized_measured_flux_distributions

                            plt.clf()
                            name3 = pathlib.Path("/workVERLEIHNIX/mb/ARTIST/hyperparameter_search/test/") / folder_name / f"reconstructed_epoch_{epoch}_rank_{rank}"
                            helper.plot_multiple_fluxes(
                                validation_bitmaps_per_heliostat, references, name=name3
                            )

                    epoch += 1

                log.info("Surfaces reconstructed.")

