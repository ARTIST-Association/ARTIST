import pathlib

import h5py
import torch

from artist.core import loss_functions
from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.core.regularizers import IdealSurfaceRegularizer, TotalVariationRegularizer
from artist.core.surface_reconstructor import SurfaceReconstructor
from artist.data_loader import flux_distribution_loader, paint_loader
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, set_logger_config, utils
from artist.util.environment_setup import get_device, setup_distributed_environment
from artist.util.nurbs import NURBSSurfaces
from examples.hyperparameter_optimization.to_be_removed.helper import (
    plot_multiple_fluxes,
    plot_normal_angle_map,
)

# Set up logger
set_logger_config()

# Set the device
device = get_device()

params = {
    "number_of_surface_points": 96,
    "number_of_control_points": 9,
    "number_of_rays": 136,
    "number_of_training_samples": 6,
    "nurbs_degree": 3,
    "scheduler": "cyclic",
    "lr_gamma": 9.28e-1,
    "lr_min": 8.11e-4,
    "lr_max": 1.83e-3,
    "lr_step_size_up": 1535,
    "lr_reduce_factor": 2.87e-1,
    "lr_patience": 7,
    "lr_threshold": 2.97e-3,
    "lr_cooldown": 0,
    "ideal_surface_loss_weight": 0.00e2,
    "total_variation_loss_points_weight": 5.40e-1,
    "total_variation_loss_normals_weight": 0.00e2,
    "total_variation_loss_number_of_neighbors_points": 1119,
    "total_variation_loss_sigma_points": 0,
    "total_variation_loss_number_of_neighbors_normals": 2653,
    "total_variation_loss_sigma_normals": 1,
    "initial_learning_rate": 6.33e-1,
    "loss_class": "KLDivergenceLoss",
}

scenario_path = pathlib.Path("/home/hgf_dlr/hgf_zas3427/artist-data/scenario.h5")

number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
    scenario_path=scenario_path
)

with setup_distributed_environment(
    number_of_heliostat_groups=number_of_heliostat_groups,
    device=device,
) as ddp_setup:
    device = ddp_setup[config_dictionary.device]

    number_of_surface_points_per_facet = torch.tensor(
        [params["number_of_surface_points"], params["number_of_surface_points"]],
        device=device,
    )

    number_of_control_points_per_facet = torch.tensor(
        [params["number_of_control_points"], params["number_of_control_points"]],
        device=device,
    )

    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file,
            number_of_surface_points_per_facet=number_of_surface_points_per_facet,
            change_number_of_control_points_per_facet=number_of_control_points_per_facet,
            device=device,
        )

    # Set number of rays.
    scenario.light_sources.light_source_list[0].number_of_rays = int(
        params["number_of_rays"]
    )

    # Set nurbs degree.
    for heliostat_group in scenario.heliostat_field.heliostat_groups:
        heliostat_group.nurbs_degrees = torch.tensor(
            [params["nurbs_degree"], params["nurbs_degree"]], device=device
        )

    # Create a heliostat data mapping for the specified number of training samples.
    heliostat_data_mapping = paint_loader.build_heliostat_data_mapping(
        base_path="/home/hgf_dlr/hgf_zas3427/artist-data/paint_sorted",
        heliostat_names=["AA39"],
        number_of_measurements=int(params["number_of_training_samples"]),
        image_variant="flux-centered",
        randomize=False,
    )

    data: dict[str, str | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]] = {
        config_dictionary.data_source: config_dictionary.paint,
        config_dictionary.heliostat_data_mapping: heliostat_data_mapping,
    }

    scheduler = params["scheduler"]
    scheduler_parameters = {
        config_dictionary.gamma: params["lr_gamma"],
        config_dictionary.min: params["lr_min"],
        config_dictionary.max: params["lr_max"],
        config_dictionary.step_size_up: params["lr_step_size_up"],
        config_dictionary.reduce_factor: params["lr_reduce_factor"],
        config_dictionary.patience: params["lr_patience"],
        config_dictionary.threshold: params["lr_threshold"],
        config_dictionary.cooldown: params["lr_cooldown"],
    }

    # Configure regularizers and their weights.
    ideal_surface_regularizer = IdealSurfaceRegularizer(
        weight=params["ideal_surface_loss_weight"], reduction_dimensions=(1, 2, 3, 4)
    )
    total_variation_regularizer_points = TotalVariationRegularizer(
        weight=params["total_variation_loss_points_weight"],
        reduction_dimensions=(1,),
        surface=config_dictionary.surface_points,
        number_of_neighbors=int(
            params["total_variation_loss_number_of_neighbors_points"]
        ),
        sigma=params["total_variation_loss_sigma_points"],
    )
    total_variation_regularizer_normals = TotalVariationRegularizer(
        weight=params["total_variation_loss_normals_weight"],
        reduction_dimensions=(1,),
        surface=config_dictionary.surface_normals,
        number_of_neighbors=int(
            params["total_variation_loss_number_of_neighbors_normals"]
        ),
        sigma=params["total_variation_loss_sigma_normals"],
    )

    regularizers = [
        ideal_surface_regularizer,
        total_variation_regularizer_points,
        total_variation_regularizer_normals,
    ]

    # Set optimizer parameters.
    optimization_configuration = {
        config_dictionary.initial_learning_rate: params["initial_learning_rate"],
        config_dictionary.tolerance: 0.00005,
        config_dictionary.max_epoch: 3000,
        config_dictionary.num_log: 3000,
        config_dictionary.early_stopping_delta: 1e-4,
        config_dictionary.early_stopping_patience: 20,
        config_dictionary.scheduler: scheduler,
        config_dictionary.scheduler_parameters: scheduler_parameters,
        config_dictionary.regularizers: regularizers,
    }

    # Create the surface reconstructor.
    surface_reconstructor = SurfaceReconstructor(
        ddp_setup=ddp_setup,
        scenario=scenario,
        data=data,
        optimization_configuration=optimization_configuration,
        number_of_surface_points=number_of_surface_points_per_facet,
        bitmap_resolution=torch.tensor([256, 256], device=device),
        device=device,
    )

    # Define loss.
    loss_class = getattr(loss_functions, str(params["loss_class"]))
    loss_definition = (
        loss_functions.PixelLoss(scenario=scenario)
        if loss_class is loss_functions.PixelLoss
        else loss_functions.KLDivergenceLoss()
    )

    # Reconstruct surfaces.
    loss = surface_reconstructor.reconstruct_surfaces(
        loss_definition=loss_definition,
        device=device,
    )

    # Validation

    print(f"loss: {loss}")

    for index, heliostat_group in enumerate(scenario.heliostat_field.heliostat_groups):
        torch.save(
            heliostat_group.nurbs_control_points,
            f"/home/hgf_dlr/hgf_zas3427/artist-data/results/group_{index}_cp.pt",
        )

    evaluation_points = (
        utils.create_nurbs_evaluation_grid(
            number_of_evaluation_points=number_of_surface_points_per_facet,
            device=device,
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(
            heliostat_group.number_of_active_heliostats,
            heliostat_group.number_of_facets_per_heliostat,
            -1,
            -1,
        )
    )

    for index, heliostat_group in enumerate(scenario.heliostat_field.heliostat_groups):
        for i in range(heliostat_group.number_of_heliostats):
            temp_nurbs = NURBSSurfaces(
                degrees=heliostat_group.nurbs_degrees,
                control_points=heliostat_group.nurbs_control_points[index].unsqueeze(0),
                device=device,
            )

            temp_points, temp_normals = temp_nurbs.calculate_surface_points_and_normals(
                evaluation_points=evaluation_points[0].unsqueeze(0),
                device=device,
            )

            name1 = pathlib.Path(
                f"/home/hgf_dlr/hgf_zas3427/artist-data/results/points_and_normals_h_{i}"
            )

            plot_normal_angle_map(
                surface_points=temp_points[0],
                surface_normals=temp_normals[0],
                reference_direction=torch.tensor([0.0, 0.0, 1.0, 0.0], device=device),
                name=name1,
            )

    validation_heliostat_data_mapping = paint_loader.build_heliostat_data_mapping(
        base_path="/home/hgf_dlr/hgf_zas3427/artist-data/paint",
        heliostat_names=["AA39"],
        number_of_measurements=16,
        image_variant="flux-centered",
        randomize=True,
    )
    for index, heliostat_group in enumerate(scenario.heliostat_field.heliostat_groups):
        validation_heliostat_flux_path_mapping = []
        validation_heliostat_calibration_mapping = []

        for heliostat, path_properties, path_pngs in validation_heliostat_data_mapping:
            if heliostat in heliostat_group.names:
                validation_heliostat_flux_path_mapping.append((heliostat, path_pngs))
                validation_heliostat_calibration_mapping.append(
                    (heliostat, path_properties)
                )

        validation_measured_flux_distributions = (
            flux_distribution_loader.load_flux_from_png(
                heliostat_flux_path_mapping=validation_heliostat_flux_path_mapping,
                heliostat_names=heliostat_group.names,
                limit_number_of_measurements=16,
                device=device,
            )
        )
        (
            _,
            validation_incident_ray_directions,
            _,
            validation_active_heliostats_mask,
            validation_target_area_mask,
        ) = paint_loader.extract_paint_calibration_properties_data(
            heliostat_calibration_mapping=validation_heliostat_calibration_mapping,
            heliostat_names=heliostat_group.names,
            target_area_names=scenario.target_areas.names,
            power_plant_position=scenario.power_plant_position,
            limit_number_of_measurements=16,
            device=device,
        )

        heliostat_group.activate_heliostats(
            active_heliostats_mask=validation_active_heliostats_mask,
            device=device,
        )

        validation_nurbs = NURBSSurfaces(
            degrees=heliostat_group.nurbs_degrees,
            control_points=heliostat_group.active_nurbs_control_points,
            uniform=True,
            device=device,
        )

        validation_evaluation_points = (
            utils.create_nurbs_evaluation_grid(
                number_of_evaluation_points=number_of_surface_points_per_facet,
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

        validation_calc_points, validation_calc_normals = (
            validation_nurbs.calculate_surface_points_and_normals(
                evaluation_points=validation_evaluation_points,
                device=device,
            )
        )

        heliostat_group.active_surface_points = validation_calc_points.reshape(
            validation_active_heliostats_mask.sum(), -1, 4
        )
        heliostat_group.active_surface_normals = validation_calc_normals.reshape(
            validation_active_heliostats_mask.sum(), -1, 4
        )

        # Align heliostats.
        heliostat_group.align_surfaces_with_incident_ray_directions(
            aim_points=scenario.target_areas.centers[validation_target_area_mask],
            incident_ray_directions=validation_incident_ray_directions,
            active_heliostats_mask=validation_active_heliostats_mask,
            device=device,
        )

        # Create a ray tracer.
        validation_ray_tracer = HeliostatRayTracer(
            scenario=scenario,
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

        name2 = pathlib.Path(
            "/home/hgf_dlr/hgf_zas3427/artist-data/results/reconstructed"
        )

        plot_multiple_fluxes(
            validation_bitmaps_per_heliostat,
            validation_measured_flux_distributions,
            name=name2,
        )
