import logging
import pathlib

import h5py
import torch
from matplotlib import pyplot as plt

from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.core.loss_functions import KLDivergenceLoss
from artist.core.regularizers import IdealSurfaceRegularizer, TotalVariationRegularizer
from artist.core.surface_reconstructor import SurfaceReconstructor
from artist.data_loader import flux_distribution_loader, paint_loader
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, set_logger_config, utils
from artist.util.environment_setup import get_device, setup_distributed_environment
from artist.util.nurbs import NURBSSurfaces

torch.manual_seed(7)
torch.cuda.manual_seed(7)


#############################################################################################################
# Define helper functions for the plots.
# Skip to line 322 for the tutorial code.
#############################################################################################################


def plot_surface_points_and_angle_map(
    surface_points: torch.Tensor,
    surface_normals: torch.Tensor,
    reference_direction: torch.Tensor,
    name: str,
) -> None:
    """
    Plot the surface points and an angle map comparing surface normals against a given reference direction.

    The function creates a side-by-side plot. The subplot on the left plots the surface points of each facet as a
    scatter plot, where the color represents the z-value. The subplot on the right is an angle map (in radians) of
    the surface normals relative to a reference vector.

    Parameters
    ----------
    surface_points : torch.Tensor
        The surface points for one heliostat.
        Tensor of shape [1, number_of_combined_surface_points_all_facets, 4].
    surface_normals : torch.Tensor
        The surface normals for one heliostat.
        Tensor of shape [1, number_of_combined_surface_points_all_facets, 4].
    reference_direction : torch.Tensor
        The reference direction.
        Tensor of shape [4].
    name : str
        The name or index of the heliostat.
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    normals = (
        (
            surface_normals[..., :3]
            / torch.linalg.norm(surface_normals[..., :3], axis=-1, keepdims=True)
        )
        .cpu()
        .detach()
    )
    reference = (
        (reference_direction[..., :3] / torch.linalg.norm(reference_direction[..., :3]))
        .cpu()
        .detach()
    )

    sc1 = sc2 = None

    for facet_points, facet_normals in zip(surface_points.cpu().detach(), normals):
        x, y, z = (
            facet_points[:, 0].cpu().detach(),
            facet_points[:, 1].cpu().detach(),
            facet_points[:, 2].cpu().detach(),
        )

        # Surface points scatter plot.
        sc1 = axes[0].scatter(x, y, c=z, cmap="viridis")

        # Angle map scatter plot.
        cos_theta = facet_normals @ reference
        angles = torch.arccos(torch.clip(cos_theta, -1.0, 1.0))
        sc2 = axes[1].scatter(x, y, c=angles.numpy(), cmap="plasma", vmin=0, vmax=0.02)

    # Titles
    axes[0].set_title("Surface points")
    axes[1].set_title("Angle map normals")

    # Add only one colorbar per subplot
    plt.colorbar(sc1, ax=axes[0], fraction=0.046, pad=0.04, label="Z-coordinate")
    plt.colorbar(sc2, ax=axes[1], fraction=0.046, pad=0.04, label="Angle (radians)")

    plt.tight_layout()
    plt.savefig(f"2d_points_and_normals_{name}.png")
    plt.clf()


def plot_multiple_fluxes(
    reconstructed: torch.Tensor,
    references: torch.Tensor,
    name: str,
) -> None:
    """
    Plot and compare multiple flux images against their corresponding references.

    For each index i the reconstructed image is on the left and the reference image is on the right.

    Parameters
    ----------
    reconstructed : torch.Tensor
        The flux density distributions raytraced on the reconstructed surfaces.
        Tensor of shape [number_of_samples, bitmap_resolution_e, bitmap_resolution_u].
    references : torch.Tensor
        The flux density distribution references.
        Tensor of shape [number_of_samples, bitmap_resolution_e, bitmap_resolution_u].
    name : str
        The name or index of the heliostat_group.
    """
    fig1, axes1 = plt.subplots(nrows=reconstructed.shape[0], ncols=2, figsize=(24, 72))
    for i in range(reconstructed.shape[0]):
        axes1[i, 0].imshow(reconstructed[i].cpu().detach(), cmap="gray")
        axes1[i, 0].axis("off")

        axes1[i, 1].imshow(references[i].cpu().detach(), cmap="gray")
        axes1[i, 1].axis("off")
    plt.tight_layout()
    plt.savefig(f"flux_comparison_{name}.png")
    plt.clf()


def create_surface_plots(name: str) -> None:
    """
    Create data to plot the surface points and angle map.

    Parameters
    ----------
    name : str
        The name for the plots.
    """
    # Plot the surface points and angle map.
    for heliostat_group_index, heliostat_group in enumerate(
        scenario.heliostat_field.heliostat_groups
    ):
        for heliostat_index in range(heliostat_group.number_of_heliostats):
            # Create evaluation points.
            evaluation_points = (
                utils.create_nurbs_evaluation_grid(
                    number_of_evaluation_points=number_of_surface_points,
                    device=device,
                )
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(
                    1,
                    heliostat_group.number_of_facets_per_heliostat,
                    -1,
                    -1,
                )
            )

            # Create NURBS surface of selected heliostat.
            temporary_nurbs = NURBSSurfaces(
                degrees=heliostat_group.nurbs_degrees,
                control_points=heliostat_group.nurbs_control_points[
                    heliostat_index
                ].unsqueeze(0),
                device=device,
            )

            # Calculate new surface points and normals for this heliostat.
            temporary_points, temporary_normals = (
                temporary_nurbs.calculate_surface_points_and_normals(
                    evaluation_points=evaluation_points[0].unsqueeze(0),
                    device=device,
                )
            )

            # Create the plot.
            plot_surface_points_and_angle_map(
                surface_points=temporary_points[0],
                surface_normals=temporary_normals[0],
                reference_direction=torch.tensor([0.0, 0.0, 1.0, 0.0], device=device),
                name=f"{name}_rank_{ddp_setup['rank']}_heliostat_group_{heliostat_group_index}_heliostat_{heliostat_index}",
            )


def create_flux_plots(
    heliostat_names: list[str], number_of_plots_per_heliostat: int, plot_name: str
) -> None:
    """
    Create data to plot the heliostat fluxes.

    Parameters
    ----------
    heliostat_names : list[str]
        The names of all heliostats to be plotted.
    number_of_plots_per_heliostat : int
        The number of flux plots for each heliostat.
    plot_name : str
        The name for the plots.
    """
    # Load reference data.
    validation_heliostat_data_mapping = paint_loader.build_heliostat_data_mapping(
        base_path="/path/to/data",
        heliostat_names=heliostat_names,
        number_of_measurements=number_of_plots_per_heliostat,
        image_variant="flux-centered",
        randomize=True,
    )

    for heliostat_group_index, heliostat_group in enumerate(
        scenario.heliostat_field.heliostat_groups
    ):
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

        # Activate heliostats.
        heliostat_group.activate_heliostats(
            active_heliostats_mask=validation_active_heliostats_mask,
            device=device,
        )

        # Create surfaces for all samples.
        validation_nurbs = NURBSSurfaces(
            degrees=heliostat_group.nurbs_degrees,
            control_points=heliostat_group.active_nurbs_control_points,
            uniform=True,
            device=device,
        )

        # Create evaluation points for all samples.
        validation_evaluation_points = (
            utils.create_nurbs_evaluation_grid(
                number_of_evaluation_points=number_of_surface_points,
                device=device,
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(
                validation_active_heliostats_mask.sum(),
                heliostat_group.number_of_facets_per_heliostat,
                -1,
                -1,
            )
        )

        # Calculate new surface points and normals for all samples.
        validation_surface_points, validation_surface_normals = (
            validation_nurbs.calculate_surface_points_and_normals(
                evaluation_points=validation_evaluation_points,
                device=device,
            )
        )

        heliostat_group.active_surface_points = validation_surface_points.reshape(
            validation_active_heliostats_mask.sum(), -1, 4
        )
        heliostat_group.active_surface_normals = validation_surface_normals.reshape(
            validation_active_heliostats_mask.sum(), -1, 4
        )

        # Align heliostats.
        heliostat_group.align_surfaces_with_incident_ray_directions(
            aim_points=scenario.target_areas.centers[validation_target_area_mask],
            incident_ray_directions=validation_incident_ray_directions,
            active_heliostats_mask=validation_active_heliostats_mask,
            device=device,
        )

        # Create a ray tracer and reduce number of rays in scenario light source.
        scenario.set_number_of_rays(number_of_rays=10)
        validation_ray_tracer = HeliostatRayTracer(
            scenario=scenario,
            heliostat_group=heliostat_group,
            batch_size=heliostat_group.number_of_active_heliostats,
            bitmap_resolution=torch.tensor([256, 256], device=device),
        )

        # Perform heliostat-based ray tracing.
        validation_bitmaps_per_heliostat = validation_ray_tracer.trace_rays(
            incident_ray_directions=validation_incident_ray_directions,
            active_heliostats_mask=validation_active_heliostats_mask,
            target_area_mask=validation_target_area_mask,
            device=device,
        )

        # Create the plots.
        plot_multiple_fluxes(
            validation_bitmaps_per_heliostat,
            validation_measured_flux_distributions,
            name=f"{plot_name}_rank_{ddp_setup['rank']}_heliostat_group_{heliostat_group_index}",
        )


#############################################################################################################
# Tutorial
#############################################################################################################

# Set up logger.
set_logger_config()
log = logging.getLogger(__name__)

# Set the device.
device = get_device()

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path(
    scenario_path=pathlib.Path(
        "please/insert/the/path/to/the/scenario/here/scenario.h5"
    )
)

# Also specify the heliostats to be calibrated and the paths to your calibration-properties.json files.
# Please use the following style: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
heliostat_data_mapping = [
    (
        "heliostat_name_1",
        [
            pathlib.Path(
                "please/insert/the/path/to/the/paint/data/here/calibration-properties.json"
            ),
            # ....
        ],
        [
            pathlib.Path("please/insert/the/path/to/the/paint/data/here/flux.png"),
            # ....
        ],
    ),
    (
        "heliostat_name_2",
        [
            pathlib.Path(
                "please/insert/the/path/to/the/paint/data/here/calibration-properties.json"
            ),
            # ....
        ],
        [
            pathlib.Path("please/insert/the/path/to/the/paint/data/here/flux.png"),
            # ....
        ],
    ),
    # ...
]

# Create dict for the data source name and the heliostat_data_mapping.
data: dict[str, str | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]] = {
    config_dictionary.data_source: config_dictionary.paint,
    config_dictionary.heliostat_data_mapping: heliostat_data_mapping,
}

number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
    scenario_path=scenario_path
)

with setup_distributed_environment(
    number_of_heliostat_groups=number_of_heliostat_groups,
    device=device,
) as ddp_setup:
    device = ddp_setup[config_dictionary.device]

    # Load the scenario.
    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file,
            device=device,
            change_number_of_control_points_per_facet=torch.tensor(
                [17, 17], device=device
            ),
        )

    # Set loss function.
    loss_definition = KLDivergenceLoss()
    # Another possibility would be the pixel loss:
    # loss_definition = PixelLoss(scenario=scenario)

    # Configure regularizers and their weights.
    ideal_surface_regularizer = IdealSurfaceRegularizer(
        weight=0.0, reduction_dimensions=(1, 2, 3, 4)
    )
    total_variation_regularizer_points = TotalVariationRegularizer(
        weight=0.3,
        reduction_dimensions=(1,),
        surface=config_dictionary.surface_points,
        number_of_neighbors=1000,
        sigma=1e-3,
    )
    total_variation_regularizer_normals = TotalVariationRegularizer(
        weight=0.8,
        reduction_dimensions=(1,),
        surface=config_dictionary.surface_points,
        number_of_neighbors=1000,
        sigma=1e-3,
    )

    regularizers = [
        ideal_surface_regularizer,
        total_variation_regularizer_points,
        total_variation_regularizer_normals,
    ]

    # Configure the learning rate scheduler. The example scheduler parameter dict includes
    # example parameters for all three possible schedulers.
    scheduler = config_dictionary.cyclic  # exponential, cyclic or reduce_on_plateau
    scheduler_parameters = {
        config_dictionary.gamma: 0.5,
        config_dictionary.min: 5e-6,
        config_dictionary.max: 8e-5,
        config_dictionary.step_size_up: 50,
        config_dictionary.reduce_factor: 0.5,
        config_dictionary.patience: 20,
        config_dictionary.threshold: 1e-4,
        config_dictionary.cooldown: 5,
    }

    # Set optimizer parameters.
    optimization_configuration = {
        config_dictionary.initial_learning_rate: 2e-4,
        config_dictionary.tolerance: 1e-5,
        config_dictionary.max_epoch: 27,
        config_dictionary.num_log: 27,
        config_dictionary.early_stopping_delta: 5e-5,
        config_dictionary.early_stopping_patience: 40,
        config_dictionary.scheduler: scheduler,
        config_dictionary.scheduler_parameters: scheduler_parameters,
        config_dictionary.regularizers: regularizers,
    }

    scenario.set_number_of_rays(number_of_rays=170)
    number_of_surface_points = torch.tensor([60, 60], device=device)
    resolution = torch.tensor([256, 256], device=device)

    # Visualize the ideal surfaces and flux distributions from ideal heliostats.
    # Please adapt the heliostat names according to the ones to be plotted.
    heliostat_names = ["heliostat_name_1, heliostat_name_2"]
    number_of_plots_per_heliostat = 2
    create_surface_plots(name="ideal")
    create_flux_plots(
        heliostat_names=heliostat_names,
        number_of_plots_per_heliostat=number_of_plots_per_heliostat,
        plot_name="ideal",
    )

    # Create the surface reconstructor.
    surface_reconstructor = SurfaceReconstructor(
        ddp_setup=ddp_setup,
        scenario=scenario,
        data=data,
        optimization_configuration=optimization_configuration,
        number_of_surface_points=number_of_surface_points,
        bitmap_resolution=resolution,
        device=device,
    )

    # Reconstruct surfaces.
    final_loss_per_heliostat = surface_reconstructor.reconstruct_surfaces(
        loss_definition=loss_definition, device=device
    )

# Inspect the synchronized loss per heliostat. Heliostats that have not been optimized have an infinite loss.
print(f"rank {ddp_setup['rank']}, final loss per heliostat {final_loss_per_heliostat}")

# Visualize the results (reconstructed surfaces and flux distributions from reconstructed heliostats).
create_surface_plots(name="reconstructed")
create_flux_plots(
    heliostat_names=heliostat_names,
    number_of_plots_per_heliostat=number_of_plots_per_heliostat,
    plot_name="reconstructed",
)
