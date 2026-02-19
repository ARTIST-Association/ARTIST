import logging
import pathlib

import h5py
import torch
from matplotlib import pyplot as plt

from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.core.loss_functions import KLDivergenceLoss
from artist.core.regularizers import IdealSurfaceRegularizer, SmoothnessRegularizer
from artist.core.surface_reconstructor import SurfaceReconstructor
from artist.data_parser import paint_scenario_parser
from artist.data_parser.calibration_data_parser import CalibrationDataParser
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, index_mapping, set_logger_config, utils
from artist.util.environment_setup import get_device, setup_distributed_environment
from artist.util.nurbs import NURBSSurfaces

torch.manual_seed(7)
torch.cuda.manual_seed(7)


#############################################################################################################
# Define helper functions for the plots.
# Skip to line 343 for the tutorial code.
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
            surface_normals[..., : index_mapping.slice_fourth_dimension]
            / torch.linalg.norm(
                surface_normals[..., : index_mapping.slice_fourth_dimension],
                axis=-1,
                keepdims=True,
            )
        )
        .cpu()
        .detach()
    )
    reference = (
        (
            reference_direction[..., : index_mapping.slice_fourth_dimension]
            / torch.linalg.norm(
                reference_direction[..., : index_mapping.slice_fourth_dimension]
            )
        )
        .cpu()
        .detach()
    )

    sc1 = sc2 = None

    for facet_points, facet_normals in zip(surface_points.cpu().detach(), normals):
        x, y, z = (
            facet_points[:, index_mapping.e].cpu().detach(),
            facet_points[:, index_mapping.n].cpu().detach(),
            facet_points[:, index_mapping.u].cpu().detach(),
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
    plt.close()


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
    plt.close()


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
                    number_of_evaluation_points=torch.tensor([50, 50], device=device),
                    device=device,
                )
                .unsqueeze(index_mapping.heliostat_dimension)
                .unsqueeze(index_mapping.facet_index_unbatched)
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
                ].unsqueeze(index_mapping.heliostat_dimension),
                device=device,
            )

            # Calculate new surface points and normals for this heliostat.
            temporary_points, temporary_normals = (
                temporary_nurbs.calculate_surface_points_and_normals(
                    evaluation_points=evaluation_points,
                    canting=heliostat_group.canting[heliostat_index].unsqueeze(
                        index_mapping.heliostat_dimension
                    ),
                    facet_translations=heliostat_group.facet_translations[
                        heliostat_index
                    ].unsqueeze(index_mapping.heliostat_dimension),
                    device=device,
                )
            )

            # Create the plot.
            plot_surface_points_and_angle_map(
                surface_points=temporary_points[index_mapping.first_heliostat],
                surface_normals=temporary_normals[index_mapping.first_heliostat],
                reference_direction=torch.tensor([0.0, 0.0, 1.0, 0.0], device=device),
                name=f"{name}_rank_{ddp_setup['rank']}_heliostat_group_{heliostat_group_index}_heliostat_{heliostat_group.names[heliostat_index]}",
            )


def create_flux_plots(
    heliostat_names: list[str],
    number_of_plots_per_heliostat: int,
    base_path_data: str,
    data_parser: CalibrationDataParser,
    plot_name: str,
) -> None:
    """
    Create data to plot the heliostat fluxes.

    Parameters
    ----------
    heliostat_names : list[str]
        The names of all heliostats to be plotted.
    number_of_plots_per_heliostat : int
        The number of flux plots for each heliostat.
    base_path_data : str
        The path to the data directory from which to load heliostat field calibration data.
    data_parser : CalibrationDataParser
        The data parser used to load calibration data from files.
    plot_name : str
        The name for the plots.
    """
    # Load reference data.
    validation_heliostat_data_mapping = (
        paint_scenario_parser.build_heliostat_data_mapping(
            base_path=base_path_data,
            heliostat_names=heliostat_names,
            number_of_measurements=number_of_plots_per_heliostat,
            image_variant="flux-centered",
            randomize=True,
        )
    )

    for heliostat_group_index, heliostat_group in enumerate(
        scenario.heliostat_field.heliostat_groups
    ):
        (
            validation_measured_flux_distributions,
            _,
            validation_incident_ray_directions,
            _,
            validation_active_heliostats_mask,
            validation_target_area_mask,
        ) = data_parser.parse_data_for_reconstruction(
            heliostat_data_mapping=validation_heliostat_data_mapping,
            heliostat_group=heliostat_group,
            scenario=scenario,
            bitmap_resolution=torch.tensor([256, 256]),
            device=device,
        )

        if validation_active_heliostats_mask.sum() > 0:
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
                    number_of_evaluation_points=torch.tensor([50, 50], device=device),
                    device=device,
                )
                .unsqueeze(index_mapping.heliostat_dimension)
                .unsqueeze(index_mapping.facet_index_unbatched)
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
                    canting=heliostat_group.active_canting,
                    facet_translations=heliostat_group.active_facet_translations,
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
                blocking_active=False,
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

# Specify the path to your scenario.h5 file and specify the configuration.
scenario_path = pathlib.Path("please/insert/the/path/to/the/scenario/here/scenario.h5")
base_path_data = "base/path/data"
heliostat_names_reconstruction = ["heliostat_1"]
heliostat_names_plots = ["heliostat_1", "..."]

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

# Or if you have a directory with downloaded data use this code to create a mapping.
# heliostat_data_mapping = paint_scenario_parser.build_heliostat_data_mapping(
#     base_path=base_path_data,
#     heliostat_names=heliostat_names_reconstruction,
#     number_of_measurements=2,
#     image_variant="flux-centered",
#     randomize=True,
# )

# Configure the optimization.
optimizer_dict = {
    config_dictionary.initial_learning_rate: 1e-4,
    config_dictionary.tolerance: 1e-5,
    config_dictionary.max_epoch: 30,
    config_dictionary.batch_size: 30,
    config_dictionary.log_step: 1,
    config_dictionary.early_stopping_delta: 1e-4,
    config_dictionary.early_stopping_patience: 100,
    config_dictionary.early_stopping_window: 100,
}
# Configure the learning rate scheduler.
scheduler_dict = {
    config_dictionary.scheduler_type: config_dictionary.exponential,
    config_dictionary.gamma: 0.99,
    config_dictionary.min: 1e-6,
    config_dictionary.max: 1e-2,
    config_dictionary.step_size_up: 100,
    config_dictionary.reduce_factor: 0.5,
    config_dictionary.patience: 10,
    config_dictionary.threshold: 1e-4,
    config_dictionary.cooldown: 5,
}
# Configure the regularizers.
ideal_surface_regularizer = IdealSurfaceRegularizer(reduction_dimensions=(1,))
smoothness_regularizer = SmoothnessRegularizer(reduction_dimensions=(1,))
regularizers = [
    ideal_surface_regularizer,
    smoothness_regularizer,
]
# Configure the regularizers and constraints.
constraint_dict = {
    config_dictionary.regularizers: regularizers,
    config_dictionary.weight_smoothness: 0.005,
    config_dictionary.weight_ideal_surface: 0.005,
    config_dictionary.initial_lambda_energy: 0.1,
    config_dictionary.rho_energy: 1.0,
    config_dictionary.energy_tolerance: 0.01,
}
# Combine configurations.
optimization_configuration = {
    config_dictionary.optimization: optimizer_dict,
    config_dictionary.scheduler: scheduler_dict,
    config_dictionary.constraints: constraint_dict,
}

# Create dict for the data parser and the heliostat_data_mapping.
data: dict[
    str,
    CalibrationDataParser | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
] = {
    config_dictionary.data_parser: PaintCalibrationDataParser(sample_limit=2),
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
            change_number_of_control_points_per_facet=torch.tensor(
                [7, 7], device=device
            ),
            device=device,
        )

    # Set loss function.
    loss_definition = KLDivergenceLoss()
    # Another possibility would be the pixel loss:
    # loss_definition = PixelLoss(scenario=scenario)

    scenario.set_number_of_rays(number_of_rays=170)
    resolution = torch.tensor([256, 256], device=device)

    # Visualize the surfaces and flux distributions from the initial heliostats.
    number_of_plots_per_heliostat = 2
    create_surface_plots(name="ideal")
    create_flux_plots(
        heliostat_names=heliostat_names_plots,
        number_of_plots_per_heliostat=number_of_plots_per_heliostat,
        base_path_data=base_path_data,
        data_parser=PaintCalibrationDataParser(
            sample_limit=number_of_plots_per_heliostat
        ),
        plot_name="ideal",
    )

    # Create the surface reconstructor.
    surface_reconstructor = SurfaceReconstructor(
        ddp_setup=ddp_setup,
        scenario=scenario,
        data=data,
        optimization_configuration=optimization_configuration,
        device=device,
    )

    # Reconstruct surfaces.
    final_loss_per_heliostat = surface_reconstructor.reconstruct_surfaces(
        loss_definition=loss_definition, device=device
    )

# Inspect the synchronized loss per heliostat. Heliostats that have not been optimized have an infinite loss.
print(f"rank {ddp_setup['rank']}, final loss per heliostat {final_loss_per_heliostat}")

# Visualize the surfaces and flux distributions from the reconstructed heliostats.
create_surface_plots(name="reconstructed")
create_flux_plots(
    heliostat_names=heliostat_names_plots,
    number_of_plots_per_heliostat=number_of_plots_per_heliostat,
    base_path_data=base_path_data,
    data_parser=PaintCalibrationDataParser(sample_limit=number_of_plots_per_heliostat),
    plot_name="reconstructed",
)
