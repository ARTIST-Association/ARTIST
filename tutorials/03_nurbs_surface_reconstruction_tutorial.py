"""NURBS surface reconstruction tutorial."""

import logging
import pathlib

import h5py
from matplotlib.ticker import FormatStrFormatter
import torch
from matplotlib import pyplot as plt

from artist.flux import bitmap
from artist.io import (
    CalibrationDataParser,
    PaintCalibrationDataParser,
    paint_scenario_parser,
)
from artist.optim import SurfaceReconstructor
from artist.optim.loss import KLDivergenceLoss
from artist.raytracing import HeliostatRayTracer
from artist.scenario import Scenario
from artist.util import constants, indices, set_logger_config
from artist.util.env import get_device, setup_distributed_environment

torch.manual_seed(7)
torch.cuda.manual_seed(7)

#############################################################################################################
# Define helper functions for the plots.
# Skip to line 270 for the tutorial code.
#############################################################################################################

def create_fluxes(
    data_parser: CalibrationDataParser,
    heliostat_data_mapping: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    resolution: torch.Tensor,
    align_method: str
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[str]]:
    """
    Create data to plot the heliostat fluxes.

    Parameters
    ----------
    data_parser : CalibrationDataParser
        Data parser used to load calibration data from files.
    heliostat_data_mapping : list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
        Mapping from heliostats to calibration data files.
    resolution : torch.Tensor
        Bitmap resolution.
    align_method : str
        Method to align the heliostats.

    Returns
    -------
    list[torch.Tensor]
        Cropped bitmaps per heliostat.
    list[torch.Tensor]
        Bitmaps per heliostat.
    list[torch.Tensor]
        Measured flux bitmap.
    list[str]
        Names of the heliostats.
    """
    cropped_bitmaps_all = []
    bitmaps_all = []
    measured_bitmaps = []
    heliostat_names = []

    for heliostat_group in scenario.heliostat_field.heliostat_groups:
        (
            measured_flux,
            _,
            incident_ray_directions,
            motor_positions,
            active_heliostats_mask,
            target_area_indices,
        ) = data_parser.parse_data_for_reconstruction(
            heliostat_data_mapping=heliostat_data_mapping,
            heliostat_group=heliostat_group,
            scenario=scenario,
            bitmap_resolution=resolution,
            device=device,
        )

        if active_heliostats_mask.sum() > 0:
            measured_bitmaps.append(measured_flux)

            # Activate heliostats.
            heliostat_group.activate_heliostats(
                active_heliostats_mask=active_heliostats_mask,
                device=device,
            )

            if align_method == "motor_pos":
                # Align heliostats.
                heliostat_group.align_surfaces_with_motor_positions(
                    motor_positions=motor_positions,
                    active_heliostats_mask=active_heliostats_mask,
                    device=device,
                )
            elif align_method == "incident_ray":
                # Align heliostats.
                heliostat_group.align_surfaces_with_incident_ray_directions(
                    aim_points=scenario.solar_tower.get_centers_of_target_areas(
                        target_area_indices=target_area_indices, device=device
                    ),
                    incident_ray_directions=incident_ray_directions,
                    active_heliostats_mask=active_heliostats_mask,
                    device=device,
                )

            # Create a ray tracer.
            scenario.set_number_of_rays(number_of_rays=500)
            ray_tracer = HeliostatRayTracer(
                scenario=scenario,
                heliostat_group=heliostat_group,
                blocking_active=False,
                batch_size=heliostat_group.number_of_active_heliostats,
                bitmap_resolution=resolution,
            )

            # Perform heliostat-based ray tracing.
            bitmaps_per_heliostat, _, _, _ = ray_tracer.trace_rays(
                incident_ray_directions=incident_ray_directions,
                active_heliostats_mask=active_heliostats_mask,
                target_area_indices=target_area_indices,
                device=device,
            )

            cropped_bitmaps = bitmap.crop_flux_distributions_around_center(
                flux_distributions=bitmaps_per_heliostat.detach(),
                solar_tower=scenario.solar_tower,
                target_area_indices=target_area_indices.detach(),
                device=device,
            )
            cropped_bitmaps_all.append(cropped_bitmaps)
            bitmaps_all.append(bitmaps_per_heliostat)
            heliostat_names.append(
                [
                    name
                    for name, count in zip(
                        heliostat_group.names, active_heliostats_mask.tolist()
                    )
                    for _ in range(count)
                ]
            )

    return cropped_bitmaps_all, bitmaps_all, measured_bitmaps, heliostat_names


def plot_fluxes(
    fluxes_before: list[torch.Tensor],
    fluxes_after: list[torch.Tensor],
    fluxes_measured: list[torch.Tensor],
    flux_labels: list[list[str]],
) -> None:
    """
    Plot the fluxes.

    Parameters
    ----------
    fluxes_before : list[torch.Tensor]
        Fluxes before the surface reconstruction.
    fluxes_after : list[torch.Tensor]
        Fluxes after the surface reconstruction.
    fluxes_measured : list[torch.Tensor]
        Measured flux references.
    flux_labels : list[list[str]]
        Labels for every heliostat in each group.
    """
    fontsize = 6
    eps = 1e-8
    dims = (indices.batched_bitmap_e, indices.batched_bitmap_u)

    for group_index, (flux_before, flux_after, flux_measured, flux_label) in enumerate(
        zip(fluxes_before, fluxes_after, fluxes_measured, flux_labels)
    ):
        normalized_after = flux_after / flux_after.sum(dim=dims, keepdim=True).clamp_min(eps)
        normalized_measured = flux_measured / flux_measured.sum(dim=dims, keepdim=True).clamp_min(eps)
        normalized_before = flux_before / flux_after.sum(dim=dims, keepdim=True).clamp_min(eps)
        
        all_vals = torch.cat(
            [
                torch.cat([x.flatten() for x in normalized_before]),
                torch.cat([x.flatten() for x in normalized_after]),
                torch.cat([x.flatten() for x in normalized_measured]),
            ]
        )
        vmin = all_vals.min().item()
        vmax = all_vals.max().item()

        n_heliostats = len(flux_before)
        n_cols = 3

        fig, axes = plt.subplots(
            nrows=n_heliostats,
            ncols=n_cols,
            figsize=(4, 3)
        )

        if n_heliostats == 1:
            axes = axes[None, :]

        axes[0, 0].set_title("a) Before\nreconstruction", fontsize=fontsize)
        axes[0, 1].set_title(
            "b) After\nreconstruction",
            fontsize=fontsize,
        )
        axes[0, 2].set_title("c) Measured\nreference", fontsize=fontsize)

        mappable = None
        for i in range(n_heliostats):
            mappable = axes[i, 0].imshow(
                normalized_before[i].detach().cpu(),
                cmap="hot",
                vmin=vmin,
                vmax=vmax,
            )
            axes[i, 0].axis("off")

            axes[i, 1].imshow(
                normalized_after[i].detach().cpu(),
                cmap="hot",
                vmin=vmin,
                vmax=vmax,
            )
            axes[i, 1].axis("off")

            axes[i, 2].imshow(
                normalized_measured[i].detach().cpu(),
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
            )
            axes[i, 2].axis("off")

        for i, label in enumerate(flux_label):
            ax = axes[i, 0]
            ax.text(
                -0.1,
                0.5,
                label,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=fontsize,
                rotation=90,
            )

        cbar = fig.colorbar(
            mappable,
            ax=axes,
            aspect=30,
        )
        cbar.set_label(
            "Flux intensity in normalized flux units",
            fontsize=fontsize,
        )
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))

        plt.savefig(
            f"reconstruction_surfaces_group_{group_index}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


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
heliostat_names = ["heliostat_1", "..."]

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
]

# Or if you have a directory with downloaded data use this code to create a mapping.
# heliostat_data_mapping = paint_scenario_parser.build_heliostat_data_mapping(
#     base_path=base_path_data,
#     heliostat_names=heliostat_names,
#     number_of_measurements=5,
#     image_variant="flux-centered",
#     randomize=True,
# )

# Configure the optimization.
optimizer_dict = {
    constants.initial_learning_rate: 1e-5,
    constants.tolerance: 1e-5,
    constants.max_epoch: 400,
    constants.batch_size: 30,
    constants.log_step: 10,
    constants.early_stopping_delta: 1e-4,
    constants.early_stopping_patience: 100,
    constants.early_stopping_window: 100,
}
# Configure the learning rate scheduler.
scheduler_dict = {
    constants.scheduler_type: constants.cyclic,
    constants.gamma: 0.99,
    constants.lr_min: 1e-6,
    constants.lr_max: 0.0001,
    constants.step_size_up: 100,
    constants.reduce_factor: 0.5,
    constants.patience: 10,
    constants.threshold: 1e-4,
    constants.cooldown: 5,
}
# Configure the regularizers and constraints.
constraint_dict = {
    constants.weight_smoothness: 0.005,
    constants.weight_ideal_surface: 0.005,
    constants.rho_flux_integral: 1.0,
    constants.energy_tolerance: 0.01,
}
# Combine configurations.
optimization_configuration = {
    constants.optimization: optimizer_dict,
    constants.scheduler: scheduler_dict,
    constants.constraints: constraint_dict,
}

# Create dict for the data parser and the heliostat_data_mapping.
data: dict[
    str,
    CalibrationDataParser | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
] = {
    constants.data_parser: PaintCalibrationDataParser(sample_limit=8),
    constants.heliostat_data_mapping: heliostat_data_mapping,
    constants.validation_sample_fraction: 0.2,
}

number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
    scenario_path=scenario_path
)

with setup_distributed_environment(
    number_of_heliostat_groups=number_of_heliostat_groups,
    device=device,
) as ddp_setup:
    device = ddp_setup[constants.device]  # type: ignore

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

    scenario.set_number_of_rays(number_of_rays=190)
    resolution = torch.tensor([256, 256], device=device)

    # Visualize the surfaces and flux distributions from the initial heliostats.
    bitmaps_before, bitmaps_before_uncropped, bitmaps_measured, heliostat_names = create_fluxes(
        data_parser=PaintCalibrationDataParser(sample_limit=1),
        heliostat_data_mapping=[
            (heliostat[0], heliostat[1][-1:], heliostat[2][-1:])
            for heliostat in heliostat_data_mapping
        ],
        resolution=resolution,
        align_method="incident_ray"
    )

    # Create the surface reconstructor.
    surface_reconstructor = SurfaceReconstructor(
        ddp_setup=ddp_setup,
        scenario=scenario,
        data=data,
        optimization_configuration=optimization_configuration,
        bitmap_resolution=resolution,
        device=device,
    )

    # Reconstruct surfaces.
    final_loss_per_heliostat, _ = surface_reconstructor.reconstruct_surfaces(
        loss_definition=loss_definition, device=device
    )

# Inspect the synchronized loss per heliostat. Heliostats that have not been optimized have an infinite loss.
print(f"rank {ddp_setup['rank']}, final loss per heliostat {final_loss_per_heliostat}")

# Visualize the surfaces and flux distributions from the reconstructed heliostats.
bitmaps_after, bitmaps_after_uncropped, bitmaps_measured, flux_labels = create_fluxes(
    data_parser=PaintCalibrationDataParser(sample_limit=1),
    heliostat_data_mapping=[
        (heliostat[0], heliostat[1][-1:], heliostat[2][-1:])
        for heliostat in heliostat_data_mapping
    ],
    resolution=resolution,
    align_method="incident_ray"
)

plot_fluxes(
    fluxes_before=bitmaps_before,
    fluxes_after=bitmaps_after,
    fluxes_measured=bitmaps_measured,
    flux_labels=flux_labels,
)
