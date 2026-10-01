"""Kinematics reconstruction tutorial."""

import logging
import pathlib

import h5py
from matplotlib.ticker import FormatStrFormatter
import paint.util.paint_mappings as paint_mappings
import torch
from matplotlib import pyplot as plt

from artist.field import HeliostatGroup
from artist.flux import bitmap
from artist.io import (
    CalibrationDataParser,
    PaintCalibrationDataParser,
    paint_scenario_parser,
)
from artist.optim import KinematicsReconstructor
from artist.optim.loss import AngleLoss, FocalSpotLoss
from artist.raytracing import HeliostatRayTracer
from artist.scenario import Scenario
from artist.util import constants, indices, set_logger_config
from artist.util.env import get_device, setup_distributed_environment

torch.manual_seed(7)
torch.cuda.manual_seed(7)

#############################################################################################################
# Define helper functions for the plots.
# Skip to line 335 for the tutorial code.
#############################################################################################################

def create_fluxes(
    data_parser: CalibrationDataParser,
    heliostat_data_mapping: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    resolution: torch.Tensor,
    align_method: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str]]:
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
        Bitmaps per heliostat.
    list[torch.Tensor]
        Measured flux bitmap.
    list[str]
        Names of the heliostats.
    """
    bitmaps = []
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

            scenario.set_number_of_rays(number_of_rays=500)
            # Create a ray tracer.
            ray_tracer = HeliostatRayTracer(
                scenario=scenario,
                heliostat_group=heliostat_group,
                occlusion_active=False,
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
            bitmaps.append(bitmaps_per_heliostat)
            heliostat_names.append(
                [
                    name
                    for name, count in zip(
                        heliostat_group.names, active_heliostats_mask.tolist()
                    )
                    for _ in range(count)
                ]
            )

    return bitmaps, measured_bitmaps, heliostat_names


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
        Fluxes before the kinematics reconstruction.
    fluxes_after : list[torch.Tensor]
        Fluxes after the kinematics reconstruction.
    fluxes_measured : list[torch.Tensor]
        Measured flux references.
    flux_labels : list[list[str]]
        Labels for every heliostat in each group.
    """
    fontsize = 6
    eps = 1e-8
    dims = (indices.batched_bitmap_e, indices.batched_bitmap_u)

    column_titles = (
        "a) Before\nreconstruction",
        "b) After\nreconstruction",
        "c) Measured\nreference",
    )

    for group_index, (
        flux_before,
        flux_after,
        flux_measured,
        labels,
    ) in enumerate(
        zip(
            fluxes_before,
            fluxes_after,
            fluxes_measured,
            flux_labels,
        )
    ):
        centers = {
            "before": bitmap.get_center_of_mass(flux_before, device=device).cpu(),
            "after": bitmap.get_center_of_mass(flux_after, device=device).cpu(),
            "measured": bitmap.get_center_of_mass(flux_measured, device=device).cpu(),
        }

        normalized = {
            "before": flux_before
            / flux_before.sum(dim=dims, keepdim=True).clamp_min(eps),
            "after": flux_after
            / flux_after.sum(dim=dims, keepdim=True).clamp_min(eps),
            "measured": flux_measured
            / flux_measured.sum(dim=dims, keepdim=True).clamp_min(eps),
        }

        all_values = torch.cat(
            [tensor.flatten() for tensor in normalized.values()]
        )

        vmin = all_values.min().item()
        vmax = all_values.max().item()

        n_heliostats = flux_before.shape[0]

        fig, axes = plt.subplots(
            nrows=n_heliostats,
            ncols=3,
            figsize=(4, 3),
        )

        if n_heliostats == 1:
            axes = axes[None, :]

        for ax, title in zip(axes[0], column_titles):
            ax.set_title(title, fontsize=fontsize)

        plot_columns = [
            (
                normalized["before"],
                "hot",
                centers["before"],
                True,
            ),
            (
                normalized["after"],
                "hot",
                centers["after"],
                False,
            ),
            (
                normalized["measured"],
                "gray",
                None,
                False,
            ),
        ]

        mappable = None

        for row in range(n_heliostats):
            for col, (images, cmap, artist_center, skip_zero) in enumerate(
                plot_columns
            ):
                ax = axes[row, col]

                mappable = ax.imshow(
                    images[row].cpu(),
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )

                ax.scatter(
                    *centers["measured"][row],
                    c="royalblue",
                    marker="o",
                    s=10,
                    label="CoM Reference",
                )

                if artist_center is not None:
                    if (
                        not skip_zero
                        or (artist_center[row] != torch.tensor([0, 0])).all()
                    ):
                        ax.scatter(
                            *artist_center[row],
                            c="forestgreen",
                            marker="D",
                            s=10,
                            label="CoM ARTIST",
                        )

                ax.axis("off")

            axes[row, 0].text(
                -0.1,
                0.5,
                labels[row],
                transform=axes[row, 0].transAxes,
                ha="center",
                va="center",
                fontsize=fontsize,
                rotation=90,
            )

        handles, legend_labels = axes[0, 1].get_legend_handles_labels()
        grid_bbox = axes[-1, -1].get_position()

        fig.legend(
            handles,
            legend_labels,
            loc="lower left",
            bbox_to_anchor=(grid_bbox.x1 - 0.1, grid_bbox.y0),
            borderaxespad=0,
            fontsize=fontsize,
        )

        cbar = fig.colorbar(
            mappable,
            ax=axes,
            aspect=30,
        )

        cbar.ax.set_position(
            [
                grid_bbox.x1 - 0.1,
                grid_bbox.y0 + 0.12,
                0.02,
                0.65,
            ]
        )

        cbar.set_label(
            "Flux intensity in normalized flux units",
            fontsize=fontsize,
        )
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.ax.yaxis.set_major_formatter(
            FormatStrFormatter("%.1e")
        )

        plt.savefig(
            f"reconstruction_kinematics_group_{group_index}.png",
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

# Specify the path to your scenario.h5 file.
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
#     number_of_measurements=16,
#     image_variant="flux",
#     randomize=True,
# )

# Configure the optimization.
optimizer_dict: dict[str, str | float | int] = {
    constants.initial_learning_rate_rotation_deviation: 1e-4,
    constants.tolerance: 0.0000,
    constants.max_epoch: 200,
    constants.batch_size: 50,
    constants.log_step: 50,
    constants.early_stopping_delta: 1e-8,
    constants.early_stopping_patience: 1000,
    constants.early_stopping_window: 2000,
}
# Configure the learning rate scheduler.
scheduler_dict: dict[str, str | float | int] = {
    constants.scheduler_type: constants.reduce_on_plateau,
    constants.gamma: 0.9,
    constants.lr_min: 1e-6,
    constants.lr_max: 1e-3,
    constants.step_size_up: 500,
    constants.reduce_factor: 0.0001,
    constants.patience: 50,
    constants.threshold: 1e-3,
    constants.cooldown: 10,
}
# Combine configurations.
optimization_configuration: dict[str, dict[str, str | float | int]] = {
    constants.optimization: optimizer_dict,
    constants.scheduler: scheduler_dict,
}

# Create dict for the data parser and the heliostat_data_mapping.
data: dict[
    str,
    CalibrationDataParser | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
] = {
    constants.data_parser: PaintCalibrationDataParser(sample_limit=20),
    constants.heliostat_data_mapping: heliostat_data_mapping,
    constants.validation_sample_fraction: 0.25,
}

number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
    scenario_path=scenario_path
)

with setup_distributed_environment(
    number_of_heliostat_groups=number_of_heliostat_groups,
    device=device,
) as ddp_setup:
    device = ddp_setup[constants.device]  # type:ignore

    # Load the scenario.
    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file,
            change_number_of_control_points_per_facet=torch.tensor(
                [7, 7], device=device
            ), 
            device=device
        )

    resolution = torch.tensor([256, 256], device=device)

    bitmaps_before, _, _ = create_fluxes(
        data_parser=PaintCalibrationDataParser(sample_limit=1),
        heliostat_data_mapping=[
            (heliostat[0], heliostat[1][-1:], heliostat[2][-1:])
            for heliostat in heliostat_data_mapping
        ],
        resolution=resolution,
        align_method="motor_pos",
    )

    scenario.set_number_of_rays(number_of_rays=6)

    optimization_configuration[constants.optimization][
        constants.initial_learning_rate_rotation_deviation
    ] = 3e-4
    optimization_configuration[constants.optimization][constants.max_epoch] = 100
    # Create the kinematics reconstructor.
    kinematics_reconstructor = KinematicsReconstructor(
        ddp_setup=ddp_setup,
        scenario=scenario,
        data=data,
        dni=500,
        optimization_configuration=optimization_configuration,
        reconstruction_method=constants.kinematics_reconstruction_alignment,
        bitmap_resolution=resolution,
    )
    # Reconstruct the kinematics.
    final_loss_per_heliostat = kinematics_reconstructor.reconstruct_kinematics(
        loss_definition=AngleLoss(), device=device
    )

    # Uncomment the code below to add a further reconstruction step using raytracing to refine the kinematics reconstruction.
    # optimization_configuration[constants.optimization][
    #     constants.initial_learning_rate_rotation_deviation
    # ] = 1e-4
    # optimization_configuration[constants.optimization][constants.max_epoch] = 700
    # # Create the kinematics reconstructor.
    # kinematics_reconstructor = KinematicsReconstructor(
    #     ddp_setup=ddp_setup,
    #     scenario=scenario,
    #     data=data,
    #     dni=500,
    #     optimization_configuration=optimization_configuration,
    #     reconstruction_method=constants.kinematics_reconstruction_raytracing,
    #     bitmap_resolution=resolution,
    #     plot_results=True
    # )
    # # Reconstruct the kinematics.
    # final_loss_per_heliostat = kinematics_reconstructor.reconstruct_kinematics(
    #     loss_definition=FocalSpotLoss(scenario=scenario), device=device
    # )

# Inspect the synchronized loss per heliostat. Heliostats that have not been optimized have an infinite loss.
print(f"rank {ddp_setup['rank']}, final loss per heliostat {final_loss_per_heliostat}")

bitmaps_after, bitmaps_measured, flux_labels = create_fluxes(
    data_parser=PaintCalibrationDataParser(sample_limit=1),
    heliostat_data_mapping=[
        (heliostat[0], heliostat[1][-1:], heliostat[2][-1:])
        for heliostat in heliostat_data_mapping
    ],
    resolution=resolution,
    align_method="incident_ray",
)

plot_fluxes(
    fluxes_before=bitmaps_before,
    fluxes_after=bitmaps_after,
    fluxes_measured=bitmaps_measured,
    flux_labels=flux_labels
)
