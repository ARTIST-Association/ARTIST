"""Kinematics reconstruction tutorial."""

import logging
import pathlib

import h5py
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
from artist.util import constants, set_logger_config
from artist.util.env import get_device, setup_distributed_environment

torch.manual_seed(7)
torch.cuda.manual_seed(7)

#############################################################################################################
# Define helper functions for the plots.
# Skip to line 170 for the tutorial code.
#############################################################################################################


def create_fluxes(
    data_parser: CalibrationDataParser,
    heliostat_data_mapping: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    resolution: torch.Tensor,
    method: str
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
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

    Returns
    -------
    list[torch.Tensor]
        Bitmaps per heliostat.
    list[torch.Tensor]
        Measured flux bitmap.
    """
    bitmaps = []
    measured_bitmaps = []
    scenario.set_number_of_rays(number_of_rays=100)
    for heliostat_group_index in range(len(scenario.heliostat_field.heliostat_groups)):
        heliostat_group: HeliostatGroup = scenario.heliostat_field.heliostat_groups[
            heliostat_group_index
        ]
        (
            measured_flux,
            focal_spots_measured,
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

            if method == "motor_pos":
                # Align heliostats.
                heliostat_group.align_surfaces_with_motor_positions(
                    motor_positions=motor_positions,
                    active_heliostats_mask=active_heliostats_mask,
                    device=device,
                )
            if method == "incident_ray":
                # Align heliostats.
                heliostat_group.align_surfaces_with_incident_ray_directions(
                    aim_points=scenario.solar_tower.get_centers_of_target_areas(target_area_indices=target_area_indices, device=device),
                    incident_ray_directions=incident_ray_directions,
                    active_heliostats_mask=active_heliostats_mask,
                    device=device,
                )

            # Create a ray tracer.
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
            bitmaps.append(bitmaps_per_heliostat)

    scenario.set_number_of_rays(number_of_rays=4)

    return bitmaps, measured_bitmaps


def create_plots(
    fluxes_before: torch.Tensor,
    fluxes_after: torch.Tensor,
    fluxes_ray: torch.Tensor,
    fluxes_measured: torch.Tensor,
) -> None:
    """
    Create the flux plots using the reconstruction results.

    Parameters
    ----------
    fluxes_before : torch.Tensor
        Fluxes before the kinematics reconstruction.
    fluxes_after : torch.Tensor
        Fluxes after the kinematics reconstruction.
    fluxes_measured : torch.Tensor
        Measured flux references.
    """
    for group_index, (flux_before, flux_after, flux_measured, flux_ray) in enumerate(
        zip(fluxes_before, fluxes_after, fluxes_measured, fluxes_ray)
    ):
        center_measured = (
            bitmap.get_center_of_mass(flux_measured, device=device).cpu().detach()
        )
        center_before = (
            bitmap.get_center_of_mass(flux_before, device=device).cpu().detach()
        )
        center_after = (
            bitmap.get_center_of_mass(flux_after, device=device).cpu().detach()
        )
        center_ray = (
            bitmap.get_center_of_mass(flux_ray, device=device).cpu().detach()
        )

        for i in range(len(flux_before)):
            _, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))

            axes[0].imshow(flux_before[i].cpu().detach(), cmap="gray")
            axes[0].set_title("Before reconstruction", fontsize=16)
            axes[0].scatter(x=center_measured[i, 0], y=center_measured[i, 1], c="green", marker="o", label="Measured")
            axes[0].scatter(x=center_before[i, 0], y=center_before[i, 1], c="red", marker="x", label="Predicted")
            #axes[0].scatter(256/2, 256/2, c="blue")
            axes[0].axis("off")
            
            axes[1].imshow(flux_after[i].cpu().detach(), cmap="gray")
            axes[1].set_title("After reconstruction", fontsize=16)
            axes[1].scatter(x=center_measured[i, 0], y=center_measured[i, 1], c="green", marker="o", label="Measured")
            axes[1].scatter(x=center_after[i, 0], y=center_after[i, 1], c="red", marker="x", label="Predicted")
            #axes[1].scatter(256/2, 256/2, c="blue")
            axes[1].axis("off")

            axes[2].imshow(flux_ray[i].cpu().detach(), cmap="gray")
            axes[2].set_title("After reconstruction align only with sun pos", fontsize=16)
            axes[2].scatter(x=center_measured[i, 0], y=center_measured[i, 1], c="green", marker="o", label="Measured")
            axes[2].scatter(x=center_ray[i, 0], y=center_ray[i, 1], c="red", marker="x", label="Predicted")
            #axes[2].scatter(256/2, 256/2, c="blue")
            axes[2].axis("off")

            axes[3].imshow(flux_measured[i].cpu().detach(), cmap="gray")
            axes[3].set_title("Measured", fontsize=16)
            axes[3].scatter(x=center_measured[i, 0], y=center_measured[i, 1], c="green", marker="o")
            axes[3].axis("off")

            axes[0].legend()
            axes[1].legend()
            axes[2].legend()
            axes[3].legend()

            plt.subplots_adjust(wspace=0.05)
            plt.show()
            plt.savefig(f"./ignored/new/heliostat_{i}_in_group_{group_index}_calibration.png")


#############################################################################################################
# Tutorial
#############################################################################################################

# Set up logger.
set_logger_config()
log = logging.getLogger(__name__)

# Set the device.
device = get_device()

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path("/workVERLEIHNIX/mb/ARTIST/examples/field_optimizations/scenarios/ideal_baseline_scenario.h5")

# Also specify the heliostats to be calibrated and the paths to your calibration-properties.json files.
# Please use the following style: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
# heliostat_data_mapping = [
#     (
#         "heliostat_name_1",
#         [
#             pathlib.Path(
#                 "please/insert/the/path/to/the/paint/data/here/calibration-properties.json"
#             ),
#             # ....
#         ],
#         [
#             pathlib.Path("please/insert/the/path/to/the/paint/data/here/flux.png"),
#             # ....
#         ],
#     ),
#     (
#         "heliostat_name_2",
#         [
#             pathlib.Path(
#                 "please/insert/the/path/to/the/paint/data/here/calibration-properties.json"
#             ),
#             # ....
#         ],
#         [
#             pathlib.Path("please/insert/the/path/to/the/paint/data/here/flux.png"),
#             # ....
#         ],
#     ),
# ]

# Or if you have a directory with downloaded data use this code to create a mapping.
heliostat_data_mapping = paint_scenario_parser.build_heliostat_data_mapping(
    base_path="/workVERLEIHNIX/share/PAINT_data",
    heliostat_names=["AG48", "AG49", "AG50", "AG51", "AG52", "AG53", "AG54", "AG55", "AG56", "AG57", "AG58"],
    #heliostat_names=["AG48", "AG49", "AG50", "AG51", "AG52", "AG53", "AG54", "AG55", "AG56", "AG57", "AG58", "AH48", "AH49", "AH50", "AH55", "AH56", "AH57", "AH58", "AI48", "AI49", "AI50", "AI51", "AI52", "AI53", "AI54", "AI55", "AI56", "AI57", "AI58", "AI59", "AI60", "AI61", "AJ49", "AJ50", "AJ51", "AJ53", "AJ54", "AJ55", "AJ56", "AJ57", "AJ58", "AJ61", "AJ62", "AK48", "AK49", "AK50", "AK51", "AK52", "AK53", "AK54", "AK55", "AK56", "AK58", "AK59", "AK60", "AK61", "AK62", "AL48", "AL49", "AL51", "AL53", "AL56", "AL58", "AL59", "AL60", "AL61", "AL62", "AM48", "AM49", "AM50", "AM51", "AM52", "AM53", "AM54", "AM55", "AM56", "AM57", "AM58", "AM59", "AM60", "AM62", "AN48", "AN49", "AN50", "AN51", "AN52", "AN53", "AN54", "AN55", "AN56", "AN57", "AN58", "AN59", "AN60", "AN61", "AN62"],
    #heliostat_names=['AZ43', 'AP35', 'AA39', 'AC31', 'AC42', 'AC24', 'AM31', 'BF65', 'AQ27', 'AF33', 'BD31', 'AW42', 'AZ39', 'AL36', 'AD42', 'AP39', 'AE28', 'BE38', 'AY55', 'AP50'],
    number_of_measurements=37,
    image_variant="flux",
    randomize=True,
    seed=7,
)

# Configure the optimization.
optimizer_dict = {
    constants.initial_learning_rate_rotation_deviation: 1e-4,
    constants.initial_learning_rate_initial_angles: 1e-3,
    constants.initial_learning_rate_initial_stroke_length: 1e-2,
    constants.tolerance: 0.0000,
    constants.max_epoch: 200,
    constants.batch_size: 50,
    constants.log_step: 1,
    constants.early_stopping_delta: 1e-8,
    constants.early_stopping_patience: 1000,
    constants.early_stopping_window: 2000,
}
# Configure the learning rate scheduler.
scheduler_dict = {
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
optimization_configuration = {
    constants.optimization: optimizer_dict,
    constants.scheduler: scheduler_dict,
}

data_parser = PaintCalibrationDataParser(
    sample_limit=17, centroid_extraction_method=paint_mappings.UTIS_KEY
)
data_parser_plots = PaintCalibrationDataParser(
    sample_limit=1, centroid_extraction_method=paint_mappings.UTIS_KEY
)

# Create dict for the data parser and the heliostat_data_mapping.
data: dict[
    str,
    CalibrationDataParser | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
] = {
    constants.data_parser: data_parser,
    constants.heliostat_data_mapping: heliostat_data_mapping,
}

data_plots: dict[
    str,
    CalibrationDataParser | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
] = {
    constants.data_parser: data_parser_plots,
    constants.heliostat_data_mapping: heliostat_data_mapping,
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
            scenario_file=scenario_file, device=device
        )

    resolution = torch.tensor([256, 256], device=device)

    bitmaps_before, _ = create_fluxes(
        data_parser=data_parser_plots,
        heliostat_data_mapping=[
            (heliostat[0], [heliostat[1][-1]], [heliostat[2][-1]])
            for heliostat in heliostat_data_mapping
        ],
        resolution=resolution,
        method="motor_pos"
    )

    scenario.set_number_of_rays(number_of_rays=4)
    optimization_configuration[constants.optimization][constants.initial_learning_rate_rotation_deviation] = 1e-4
    optimization_configuration[constants.optimization][constants.max_epoch] = 300
    # Create the kinematics reconstructor.
    kinematics_reconstructor = KinematicsReconstructor(
        ddp_setup=ddp_setup,
        scenario=scenario,
        data=data,
        dni=500,
        optimization_configuration=optimization_configuration,
        reconstruction_method=constants.kinematics_reconstruction_geometry,
        bitmap_resolution=resolution,
    )
    # Reconstruct the kinematics.
    final_loss_per_heliostat = kinematics_reconstructor.reconstruct_kinematics(
        loss_definition=AngleLoss(), device=device
    )

    optimization_configuration[constants.optimization][constants.initial_learning_rate_rotation_deviation] = 1e-4
    optimization_configuration[constants.optimization][constants.max_epoch] = 300
    # Create the kinematics reconstructor.
    kinematics_reconstructor = KinematicsReconstructor(
        ddp_setup=ddp_setup,
        scenario=scenario,
        data=data,
        dni=500,
        optimization_configuration=optimization_configuration,
        reconstruction_method=constants.kinematics_reconstruction_raytracing,
        bitmap_resolution=resolution,
    )
    # Reconstruct the kinematics.
    final_loss_per_heliostat = kinematics_reconstructor.reconstruct_kinematics(
        loss_definition=FocalSpotLoss(scenario=scenario), device=device
    )

# Inspect the synchronized loss per heliostat. Heliostats that have not been optimized have an infinite loss.
print(f"rank {ddp_setup['rank']}, final loss per heliostat {final_loss_per_heliostat}")

bitmaps_after, bitmaps_measured = create_fluxes(
    data_parser=data_parser_plots,
    heliostat_data_mapping=[
        (heliostat[0], [heliostat[1][-1]], [heliostat[2][-1]])
        for heliostat in heliostat_data_mapping
    ],
    resolution=resolution,
    method="motor_pos",
)

bitmaps_ray, bitmaps_measured = create_fluxes(
    data_parser=data_parser_plots,
    heliostat_data_mapping=[
        (heliostat[0], [heliostat[1][-1]], [heliostat[2][-1]])
        for heliostat in heliostat_data_mapping
    ],
    resolution=resolution,
    method="incident_ray",
)

create_plots(
    fluxes_before=bitmaps_before,
    fluxes_after=bitmaps_after,
    fluxes_ray=bitmaps_ray,
    fluxes_measured=bitmaps_measured,
)