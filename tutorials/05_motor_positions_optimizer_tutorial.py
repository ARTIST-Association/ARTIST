"""Motor positions optimizer tutorial."""

import pathlib

import h5py
import torch
from matplotlib import pyplot as plt

from artist.flux import bitmap
from artist.optim import MotorPositionsOptimizer
from artist.optim.loss import KLDivergenceLoss
from artist.raytracing import HeliostatRayTracer
from artist.scenario import Scenario
from artist.util import constants, indices, set_logger_config
from artist.util.env import get_device, setup_distributed_environment

torch.manual_seed(7)
torch.cuda.manual_seed(7)

#############################################################################################################
# Define helper functions for the plots.
# Skip to line 115 for the tutorial code.
#############################################################################################################


def create_flux_plot(label: str, resolution: torch.Tensor) -> None:
    """
    Create flux plots.

    Parameters
    ----------
    label : str
        Identifier of flux.
    resolution : torch.Tensor
        Bitmap resolution.
    """
    total_flux = torch.zeros(
        (
            resolution[indices.unbatched_bitmap_u],
            resolution[indices.unbatched_bitmap_e],
        ),
        device=device,
    )

    for heliostat_group_index, heliostat_group in enumerate(
        scenario.heliostat_field.heliostat_groups
    ):
        (active_heliostats_mask, target_area_indices, incident_ray_directions) = (
            scenario.index_mapping(
                heliostat_group=heliostat_group,
                single_incident_ray_direction=incident_ray_direction,
                single_target_area_index=target_area_index,
                device=device,
            )
        )

        # Activate heliostats.
        heliostat_group.activate_heliostats(
            active_heliostats_mask=active_heliostats_mask,
            device=device,
        )

        # Align heliostats.
        if label == "before":
            heliostat_group.align_surfaces_with_incident_ray_directions(
                aim_points=scenario.solar_tower.get_centers_of_target_areas(
                    target_area_indices=target_area_indices, device=device
                ),
                incident_ray_directions=incident_ray_directions,
                active_heliostats_mask=active_heliostats_mask,
                device=device,
            )
        elif label == "after":
            heliostat_group.align_surfaces_with_motor_positions(
                motor_positions=heliostat_group.kinematics.active_motor_positions,
                active_heliostats_mask=active_heliostats_mask,
                device=device,
            )

    for heliostat_group_index, heliostat_group in enumerate(
        scenario.heliostat_field.heliostat_groups
    ):
        (active_heliostats_mask, target_area_indices, incident_ray_directions) = (
            scenario.index_mapping(
                heliostat_group=heliostat_group,
                single_incident_ray_direction=incident_ray_direction,
                single_target_area_index=target_area_index,
                device=device,
            )
        )
        # Create a ray tracer.
        ray_tracer = HeliostatRayTracer(
            scenario=scenario,
            heliostat_group=heliostat_group,
            batch_size=heliostat_group.number_of_active_heliostats,
            bitmap_resolution=resolution,
            dni=dni,
        )

        # Perform heliostat-based ray tracing.
        bitmaps_per_heliostat, _, _, _ = ray_tracer.trace_rays(
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_heliostats_mask,
            target_area_indices=target_area_indices,
            device=device,
        )

        flux_distribution_on_target = ray_tracer.get_bitmaps_per_target(
            bitmaps_per_heliostat=bitmaps_per_heliostat,
            target_area_indices=target_area_indices,
            device=device,
        )[target_area_index]

        total_flux += flux_distribution_on_target

    # Create the plot.
    plt.imshow(total_flux.cpu().detach(), cmap="gray")
    plt.axis("off")
    plt.title(f"Flux {label} aimpoint optimization {total_flux.sum():.3f}")
    plt.savefig(f"flux_{label}_aimpoint_optimization.png")


#############################################################################################################
# Tutorial
#############################################################################################################

# Set up logger.
set_logger_config()

# Set the device.
device = get_device()

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path("please/insert/the/path/to/the/scenario/here/scenario.h5")

# Set optimizer parameters.
optimizer_dict = {
    constants.initial_learning_rate: 3e-4,
    constants.tolerance: 0.0005,
    constants.max_epoch: 100,
    constants.batch_size: 50,
    constants.log_step: 3,
    constants.early_stopping_delta: 1e-4,
    constants.early_stopping_patience: 100,
    constants.early_stopping_window: 100,
}
# Configure the learning rate scheduler.
scheduler_dict = {
    constants.scheduler_type: constants.reduce_on_plateau,
    constants.gamma: 0.9,
    constants.lr_min: 1e-6,
    constants.lr_max: 1e-3,
    constants.step_size_up: 500,
    constants.reduce_factor: 0.3,
    constants.patience: 100,
    constants.threshold: 1e-3,
    constants.cooldown: 10,
}
# Configure the regularizers and constraints.
constraint_dict = {
    constants.rho_flux_integral: 1.0,
    constants.rho_local_flux: 1.0,
    constants.rho_intercept: 1.0,
    constants.max_flux_density: 1000000,
}
# Combine configurations.
optimization_configuration = {
    constants.optimization: optimizer_dict,
    constants.scheduler: scheduler_dict,
    constants.constraints: constraint_dict,
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
            device=device,
        )

    bitmap_resolution = torch.tensor([256, 256], device=device)
    # Set DNI W/m^2.
    dni = 800
    # Set number of rays per surface point.
    scenario.set_number_of_rays(number_of_rays=4)
    # Set incident ray direction.
    incident_ray_direction = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)
    # Set target area.
    target_area_index = 3  # (receiver)
    # Set target flux integral.
    canting_norm = (
        torch.norm(scenario.heliostat_field.heliostat_groups[0].canting[0], dim=1)[0]
    )[:2]
    dimensions = (canting_norm * 4) + 0.02
    heliostat_surface_area = dimensions[0] * dimensions[1]
    total_heliostat_area = (
        heliostat_surface_area
        * scenario.heliostat_field.number_of_heliostats_per_group.sum()
    )
    target_flux_integral = (
        dni * total_heliostat_area * 0.75
    )  # account for mirror and angle based losses.

    # Set loss function and define the ground truth.
    # For an optimization using a focal spot as ground truth use this loss definition:
    # ground_truth = torch.tensor(
    #     [1.1493, -0.5030, 57.0474, 1.0000], device=device
    # )
    # loss_definition = FocalSpotLoss(scenario=scenario)
    # For an optimization using a distribution as target use this loss function definition:
    e_trapezoid = bitmap.trapezoid_distribution(
        total_width=bitmap_resolution[indices.unbatched_bitmap_e],
        slope_width=30,
        plateau_width=110,
        device=device,
    )
    u_trapezoid = bitmap.trapezoid_distribution(
        total_width=bitmap_resolution[indices.unbatched_bitmap_u],
        slope_width=30,
        plateau_width=110,
        device=device,
    )
    ground_truth = u_trapezoid.unsqueeze(
        indices.unbatched_bitmap_u
    ) * e_trapezoid.unsqueeze(indices.unbatched_bitmap_e)
    ground_truth = (ground_truth / ground_truth.sum()) * target_flux_integral

    loss_definition = KLDivergenceLoss()

    create_flux_plot(label="before", resolution=bitmap_resolution)

    # Create the motor positions optimizer.
    motor_positions_optimizer = MotorPositionsOptimizer(
        ddp_setup=ddp_setup,
        scenario=scenario,
        optimization_configuration=optimization_configuration,
        incident_ray_direction=incident_ray_direction,
        target_area_index=target_area_index,
        ground_truth=ground_truth,
        dni=dni,
        bitmap_resolution=bitmap_resolution,
        device=device,
    )

    # Optimize the motor positions.
    final_loss, _, _, _, _ = motor_positions_optimizer.optimize(
        loss_definition=loss_definition, device=device
    )

# Inspect the synchronized loss per heliostat. Heliostats that have not been optimized have an infinite loss.
print(f"rank {ddp_setup['rank']}, final loss {final_loss}")

create_flux_plot(label="after", resolution=bitmap_resolution)
