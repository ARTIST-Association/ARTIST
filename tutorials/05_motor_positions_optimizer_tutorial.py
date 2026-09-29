"""Aim point optimization tutorial."""

import pathlib

import h5py
import torch
from matplotlib import pyplot as plt

from artist.flux import bitmap
from artist.optim import AimPointOptimizer
from artist.optim.loss import KLDivergenceLoss
from artist.raytracing import HeliostatRayTracer
from artist.scenario import Scenario
from artist.util import constants, indices, set_logger_config
from artist.util.env import get_device, setup_distributed_environment

torch.manual_seed(7)
torch.cuda.manual_seed(7)

#############################################################################################################
# Define helper functions for the plots.
# Skip to line 124 for the tutorial code.
#############################################################################################################

def create_flux(label: str, resolution: torch.Tensor) -> torch.Tensor:
    """
    Create flux.

    Parameters
    ----------
    label : str
        Identifier of flux.
    resolution : torch.Tensor
        Bitmap resolution.

    Returns
    -------
    torch.Tensor
        The flux.
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
        scenario.set_number_of_rays(number_of_rays=70)
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

    return total_flux


def plot_flux(
    flux_before: torch.Tensor,
    flux_after: torch.Tensor,
    flux_target: torch.Tensor,
) -> None:
    """
    Plot the fluxes.

    Parameters
    ----------
    fluxes_before : list[torch.Tensor]
        Flux before the aim point optimization.
    fluxes_after : list[torch.Tensor]
        Flux after the aim point optimization.
    flux_target : list[torch.Tensor]
        Target flux.
    """
    fontsize = 6

    fluxes = [flux_before, flux_after, flux_target]
    titles = [
        "a) Central aim points",
        "b) Optimized aim points",
        "c) Target distribution",
    ]

    all_vals = torch.cat( 
        [ 
            torch.cat([x.flatten() for x in flux_before]), 
            torch.cat([x.flatten() for x in flux_after]), 
            torch.cat([x.flatten() for x in flux_target]), 
        ] 
        ) 
    vmin = all_vals.min().item() 
    vmax = all_vals.max().item()

    fig, axes = plt.subplots(
        1, 3,
        figsize=(5, 2.5),
        constrained_layout=True,
    )

    mappable = None
    for ax, flux, title in zip(axes, fluxes, titles):
        mappable = ax.imshow(
            flux.cpu().detach(),
            cmap="hot",
            vmin=vmin,
            vmax=vmax,
        )
        ax.axis("off")

        if title == "c) Target distribution":
            ax.text(
                0.5,
                -0.04,
                f"{title}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=fontsize,
            )
        else:
            ax.text(
                0.5,
                -0.04,
                f"{title}\nIntegrated flux: {(flux.sum().item()/1000):.2f} kW",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=fontsize,
            )

    cbar = fig.colorbar(
        mappable,
        ax=axes,
        shrink=0.55,
        pad=0.02,
        aspect=25,
    )
    cbar.set_label(
        "Power (W)",
        fontsize=fontsize,
    )
    cbar.ax.tick_params(labelsize=fontsize)

    plt.savefig("aimpoint_optimization.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


#############################################################################################################
# Tutorial
#############################################################################################################

# Set up logger.
set_logger_config()

# Set the device.
device = get_device()

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path("please/insert/the/path/to/the/scenario/here/scenario.h5")
reconstruction_data_exists = False

# Set optimizer parameters.
optimizer_dict = {
    constants.initial_learning_rate: 1e-4,
    constants.tolerance: 0.0005,
    constants.max_epoch: 800,
    constants.batch_size: 50,
    constants.log_step: 3,
    constants.early_stopping_delta: 1e-4,
    constants.early_stopping_patience: 1000,
    constants.early_stopping_window: 1000,
}
# Configure the learning rate scheduler.
scheduler_dict = {
    constants.scheduler_type: constants.cyclic,
    constants.gamma: 0.92,
    constants.lr_min: 1e-6,
    constants.lr_max: 1e-3,
    constants.step_size_up: 50,
    constants.reduce_factor: 0.2,
    constants.patience: 30,
    constants.threshold: 1e-3,
    constants.cooldown: 5,
}
# Configure the regularizers and constraints.
constraint_dict = {
    constants.rho_flux_integral: 1.0,
    constants.rho_local_flux: 0.1,
    constants.rho_intercept: 0.01,
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
        if reconstruction_data_exists:
            reconstructed_nurbs_control_points = torch.load(
                "ignored/z_paper/cp.pt", weights_only=False
            )
            reconstructed_kinematics = torch.load(
                "ignored/z_paper/rdp.pt", weights_only=False
            )
            for heliostat_group, control_points, deviation_parameters in zip(
                scenario.heliostat_field.heliostat_groups,
                reconstructed_nurbs_control_points,
                reconstructed_kinematics
            ):
                heliostat_group.nurbs_control_points = control_points
                heliostat_group.kinematics.rotation_deviation_parameters = deviation_parameters
            scenario.heliostat_field.update_surfaces(device=device)

    bitmap_resolution = torch.tensor([256, 256], device=device)
    # Set DNI W/m^2.
    dni = 800
    # Set number of rays per surface point.
    scenario.set_number_of_rays(number_of_rays=4)
    # Set incident ray direction.
    incident_ray_direction = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)
    # Set target area.
    target_area_index = 0 
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

    # Create target flux distribution as ground truth for the optimization.
    e_trapezoid = bitmap.trapezoid_distribution(
        total_width=bitmap_resolution[indices.unbatched_bitmap_e],
        slope_width=40,
        plateau_width=110,
        device=device,
    )
    u_trapezoid = bitmap.trapezoid_distribution(
        total_width=bitmap_resolution[indices.unbatched_bitmap_u],
        slope_width=40,
        plateau_width=110,
        device=device,
    )
    ground_truth = u_trapezoid.unsqueeze(
        indices.unbatched_bitmap_u
    ) * e_trapezoid.unsqueeze(indices.unbatched_bitmap_e)
    ground_truth = (ground_truth / ground_truth.sum()) * target_flux_integral

    loss_definition = KLDivergenceLoss()

    flux_before = create_flux(label="before", resolution=bitmap_resolution)

    # Create the aim point optimizer.
    aim_point_optimizer = AimPointOptimizer(
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
    final_loss, _, _, _, _ = aim_point_optimizer.optimize(
        loss_definition=loss_definition, device=device
    )

# Inspect the synchronized loss per heliostat.
print(f"rank {ddp_setup['rank']}, final loss {final_loss}")

flux_after = create_flux(label="after", resolution=bitmap_resolution)

plot_flux(
    flux_before=flux_before,
    flux_after=flux_after,
    flux_target=ground_truth
)