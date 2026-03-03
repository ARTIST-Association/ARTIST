import argparse
import pathlib
import warnings
from typing import Any

from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

from artist.core.loss_functions import KLDivergenceLoss
from artist.util import utils
from artist.util.environment_setup import get_device

plot_colors = {
    "darkblue": "#002864",
    "lightblue": "#14c8ff",
    "blue_1": "#0057B8",
    "blue_2": "#1F8FE5",
    "blue_3": "#009CA6",
    "blue_4": "#006D6F",
    "darkred": "#cd5c5c",
    "darkgray": "#686868",
}

plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"
plt.rcParams["text.latex.preamble"] = r"\setlength{\parindent}{0pt}"
cmap = "inferno"

def plot_kinematics_reconstruction_flux(
    results: dict[str, Any],
    save_dir: pathlib.Path,
) -> None:
    """
    Plot the kinematic reconstruction flux results.

    Parameters
    ----------
    results : dict[str, Any]
        Results of the ablation study.
    save_dir : pathlib.Path
        Path to the location where the plots are saved.
    """
    reconstructions = [
        "kinematic_reconstruction_ideal_surface",
        "kinematic_reconstruction_reconstructed_surface",
    ]

    col_labels = [
        "Calibration Flux",
        "Default\\\\Kinematics",
        "Reconstructed\\\\Kinematics",
    ]

    for heliostat_index, reconstruction in enumerate(reconstructions):
        heliostat_dict = results[reconstruction]
        heliostat_names = list(heliostat_dict.keys())
        n_rows = len(heliostat_names)
        n_cols = 3

        fig = plt.figure(figsize=(6, 4))
        gs = GridSpec(
            n_rows,
            n_cols,
            figure=fig,
            left=0.02,
            right=0.98,
            top=0.99,
            bottom=0.02,
            wspace=0.01,
            hspace=0.01,
        )
        
        axes = np.empty((n_rows, n_cols), dtype=object)

        for i in range(n_rows):
            for j in range(n_cols):
                ax = fig.add_subplot(gs[i, j])
                ax.axis("off")
                axes[i, j] = ax
        
        for col_index, label in enumerate(col_labels):
            axes[0, col_index].set_title(
                rf"\textbf{{{label}}}",
                fontsize=13,
                ha="center",
            )

        for row_index, heliostat_name in enumerate(heliostat_names):
            flux_data = (
                heliostat_dict[heliostat_name]["fluxes"].detach().cpu()
            )

            position = results["heliostat_positions"][heliostat_name]
            position_str = ", ".join(f"{x:.2f}" for x in position[:3])

            for col_index in range(n_cols):
                axes[row_index, col_index].imshow(flux_data[col_index], cmap=cmap)

            ax_left = axes[row_index, 0]

            ax_left.text(
                -0.05,
                0.5,
                rf"\textbf{{Heliostat: {heliostat_name}}}",
                transform=ax_left.transAxes,
                fontsize=13,
                ha="right",
                va="center",
            )

            ax_left.text(
                -0.05,
                0.4,
                r"\textit{ENU Position:}",
                transform=ax_left.transAxes,
                fontsize=12,
                color=plot_colors["darkgray"],
                ha="right",
                va="center",
            )

            ax_left.text(
                -0.05,
                0.30,
                rf"\textit{{{position_str}}}",
                transform=ax_left.transAxes,
                fontsize=12,
                color=plot_colors["darkgray"],
                ha="right",
                va="center",
            )

            ax_left.set_ylabel(
                heliostat_name,
                rotation=0,
                ha="right",
                va="center",
                fontsize=13,
            )

        save_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            save_dir
            / f"kinematics_reconstruction_fluxes_{heliostat_index}_{results_number}.pdf"
        )
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved kinematics reconstruction flux plot at: {filename}.")


def plot_kinematics_error_distribution(
    results: dict[str, Any],
    save_dir: pathlib.Path,
) -> None:
    """
    Plot the kinematic reconstruction error distribution.

    Parameters
    ----------
    results : dict[str, Any]
        Results of the ablation study.
    save_dir : pathlib.Path
        Path to the location where the plots are saved.
    """
    cases = ["ablation_study_case_3", "ablation_study_case_7"]
    case_labels = ["Ideal Surfaces", "Reconstructed Surfaces"]
    case_colors = ["darkblue", "lightblue"]
    errors_dict = {}

    for case in cases:
        errors_in_meters = np.array(
            results[case]["kinematic_reconstruction_loss_per_heliostat"].detach().cpu()
        )
        positions = list(results["heliostat_positions"].values())
        distances = np.linalg.norm(np.array(positions)[:, :3], axis=1)
        errors_in_mrad = (errors_in_meters / distances) * 1000
        errors_dict[case] = {
            "meters": errors_in_meters,
            "mrad": errors_in_mrad
        }

    for unit in ["meters", "mrad"]:
        fig, ax = plt.subplots(figsize=(6, 4))

        for case, label, color in zip(cases, case_labels, case_colors):
            errors = errors_dict[case][unit]
            x_max = max(errors)
            x_vals = np.linspace(0, x_max, 100)
            kde = gaussian_kde(errors, bw_method="scott")
            kde_values = kde(x_vals)
            mean = np.mean(errors)

            ax.hist(
                errors,
                bins=25,
                range=(0, x_max),
                density=True,
                alpha=0.3,
                label=f"{label} Histogram",
                color=plot_colors[color]
            )
            ax.plot(
                x_vals,
                kde_values,
                label=f"{label} KDE",
                color=plot_colors[color]
            )
            ax.axvline(
                mean,
                color=plot_colors[color],
                linestyle="--",
                label=f"{label} Mean: {mean:.2f} {unit}"
            )

        ax.set_xlabel(f"\\textbf{{Pointing Error}} \n{{\\small {unit}}}")
        ax.set_ylabel("\\textbf{Density}")
        ax.legend(fontsize=8)
        ax.grid(True)

        if not save_dir.is_dir():
            save_dir.mkdir(parents=True, exist_ok=True)
        filename = save_dir / f"kinematics_reconstruction_error_distribution_{unit}_{results_number}.pdf"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved kinematics reconstruction error distribution plot at: {filename}.")

def plot_kinematics_error_against_distance(
    results: dict[str, Any],
    save_dir: pathlib.Path,
) -> None:
    """
    Plot the kinematic reconstruction error against the distance to the tower.

    Parameters
    ----------
    results : dict[str, Any]
        Results of the ablation study.
    save_dir : pathlib.Path
        Path to the location where the plots are saved.
    """
    cases = ["ablation_study_case_3", "ablation_study_case_7"]

    for plot_index, case in enumerate(cases):
        errors_in_meters = np.array(
            results[case]["kinematic_reconstruction_loss_per_heliostat"].detach().cpu()
        )
        positions = list(results["heliostat_positions"].values())
        distances = np.linalg.norm(np.array(positions)[:, :3], axis=1)
        errors_in_mrad = (errors_in_meters / distances) * 1000

        fig, ax_m = plt.subplots(figsize=(7, 4))
        ax_m.scatter(
            distances,
            errors_in_meters,
            color=plot_colors["lightblue"],
            marker="o",
            label="Error (m)",
            alpha=0.7,
        )

        fit_meters = np.poly1d(np.polyfit(distances, errors_in_meters, 1))
        x_vals = np.linspace(distances.min(), distances.max(), 200)
        ax_m.plot(
            x_vals, fit_meters(x_vals), color=plot_colors["lightblue"], linestyle="--"
        )
        ax_m.set_xlabel("\\textbf{Heliostat Distance from Tower [m]}")
        ax_m.set_ylabel(
            "\\textbf{Mean Pointing Error [m]}",
            color=plot_colors["lightblue"],
        )
        ax_m.grid(True)

        ax_a = ax_m.twinx()
        ax_a.scatter(
            distances,
            errors_in_mrad,
            color=plot_colors["darkblue"],
            marker="^",
            label="Error (mrad)",
            alpha=0.7,
        )

        fit_a = np.poly1d(np.polyfit(distances, errors_in_mrad, 1))
        ax_a.plot(x_vals, fit_a(x_vals), color="darkblue", linestyle="--")
        ax_a.set_ylabel("\\textbf{Mean Pointing Error [mrad]}", color="darkblue")
        ax_a.tick_params(axis="y", labelcolor="black")

        handles_m, labels_m = ax_m.get_legend_handles_labels()
        handles_a, labels_a = ax_a.get_legend_handles_labels()
        ax_m.legend(
            handles_m + handles_a,
            labels_m + labels_a,
            fontsize=8,
            loc="upper right",
            ncol=2,
        )

        if not save_dir.is_dir():
            save_dir.mkdir(parents=True, exist_ok=True)
        filename = save_dir / f"kinematics_reconstruction_error_distance_{plot_index}_{results_number}.pdf"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved kinematics reconstruction error distance plot at: {filename}.")

def plot_kinematics_loss_history(
    results: dict[str, Any],
    save_dir: pathlib.Path,
) -> None:
    """
    Plot the kinematic reconstruction loss history.

    Parameters
    ----------
    results : dict[str, Any]
        Results of the ablation study.
    save_dir : pathlib.Path
        Path to the location where the plots are saved.
    """
    cases = ["ablation_study_case_3", "ablation_study_case_7"]

    for plot_index, case in enumerate(cases):
        loss_history = results[case]["loss_histories"][1]
        epochs = np.arange(0, len(loss_history["total_loss_history"]))
        
        fig, ax1 = plt.subplots(figsize=(8, 5))

        ax1.plot(
            epochs,
            loss_history["total_loss_history"],
            label=r"Total Loss",
            color=plot_colors["darkblue"],
        )

        ax1.set_xlabel(r"Epoch")
        ax1.set_ylabel(r"Loss Terms")
        ax1.grid(True)
        ax1.legend(loc="upper right")

        ax1.set_title(r"\textbf{Loss History}", fontsize=13, ha="center")

        if not save_dir.is_dir():
            save_dir.mkdir(parents=True, exist_ok=True)
        filename = save_dir / f"kinematics_reconstruction_loss_history_{plot_index}_{results_number}.pdf"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved kinematics reconstructions loss history plot at: {filename}.")

def plot_surface_reconstruction_flux(
    results: dict[str, Any],
    save_dir: pathlib.Path,
) -> None:
    """
    Plot the surface reconstruction flux results.

    Parameters
    ----------
    results : dict[str, Any]
        Results of the ablation study.
    save_dir : pathlib.Path
        Path to the location where the plots are saved.
    """
    number_of_heliostats = len(results["surface_reconstruction"])
    fig, axes = plt.subplots(number_of_heliostats, 7, figsize=(35, 15))
    for index, heliostat_name in enumerate(results["surface_reconstruction"]):
        heliostat_data = results["surface_reconstruction"][heliostat_name]
        axes[index, 0].imshow(
            heliostat_data["fluxes"][0].cpu().detach(), cmap="inferno"
        )
        axes[index, 0].set_title("Calibration Flux")
        axes[index, 0].axis("off")

        axes[index, 1].imshow(
            heliostat_data["fluxes"][1].cpu().detach(), cmap="inferno"
        )
        axes[index, 1].set_title("Surface not reconstructed")
        axes[index, 1].axis("off")

        axes[index, 2].imshow(
            heliostat_data["fluxes"][2].cpu().detach(), cmap="inferno"
        )
        axes[index, 2].set_title("Surface reconstructed")
        axes[index, 2].axis("off")

        reference_direction = torch.tensor([0.0, 0.0, 1.0], device=torch.device("cpu"))
        canting = heliostat_data["canting"].cpu().detach()

        # Process original deflectometry data.
        deflectometry_original = (
            results["deflectometry_original"][heliostat_name].cpu().detach()
        )
        ones = torch.ones_like(deflectometry_original, device=torch.device("cpu"))
        deflectometry_original = torch.cat(
            (deflectometry_original, ones[..., 0, None]), dim=-1
        )
        deflectometry_uncanted_original = utils.perform_canting(
            canting_angles=canting.expand(2, -1, -1, -1),
            data=deflectometry_original,
            inverse=True,
            device=torch.device("cpu"),
        )
        deflectometry_points_original = deflectometry_uncanted_original[
            0, :, :, :3
        ].reshape(-1, 3)
        deflectometry_normals_original = torch.nn.functional.normalize(
            deflectometry_uncanted_original[1, :, :, :3], dim=-1
        ).reshape(-1, 3)
        cos_theta_deflectometry_original = (
            deflectometry_normals_original @ reference_direction
        )
        angles_deflectometry_original = torch.clip(
            torch.arccos(torch.clip(cos_theta_deflectometry_original, -1.0, 1.0)),
            -0.1,
            0.1,
        )
        sc3 = axes[index, 3].scatter(
            x=deflectometry_points_original[:, 0],
            y=deflectometry_points_original[:, 1],
            c=deflectometry_points_original[:, 2],
            cmap="inferno",
            vmin=0.0345,
            vmax=0.036,
        )
        axes[index, 3].set_title("Deflectometry Points original")
        axes[index, 3].axis("off")
        axes[index, 3].set_aspect("equal", adjustable="box")
        cbar3 = fig.colorbar(
            sc3, ax=axes[index, 3], orientation="horizontal", fraction=0.046, pad=0.1
        )
        cbar3.set_label("m")

        sc4 = axes[index, 4].scatter(
            x=deflectometry_points_original[:, 0],
            y=deflectometry_points_original[:, 1],
            c=angles_deflectometry_original,
            cmap="inferno",
            vmin=0.0,
            vmax=0.005,
        )
        axes[index, 4].set_title("Deflectometry normals")
        axes[index, 4].axis("off")
        axes[index, 4].set_aspect("equal", adjustable="box")
        cbar4 = fig.colorbar(
            sc4, ax=axes[index, 4], orientation="horizontal", fraction=0.046, pad=0.1
        )
        cbar4.set_label("Angle (rad)")

        # Process reconstructed data.
        points_uncanted = utils.perform_canting(
            canting_angles=canting.expand(2, -1, -1, -1),
            data=heliostat_data["surface_points"].cpu().detach().reshape(2, 4, -1, 4),
            inverse=True,
            device=torch.device("cpu"),
        )
        normals_uncanted = utils.perform_canting(
            canting_angles=canting.expand(2, -1, -1, -1),
            data=heliostat_data["surface_normals"].cpu().detach().reshape(2, 4, -1, 4),
            inverse=True,
            device=torch.device("cpu"),
        )
        reconstructed_points = points_uncanted[1, :, :, :3].reshape(-1, 3)
        reconstructed_normals = torch.nn.functional.normalize(
            normals_uncanted[1, :, :, :3], dim=-1
        ).reshape(-1, 3)
        cos_theta_reconstructed = reconstructed_normals @ reference_direction
        angles_reconstructed = torch.clip(
            torch.arccos(torch.clip(cos_theta_reconstructed, -1.0, 1.0)), -0.1, 0.1
        )
        sc5 = axes[index, 5].scatter(
            x=reconstructed_points[:, 0],
            y=reconstructed_points[:, 1],
            c=reconstructed_points[:, 2],
            cmap="inferno",
            vmin=0.0345,
            vmax=0.036,
        )
        axes[index, 5].set_title("Reconstructed Surface (Points)")
        axes[index, 5].axis("off")
        axes[index, 5].set_aspect("equal", adjustable="box")
        cbar5 = fig.colorbar(
            sc5, ax=axes[index, 5], orientation="horizontal", fraction=0.046, pad=0.1
        )
        cbar5.set_label("m")

        sc6 = axes[index, 6].scatter(
            x=reconstructed_points[:, 0],
            y=reconstructed_points[:, 1],
            c=angles_reconstructed,
            cmap="inferno",
            vmin=0.0,
            vmax=0.005,
        )
        axes[index, 6].set_title("Reconstructed normals")
        axes[index, 6].axis("off")
        axes[index, 6].set_aspect("equal", adjustable="box")
        cbar6 = fig.colorbar(
            sc6, ax=axes[index, 6], orientation="horizontal", fraction=0.046, pad=0.1
        )
        cbar6.set_label("Angle (rad)")

    plt.tight_layout()
    plt.savefig(
        save_dir / f"surface_reconstruction_flux_{results_number}.png",
        bbox_inches="tight",
        pad_inches=1,
    )

def plot_surface_error_distribution(
    results: dict[str, Any],
    save_dir: pathlib.Path,
) -> None:
    """
    Plot the surface reconstruction error distribution.

    Parameters
    ----------
    results : dict[str, Any]
        Results of the ablation study.
    save_dir : pathlib.Path
        Path to the location where the plots are saved.
    """
    losses_list = results["ablation_study_case_5"]["surface_reconstruction_loss_per_heliostat"]

    safe_losses = []
    for loss_list in losses_list:
        safe_losses.append(torch.where(torch.isinf(loss_list), torch.tensor(1e6, device=loss_list.device), loss_list))
    errors = (torch.min(torch.stack(safe_losses, dim=0), dim=0).values).cpu().detach().numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    x_max = np.max(errors)
    x_vals = np.linspace(0, x_max, 100)
    kde = gaussian_kde(errors, bw_method="scott")
    kde_values = kde(x_vals)
    mean = np.mean(errors)
    ax.hist(
        errors,
        bins=25,
        range=(0, x_max),
        density=True,
        alpha=0.3,
        label="Histogram KL-Div",
        color=plot_colors["lightblue"]
    )
    ax.plot(
        x_vals,
        kde_values,
        label="KDE",
        color=plot_colors["lightblue"]
    )
    ax.axvline(
        mean,
        color=plot_colors["lightblue"],
        linestyle="--",
        label=f"Mean: {mean:.2f}"
    )

    ax.set_xlabel(r"\textbf{KL-Divergence}")
    ax.set_ylabel(r"\textbf{Density}")
    ax.legend(fontsize=8)
    ax.grid(True)

    save_dir.mkdir(parents=True, exist_ok=True)
    filename = save_dir / f"surface_reconstruction_error_distribution_{results_number}.pdf"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved surface reconstruction error distribution plot at: {filename}.")

def plot_surface_error_against_distance(
    results: dict[str, Any],
    save_dir: pathlib.Path,
) -> None:
    """
    Plot the surface reconstruction error against the distance to the tower.

    Parameters
    ----------
    results : dict[str, Any]
        Results of the ablation study.
    save_dir : pathlib.Path
        Path to the location where the plots are saved.
    """
    losses_list = results["ablation_study_case_5"]["surface_reconstruction_loss_per_heliostat"]
    safe_losses = []
    for loss_list in losses_list:
        safe_losses.append(torch.where(torch.isinf(loss_list), torch.tensor(1e6, device=loss_list.device), loss_list)
    )
    errors = (torch.min(torch.stack(safe_losses, dim=0), dim=0).values).cpu().detach().numpy()

    positions = list(results["heliostat_positions"].values())
    distances = np.linalg.norm(np.array(positions)[:, :3], axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(
        distances,
        errors,
        color=plot_colors["lightblue"],
        marker="o",
        label="Error (Combined Loss Terms)",
        alpha=0.7,
    )

    fit_meters = np.poly1d(np.polyfit(distances, errors, 1))
    x_vals = np.linspace(distances.min(), distances.max(), 200)
    ax.plot(
        x_vals, fit_meters(x_vals), color=plot_colors["lightblue"], linestyle="--"
    )
    ax.set_xlabel("\\textbf{Heliostat Distance from Tower [m]}")
    ax.set_ylabel(
        "\\textbf{Mean Reconstruction Error [m]}",
        color=plot_colors["lightblue"],
    )
    ax.grid(True)
    ax.legend(loc="upper right")

    if not save_dir.is_dir():
        save_dir.mkdir(parents=True, exist_ok=True)
    filename = save_dir / f"surface_reconstruction_error_distance_{results_number}.pdf"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved surface reconstruction error distance plot at: {filename}.")


def plot_surface_loss_history(
    results: dict[str, Any],
    save_dir: pathlib.Path,
) -> None:
    """
    Plot the surface reconstruction loss history.

    Parameters
    ----------
    results : dict[str, Any]
        Results of the ablation study.
    save_dir : pathlib.Path
        Path to the location where the plots are saved.
    """
    loss_history = results["ablation_study_case_5"]["loss_histories"][0]
    epochs = np.arange(0, len(loss_history[0]["total_loss_history"]))
    
    for batch_index, batch in enumerate(loss_history):
        fig, ax1 = plt.subplots(figsize=(8, 5))

        l1, = ax1.plot(
            epochs,
            batch["total_loss_history"],
            label=r"Total Loss",
            color=plot_colors["darkblue"],
        )

        l2, = ax1.plot(
            epochs,
            batch["flux_loss_history"],
            label=r"KL-Divergence",
            color=plot_colors["blue_1"],
        )

        l3, = ax1.plot(
            epochs,
            batch["ideal_history"],
            label=r"Ideal Surface Regularization",
            color=plot_colors["blue_2"],
        )

        l4, = ax1.plot(
            epochs,
            batch["smoothness_history"],
            label=r"Smooth Surface Regularization",
            color=plot_colors["blue_3"],
        )

        l5, = ax1.plot(
            epochs,
            batch["energy_history"],
            label=r"Energy Constraint",
            color=plot_colors["blue_4"],
        )

        ax1.set_xlabel(r"Epoch")
        ax1.set_ylabel(r"Loss Terms")
        ax1.grid(True)
        
        ax2 = ax1.twinx()
        l6, = ax2.plot(
            epochs,
            batch["energy_gain"],
            label=r"Energy Gain %",
            color=plot_colors["darkred"],
        )

        ax2.set_ylabel(r"Energy Gain \%")
        lines = [l1, l2, l3, l4, l5, l6]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="upper right")

        ax1.set_title(r"\textbf{Loss History}", fontsize=13, ha="center")

        if not save_dir.is_dir():
            save_dir.mkdir(parents=True, exist_ok=True)
        filename = save_dir / f"surface_reconstruction_loss_history_{batch_index}_{results_number}.pdf"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved surface reconstruction loss history plot at: {filename}.")

def plot_aim_point_flux(
    results: dict[str, Any],
    save_dir: pathlib.Path,
) -> None:
    """
    Plot the aim point optimization flux results.

    Parameters
    ----------
    results : dict[str, Any]
        Results of the ablation study.
    save_dir : pathlib.Path
        Path to the location where the plots are saved.
    """
    measured_flux = results["measured_flux"]
    homogeneous_distribution = results["homogeneous_distribution"]
    unoptimized_flux = results["ablation_study_case_7"]["flux"].squeeze(0)
    optimized_flux = results["ablation_study_case_8"]["flux"].squeeze(0)

    measured_flux_normed = measured_flux * (unoptimized_flux.sum() / measured_flux.sum())
    homogeneous_distribution_normed = homogeneous_distribution * (optimized_flux.sum() / homogeneous_distribution.sum())

    # 1. KL-Divergence between measured flux and homogeneous distribution.
    # 2. KL-Divergence between reconstructed field but unoptimized aim points and homogenous distribution.
    # 3. KL-Divergence between reconstructed field with optimized aim points and homogeneous distribution.
    kl_divergence = KLDivergenceLoss()
    kl_div_measured_homogeneous = kl_divergence(
        measured_flux_normed.unsqueeze(0),
        homogeneous_distribution.unsqueeze(0),
        reduction_dimensions=(1, 2),
    )
    kl_div_unoptimized_homogeneous = kl_divergence(
        unoptimized_flux.unsqueeze(0),
        homogeneous_distribution.unsqueeze(0),
        reduction_dimensions=(1, 2),
    )
    kl_div_optimized_homogeneous = kl_divergence(
        optimized_flux.unsqueeze(0),
        homogeneous_distribution.unsqueeze(0),
        reduction_dimensions=(1, 2),
    )
    improvement = (100/unoptimized_flux.sum()) * optimized_flux.sum()

    images = [
        measured_flux_normed.cpu().detach(),
        unoptimized_flux.cpu().detach(),
        optimized_flux.cpu().detach(),
        homogeneous_distribution_normed.cpu().detach(),
    ]
    vmin = min(img.min() for img in images)
    vmax = max(img.max() for img in images)

    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(
        2,
        len(images),
        figure=fig,
        left=0.05,
        right=0.95,
        top=0.95,
        bottom=0.15,
        height_ratios=[1, 0.05],
        wspace=0.01,
        hspace=0.01,
    )
    axes = []
    
    kl_divs = [
        rf"$\mathrm{{KL}}(\mathrm{{M}} \,\|\, \mathrm{{H}}) = {kl_div_measured_homogeneous.item():.4f}$",
        rf"$\mathrm{{KL}}(\mathrm{{U}} \,\|\, \mathrm{{H}}) = {kl_div_unoptimized_homogeneous.item():.4f}$",
        rf"$\mathrm{{KL}}(\mathrm{{O}} \,\|\, \mathrm{{H}}) = {kl_div_optimized_homogeneous.item():.4f} $",
        ""
    ]
    improvement_labels = [
        "",
        rf"$\mathrm{{100\%}}$",
        rf"$\mathrm{improvement:.2f}\%$",
        ""
    ]
    for index, flux in enumerate(images):
        ax = fig.add_subplot(gs[0, index])
        ax.axis("off")
        im = ax.imshow(flux, cmap=cmap, vmin=vmin, vmax=vmax)
        axes.append(ax)
        pos = ax.get_position()
        fig.text(
            x=pos.x0 + pos.width / 2,
            y=pos.y0 - 0.03,
            s=f"Flux integral: {flux.sum():.2f}\n{kl_divs[index]}\n{improvement_labels[index]}",
            ha="center",
            va="top",
            fontsize=12,
        )
    
    axes[0].set_title(r"\textbf{Baseline Aim Point}", fontsize=13, ha="center")
    axes[1].set_title(r"\textbf{Reconstructed Model}", fontsize=13, ha="center")
    axes[2].set_title(r"\textbf{Optimized Aim Points}", fontsize=13, ha="center")
    axes[3].set_title(r"\textbf{Homogeneous Distribution}", fontsize=13, ha="center")

    cbar_ax = fig.add_subplot(gs[1, :])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=14)

    fig.text(
        0.95, 0.05,
        "M = Measured flux\nU = Unoptimized flux\nO = Optimized flux,\nH = Homogeneous flux",
        ha="right", va="bottom",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=3)
    )

    if not save_dir.is_dir():
        save_dir.mkdir(parents=True, exist_ok=True)
    filename = save_dir / f"aim_point_optimization_flux_{results_number}.pdf"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved aim point optimization flux plot at: {filename}.")

def plot_aim_point_loss_history(
    results: dict[str, Any],
    save_dir: pathlib.Path,
) -> None:
    """
    Plot the aim point optimization loss history.

    Parameters
    ----------
    results : dict[str, Any]
        Results of the ablation study.
    save_dir : pathlib.Path
        Path to the location where the plots are saved.
    """
    loss_history = results["ablation_study_case_8"]["loss_histories"][2]
    epochs = np.arange(0, len(loss_history["total_loss_history"]))
    
    fig, ax1 = plt.subplots(figsize=(8, 5))

    l1, = ax1.plot(
        epochs,
        loss_history["total_loss_history"],
        label=r"Total Loss",
        color=plot_colors["darkblue"],
    )

    l2, = ax1.plot(
        epochs,
        loss_history["flux_loss_history"],
        label=r"KL-Divergence",
        color=plot_colors["blue_1"],
    )

    l3, = ax1.plot(
        epochs,
        loss_history["energy_reward_history"],
        label=r"Energy Integral",
        color=plot_colors["blue_2"],
    )

    l4, = ax1.plot(
        epochs,
        loss_history["pixel_constraint_history"],
        label=r"Maximum Flux Density",
        color=plot_colors["blue_3"],
    )

    ax1.set_xlabel(r"Epoch")
    ax1.set_ylabel(r"Loss Terms")
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    l5, = ax2.plot(
        epochs,
        loss_history["energy_gain"],
        label=r"Energy Gain %",
        color=plot_colors["darkred"],
    )

    ax2.set_ylabel(r"Energy Gain \%")
    lines = [l1, l2, l3, l4, l5]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")

    ax1.set_title(r"\textbf{Loss History}", fontsize=13, ha="center")

    if not save_dir.is_dir():
        save_dir.mkdir(parents=True, exist_ok=True)
    filename = save_dir / f"aim_point_optimization_loss_history_{results_number}.pdf"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved aim point optimization loss history plot at: {filename}.")


def plot_tradeoffs(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Panel A: Stacked runtime per task
    sns.barplot(
        data=df, x="run_id", y="runtime", hue="task", ax=axes[0,0], ci=None
    )
    axes[0,0].set_title("Task-wise Runtime")

    # Panel B: Loss per task
    sns.barplot(
        data=df, x="run_id", y="loss", hue="optimized", ax=axes[0,1], ci=None
    )
    axes[0,1].set_title("Task-wise Loss")

    # Panel C: Total runtime vs total loss
    # df_runs = df.groupby(["hardware","run_id"]).agg(
    #     total_runtime=("total_runtime","first"),
    #     total_loss=("loss","sum")
    # ).reset_index()
    # sns.scatterplot(
    #     data=df_runs, x="total_runtime", y="total_loss",
    #     hue="hardware", style="run_id", s=100, ax=axes[1,0]
    # )
    # axes[1,0].set_title("Runtime vs Cumulative Loss (Pareto)")

    # Panel D: Heatmap example for surface task
    # df_surface = df[df["task"]=="surface"]
    # pivot = df_surface.pivot_table(
    #     index="batch_size_outer", columns="max_epoch", values="runtime", aggfunc="mean"
    # )
    # sns.heatmap(pivot, annot=True, ax=axes[1,1])
    # axes[1,1].set_title("Surface Runtime Heatmap")

    plt.tight_layout()
    plt.savefig("test")

def build_dataframe(results: list[dict[str, Any]], save_dir: pathlib.Path):
    rows = []
    tasks = {"surface": "surface_reconstruction_loss_per_heliostat", "kinematic": "kinematic_reconstruction_loss_per_heliostat", "aim_points": "aimpoint_optimization_loss_per_heliostat"}
    parameters = {"surface": ["max_epoch", "batch_size", "batch_size_outer", "number_of_rays"], "kinematics": ["max_epoch", "batch_size", "number_of_rays"], "aim_points": ["max_epoch", "batch_size", "number_of_rays"]}
    
    for run in results:
        hw = "NVIDIA RTX A6000"
        run_id = run["run_info"]["run_id"]
        total_runtime = np.sum(run["run_info"]["runtimes"])
        for i in range(1, 9):
            study_case = run[f"ablation_study_case_{i}"]
            if i == 5:
                study_case["surface_reconstruction_loss_per_heliostat"] = torch.min(torch.stack(study_case["surface_reconstruction_loss_per_heliostat"], dim=0), dim=0).values
            for key, value in zip(tasks.keys(), tasks.values()):
                row = {
                    "hardware": hw,
                    "run_id": run_id,
                    "ablation_case": i,
                    "task": key,
                    "optimized": True if study_case[value] is not None else False,
                    "runtime": run["run_info"]["runtimes"][i-1],
                    "loss": np.mean(list(study_case[value].cpu().detach())) if study_case[value] is not None else np.nan,
                    "total_runtime": total_runtime
                }
                # Add task-specific parameters
                for parameter in parameters:
                    row[parameter] = run["run_info"]["parameters"][parameter]
                rows.append(row)
    df = pd.DataFrame(rows)
    
    plot_tradeoffs(df)

if __name__ == "__main__":
    """
    Generate plots based on the kinematic reconstruction results.

    This script loads the results from the ``ARTIST`` reconstruction and generates two plots, one comparing the loss when
    using different centroid extraction methods and one comparing the loss as a function of distance from the tower.

    Parameters
    ----------
    config : str
        Path to the configuration file.
    device : str
        Device to use for the computation.
    results_dir : str
        Path to directory where the results are saved.
    plots_dir : str
        Path to the directory where the plots are saved.
    number_of_points_to_plot : int
        Number of data points to plot in the distance error plot.
    random_seed : int
        Random seed for the selection of points to plot.
    """

    # Set default location for configuration file.
    script_dir = pathlib.Path(__file__).resolve().parent
    default_config_path = script_dir / "config.yaml"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML configuration file.",
        default=default_config_path,
    )

    # Parse the config argument first to load the configuration.
    args, unknown = parser.parse_known_args()
    config_path = pathlib.Path(args.config)
    config = {}
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            warnings.warn(f"Error parsing YAML file: {exc}.")
    else:
        warnings.warn(
            f"Warning: Configuration file not found at {config_path}. Using defaults."
        )

    # Add remaining arguments to the parser with defaults loaded from the config.
    device_default = config.get("device", "cuda")
    results_dir_default = config.get("results_dir", "./results")
    plots_dir_default = config.get("plots_dir", "./plots")

    parser.add_argument(
        "--device",
        type=str,
        help="Device to use.",
        default=device_default,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Path to load the results.",
        default=results_dir_default,
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        help="Path to save the plots.",
        default=plots_dir_default,
    )

    # Re-parse the full set of arguments.
    args = parser.parse_args(args=unknown)

    device = get_device(torch.device(args.device))

    # for case in ["baseline", "full_field"]:
    for case in ["baseline"]:
        plots_path = pathlib.Path(args.plots_dir) / case

        plots_path.mkdir(parents=True, exist_ok=True)
        results_number = 9
        results_path = pathlib.Path(args.results_dir) / case / f"results_{results_number}.pt"

        if not results_path.exists():
            raise FileNotFoundError(
                f"Results file not found: {results_path}. Please run ``reconstruction_generate_results.py``"
                f"or adjust the location of the results file and try again!"
            )

        results = torch.load(
            results_path,
            weights_only=False,
            map_location=device,
        )

        dataframe_results = [results]

        build_dataframe(results=dataframe_results, save_dir=plots_path)

        #plot_surface_reconstruction_flux(results=results, save_dir=plots_path)        
        #plot_surface_error_distribution(results=results, save_dir=plots_path)
        #plot_surface_error_against_distance(results=results, save_dir=plots_path)
        #plot_surface_loss_history(results=results, save_dir=plots_path)

        # plot_kinematics_reconstruction_flux(results=results, save_dir=plots_path)
        # plot_kinematics_error_distribution(results=results, save_dir=plots_path) 
        # plot_kinematics_error_against_distance(results=results, save_dir=plots_path)
        # plot_kinematics_loss_history(results=results, save_dir=plots_path)

        # plot_aim_point_flux(results=results, save_dir=plots_path)        
        # plot_aim_point_loss_history(results=results, save_dir=plots_path)
        