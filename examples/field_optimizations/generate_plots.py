import argparse
import pathlib
import warnings
from typing import Any

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
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
plt.rcParams["text.latex.preamble"] = r"""
\usepackage{cmbright}
\setlength{\parindent}{0pt}
"""
cmap = "inferno"


def plot_heliostat_positions(
    results: dict[str, Any], results_ftp: dict[str, Any], save_dir: pathlib.Path
) -> None:
    """
    Plot heliostat positions.

    Parameters
    ----------
    results : dict[str, Any]
        Results of the baseline study.
    results_ftp : dict[str, Any]
        Results of the full field study.
    save_dir : pathlib.Path
        Directory to save the plots.
    """
    positions = list(results["heliostat_positions"].values())

    x = [pos[0] for pos in positions]
    y = [pos[1] for pos in positions]

    positions_all_heliostats = list(results_ftp["heliostat_positions"].values())
    x_all = [pos[0] for pos in positions_all_heliostats]
    y_all = [pos[1] for pos in positions_all_heliostats]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x=x_all, y=y_all, c=plot_colors["lightblue"], s=10)

    ax.scatter(
        x,
        y,
        facecolors="none",
        edgecolors="red",
        s=2,
        linewidths=2,
    )

    ax.plot([-2 / 2, 2 / 2], [0, 0], color="red", linewidth=2)
    ax.grid(True)

    ax.set_xlabel("\\textbf{East-West distance to tower [m]}")
    ax.set_ylabel("\\textbf{North-South distance to tower [m]}")
    ax.grid(True)

    save_dir.mkdir(parents=True, exist_ok=True)
    filename = save_dir / "heliostat_positions.pdf"
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved heliostat position plot at: {filename}.")


def plot_surface_reconstruction_flux(
    results: dict[str, Any],
    save_dir: pathlib.Path,
    results_number: int,
) -> None:
    """
    Plot the surface reconstruction flux results.

    Parameters
    ----------
    results : dict[str, Any]
        Results of the study.
    save_dir : pathlib.Path
        Path to the location where the plots are saved.
    results_number : int
        Identifier of the results run.
    """
    plot_data = results["surface_reconstruction"]["flux_plot_data"]
    number_of_heliostats = len(plot_data)
    fig, axes = plt.subplots(number_of_heliostats, 7, figsize=(35, 15))
    fontsize = 28
    for index in range(1):
        heliostat_name = "AK54"
        heliostat_data = plot_data[heliostat_name]
        axes[index, 0].imshow(
            heliostat_data["fluxes"][0].cpu().detach(), cmap="inferno"
        )
        axes[index, 0].set_title("Calibration Flux", fontsize=fontsize)
        axes[index, 0].axis("off")

        axes[index, 1].imshow(
            heliostat_data["fluxes"][1].cpu().detach(), cmap="inferno"
        )
        axes[index, 1].set_title("Surface not reconstructed", fontsize=fontsize)
        axes[index, 1].axis("off")

        axes[index, 2].imshow(
            heliostat_data["fluxes"][2].cpu().detach(), cmap="inferno"
        )
        axes[index, 2].set_title("Surface reconstructed", fontsize=fontsize)
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
        axes[index, 3].set_title("Deflectometry Points", fontsize=fontsize)
        axes[index, 3].axis("off")
        axes[index, 3].set_aspect("equal", adjustable="box")
        cbar3 = fig.colorbar(
            sc3, ax=axes[index, 3], orientation="horizontal", fraction=0.046, pad=0.1
        )
        cbar3.set_label("m", fontsize=fontsize)
        cbar3.ax.tick_params(labelsize=23)

        sc4 = axes[index, 4].scatter(
            x=deflectometry_points_original[:, 0],
            y=deflectometry_points_original[:, 1],
            c=angles_deflectometry_original,
            cmap="inferno",
            vmin=0.0,
            vmax=0.005,
        )
        axes[index, 4].set_title("Deflectometry Normals", fontsize=fontsize)
        axes[index, 4].axis("off")
        axes[index, 4].set_aspect("equal", adjustable="box")
        cbar4 = fig.colorbar(
            sc4, ax=axes[index, 4], orientation="horizontal", fraction=0.046, pad=0.1
        )
        cbar4.set_label("Angle (rad)", fontsize=fontsize)
        cbar4.ax.tick_params(labelsize=23)

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
        axes[index, 5].set_title("Reconstructed Points", fontsize=fontsize)
        axes[index, 5].axis("off")
        axes[index, 5].set_aspect("equal", adjustable="box")
        cbar5 = fig.colorbar(
            sc5, ax=axes[index, 5], orientation="horizontal", fraction=0.046, pad=0.1
        )
        cbar5.set_label("m", fontsize=fontsize)
        cbar5.ax.tick_params(labelsize=23)

        sc6 = axes[index, 6].scatter(
            x=reconstructed_points[:, 0],
            y=reconstructed_points[:, 1],
            c=angles_reconstructed,
            cmap="inferno",
            vmin=0.0,
            vmax=0.005,
        )
        axes[index, 6].set_title("Reconstructed Normals", fontsize=fontsize)
        axes[index, 6].axis("off")
        axes[index, 6].set_aspect("equal", adjustable="box")
        cbar6 = fig.colorbar(
            sc6, ax=axes[index, 6], orientation="horizontal", fraction=0.046, pad=0.1
        )
        cbar6.set_label("Angle (rad)", fontsize=fontsize)
        cbar6.ax.tick_params(labelsize=23)

    save_dir = save_dir / f"run_{results_number}"
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = save_dir / "surface_reconstruction_flux.png"
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved surface reconstruction flux plot at: {filename}.")


def plot_surface_error_analysis(
    results: dict[str, Any],
    save_dir: pathlib.Path,
    results_number: int,
) -> None:
    """
    Plot the surface reconstruction error analysis.

    Parameters
    ----------
    results : dict[str, Any]
        Results of the study.
    save_dir : pathlib.Path
        Path to the location where the plots are saved.
    results_number : int
        Identifier of the results run.
    """
    losses_list = results["surface_reconstruction"]["loss"]
    errors = torch.min(torch.stack(losses_list, dim=0), dim=0).values.cpu().numpy()

    positions = list(results["heliostat_positions"].values())
    distances = np.linalg.norm(np.array(positions)[:, :3], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax1, ax2 = axes

    x_max = np.max(errors)
    x_vals = np.linspace(0, x_max, 100)

    kde = gaussian_kde(errors, bw_method="scott")
    kde_values = kde(x_vals)
    mean = np.mean(errors)

    ax1.hist(
        errors,
        bins=25,
        range=(0, x_max),
        density=True,
        alpha=0.3,
        label="Histogram KL-Div",
        color=plot_colors["lightblue"],
    )

    ax1.plot(
        x_vals,
        kde_values,
        label="KDE",
        color=plot_colors["lightblue"],
    )

    ax1.axvline(
        mean,
        color=plot_colors["lightblue"],
        linestyle="--",
        label=f"Mean: {mean:.2f}",
    )

    ax1.set_xlabel(r"\textbf{KL-Divergence}")
    ax1.set_ylabel(r"\textbf{Density}")
    ax1.legend(fontsize=8)
    ax1.grid(True)

    ax2.scatter(
        distances,
        errors,
        color=plot_colors["lightblue"],
        marker="o",
        label="Reconstruction Error",
        alpha=0.7,
    )

    fit = np.poly1d(np.polyfit(distances, errors, 1))
    x_vals = np.linspace(distances.min(), distances.max(), 200)

    ax2.plot(
        x_vals,
        fit(x_vals),
        color=plot_colors["lightblue"],
        linestyle="--",
        label="Linear Fit",
    )

    ax2.set_xlabel(r"\textbf{Heliostat Distance from Tower [m]}")
    ax2.set_ylabel(r"\textbf{Mean Reconstruction Error}")
    ax2.grid(True)
    ax2.legend(loc="upper right")

    save_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        save_dir / f"run_{results_number}/surface_reconstruction_error_analysis.pdf"
    )
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved surface reconstruction error analysis plot at: {filename}.")


def plot_surface_loss_history(
    results: dict[str, Any],
    save_dir: pathlib.Path,
    results_number: int,
) -> None:
    """
    Plot the surface reconstruction loss history.

    Parameters
    ----------
    results : dict[str, Any]
        Results of the study.
    save_dir : pathlib.Path
        Path to the location where the plots are saved.
    results_number : int
        Identifier of the results run.
    """
    loss_history = results["surface_reconstruction"]["loss_history"]
    epochs = np.arange(0, len(loss_history[0][0][0]["total_loss"]))

    for batch_index, batch in enumerate(loss_history):
        fig, ax1 = plt.subplots(figsize=(8, 5))

        (l1,) = ax1.plot(
            epochs,
            batch[0][0]["total_loss"],
            label=r"Total Loss",
            color=plot_colors["darkblue"],
        )

        (l2,) = ax1.plot(
            epochs,
            batch[0][0]["flux_loss"],
            label=r"KL-Divergence",
            color=plot_colors["blue_1"],
        )

        (l3,) = ax1.plot(
            epochs,
            batch[0][0]["ideal_regularizer"],
            label=r"Ideal Surface Regularization",
            color=plot_colors["blue_2"],
        )

        (l4,) = ax1.plot(
            epochs,
            batch[0][0]["smoothness_regularizer"],
            label=r"Smooth Surface Regularization",
            color=plot_colors["blue_3"],
        )

        (l5,) = ax1.plot(
            epochs,
            batch[0][0]["flux_integral_constraint"],
            label=r"Flux integral Constraint",
            color=plot_colors["blue_4"],
        )

        ax1.set_xlabel(r"Epoch")
        ax1.set_ylabel(r"Loss Terms")
        ax1.grid(True)

        ax2 = ax1.twinx()
        (l6,) = ax2.plot(
            epochs,
            batch[0][0]["flux_integral"],
            label=r"Flux integral difference %",
            color=plot_colors["darkred"],
        )

        ax2.set_ylabel(r"Flux integral \%")
        lines = [l1, l2, l3, l4, l5, l6]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="upper right")

        ax1.set_title(r"\textbf{Loss History}", fontsize=13, ha="center")

        save_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            save_dir
            / f"run_{results_number}/surface_reconstruction_loss_history_{batch_index}.pdf"
        )
        fig.tight_layout()
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved surface reconstruction loss history plot at: {filename}.")


def plot_surface_validation(
    results: dict[str, Any],
    save_dir: pathlib.Path,
    results_number: int,
) -> None:
    """
    Plot the surface reconstruction validation.

    Parameters
    ----------
    results : dict[str, Any]
        Results of the study.
    save_dir : pathlib.Path
        Path to the location where the plots are saved.
    results_number : int
        Identifier of the results run.
    """
    plot_data = results["surface_reconstruction"]["validation_data"]

    kl_divergence = KLDivergenceLoss()
    kl_losses = []
    mse_losses = []
    l1_norm_losses = []

    keys = list(plot_data.keys())

    for k in keys:
        measured_flux = torch.nn.functional.normalize(
            plot_data[k]["measured_flux"],
            p=1,
            dim=(0, 1),
            eps=1e-12,
        )
        artist_flux = torch.nn.functional.normalize(
            plot_data[k]["artist_flux"],
            p=1,
            dim=(0, 1),
            eps=1e-12,
        )

        kl = kl_divergence(
            measured_flux.unsqueeze(0),
            artist_flux.unsqueeze(0),
            reduction_dimensions=(1, 2),
        )
        kl_losses.append(float(kl))

        # MSE
        mse = torch.nn.functional.mse_loss(measured_flux, artist_flux).item()
        mse_losses.append(mse)

        # normalized L1-like
        l1_norm = (
            torch.abs(artist_flux - measured_flux).sum() / measured_flux.sum()
        ).item()
        l1_norm_losses.append(l1_norm)

    x = list(range(len(keys)))

    plt.figure(figsize=(10, 5))
    plt.scatter(x, kl_losses, label="KL Divergence")
    plt.scatter(x, mse_losses, label="MSE")
    plt.scatter(x, l1_norm_losses, label="L1 normalized")
    plt.title("Loss comparison per heliostat")
    plt.xlabel("Heliostat index")
    plt.ylabel("Loss value")
    plt.legend()
    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        save_dir / f"run_{results_number}/surface_reconstruction_validation_loss.pdf"
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.clf()

    print(f"Saved surface reconstruction validation loss at: {filename}.")

    selected_indices = list(range(0, len(keys), 1))
    n_rows = len(selected_indices)

    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(8, 3 * n_rows),
    )

    if n_rows == 1:
        axes = [axes]

    for row, idx in enumerate(selected_indices):
        k = keys[idx]

        measured_flux = plot_data[k]["measured_flux"].detach().cpu()
        artist_flux = plot_data[k]["artist_flux"].detach().cpu()

        ax1 = axes[row][0]
        ax2 = axes[row][1]

        im1 = ax1.imshow(measured_flux)
        ax1.set_title(f"Measured {k}")
        ax1.axis("off")

        im2 = ax2.imshow(artist_flux)
        ax2.set_title(f"Artist {k}")
        ax2.axis("off")

    save_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        save_dir / f"run_{results_number}/surface_reconstruction_validation_fluxes.pdf"
    )
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved surface reconstruction validation fluxes at: {filename}.")


def plot_kinematics_reconstruction_flux(
    results: dict[str, Any],
    save_dir: pathlib.Path,
    results_number: int,
) -> None:
    """
    Plot the kinematic reconstruction flux results.

    Parameters
    ----------
    results : dict[str, Any]
        Results of the study.
    save_dir : pathlib.Path
        Path to the location where the plots are saved.
    results_number : int
        Identifier of the results run.
    """
    reconstructions = [
        "kinematics_reconstruction_with_ideal_surfaces",
        "kinematics_reconstruction_with_reconstructed_surfaces",
    ]

    col_labels = [
        "Calibration Flux",
        "Default\\\\Kinematics",
        "Reconstructed\\\\Kinematics",
    ]

    save_dir = save_dir / f"run_{results_number}"
    save_dir.mkdir(parents=True, exist_ok=True)

    for reconstruction in reconstructions:
        heliostat_dict = results[reconstruction]["flux_plot_data"]
        heliostat_names = list(heliostat_dict.keys())
        n_rows = len(heliostat_names)
        n_cols = 3

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * 2, n_rows * 2),
        )

        if n_rows == 1:
            axes = axes[np.newaxis, :]
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        for i in range(n_rows):
            for j in range(n_cols):
                axes[i, j].axis("off")

        for col_index, label in enumerate(col_labels):
            axes[0, col_index].set_title(
                rf"\textbf{{{label}}}",
                fontsize=13,
                ha="center",
            )

        for row_index, heliostat_name in enumerate(heliostat_names):
            flux_data = heliostat_dict[heliostat_name]["fluxes"].detach().cpu()
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

        filename = save_dir / f"{reconstruction}_fluxes.pdf"
        fig.tight_layout()
        fig.savefig(filename, dpi=300)
        plt.close(fig)

        print(f"Saved kinematics reconstruction flux plot at: {filename}.")


def plot_kinematics_error_analysis(
    results: dict[str, Any],
    save_dir: pathlib.Path,
    split_left: bool = True,
    results_number: int = 1,
) -> None:
    """
    Plot kinematic reconstruction error analysis.

    Parameters
    ----------
    results : dict[str, Any]
        Results of the study.
    save_dir : pathlib.Path
        Path to save the plots.
    split_left : bool, default=True
        If True, left column contains two stacked KDE plots (meters + mrad).
        If False, left column contains only meters KDE.
    results_number : int
        Identifier of the results run.
    """
    reconstructions = [
        "kinematics_reconstruction_with_ideal_surfaces",
        "kinematics_reconstruction_with_reconstructed_surfaces",
    ]
    case_labels = ["Ideal Surfaces", "Reconstructed Surfaces"]

    positions = list(results["heliostat_positions"].values())
    distances = np.linalg.norm(np.array(positions)[:, :3], axis=1)

    save_dir = save_dir / f"run_{results_number}"
    save_dir.mkdir(parents=True, exist_ok=True)

    for reconstruction, label in zip(reconstructions, case_labels):
        errors_m = np.array(results[reconstruction]["loss"].detach().cpu())
        errors_mrad = (errors_m / distances) * 1000
        if split_left:
            fig = plt.figure(figsize=(10, 6))
            gs = GridSpec(
                2,
                2,
                figure=fig,
                width_ratios=[1, 1.3],
                height_ratios=[0.5, 0.5],
                wspace=0.3,
                hspace=0.25,
            )
            ax_kde_m = fig.add_subplot(gs[0, 0])
            ax_kde_mrad = fig.add_subplot(gs[1, 0])
        else:
            fig = plt.figure(figsize=(10, 4))
            gs = GridSpec(
                1,
                2,
                figure=fig,
                width_ratios=[1, 1.3],
                wspace=0.3,
            )
            ax_kde_m = fig.add_subplot(gs[0, 0])
            ax_kde_mrad = None

        ax_dist_m = (
            fig.add_subplot(gs[:, 1]) if split_left else fig.add_subplot(gs[0, 1])
        )
        ax_dist_mrad = ax_dist_m.twinx()

        x_max = max(errors_m)
        x_vals = np.linspace(0, x_max, 100)
        kde = gaussian_kde(errors_m, bw_method="scott")
        kde_vals = kde(x_vals)

        ax_kde_m.hist(
            errors_m,
            bins=25,
            range=(0, x_max),
            density=True,
            alpha=0.3,
            color=plot_colors["lightblue"],
            label="Histogram",
        )
        ax_kde_m.plot(
            x_vals,
            kde_vals,
            color=plot_colors["lightblue"],
            label="KDE",
        )
        ax_kde_m.axvline(
            np.mean(errors_m),
            linestyle="--",
            color=plot_colors["lightblue"],
            label=f"Mean: {np.mean(errors_m):.3f} m",
        )
        ax_kde_m.set_xlabel(r"\textbf{Pointing Error [m]}")
        ax_kde_m.set_ylabel(r"\textbf{Density}")
        ax_kde_m.grid(True)
        ax_kde_m.legend(fontsize=8)

        if split_left:
            x_max = max(errors_mrad)
            x_vals = np.linspace(0, x_max, 100)
            kde = gaussian_kde(errors_mrad, bw_method="scott")
            kde_vals = kde(x_vals)

            ax_kde_mrad.hist(
                errors_mrad,
                bins=25,
                range=(0, x_max),
                density=True,
                alpha=0.3,
                color=plot_colors["darkblue"],
                label="Histogram",
            )
            ax_kde_mrad.plot(
                x_vals,
                kde_vals,
                color=plot_colors["darkblue"],
                label="KDE",
            )
            ax_kde_mrad.axvline(
                np.mean(errors_mrad),
                linestyle="--",
                color=plot_colors["darkblue"],
                label=f"Mean: {np.mean(errors_mrad):.3f} mrad",
            )
            ax_kde_mrad.set_xlabel(r"\textbf{Pointing Error [mrad]}")
            ax_kde_mrad.set_ylabel(r"\textbf{Density}")
            ax_kde_mrad.grid(True)
            ax_kde_mrad.legend(fontsize=8)

        ax_dist_m.scatter(
            distances,
            errors_m,
            marker="o",
            color=plot_colors["lightblue"],
            alpha=0.7,
            label="Error (m)",
        )
        ax_dist_mrad.scatter(
            distances,
            errors_mrad,
            marker="^",
            color=plot_colors["darkblue"],
            alpha=0.7,
            label="Error (mrad)",
        )

        x_fit = np.linspace(distances.min(), distances.max(), 200)
        fit_m = np.poly1d(np.polyfit(distances, errors_m, 1))
        fit_mrad = np.poly1d(np.polyfit(distances, errors_mrad, 1))

        ax_dist_m.plot(
            x_fit, fit_m(x_fit), linestyle="--", color=plot_colors["lightblue"]
        )
        ax_dist_mrad.plot(
            x_fit, fit_mrad(x_fit), linestyle="--", color=plot_colors["darkblue"]
        )

        ax_dist_m.set_xlabel(r"\textbf{Heliostat Distance from Tower [m]}")
        ax_dist_m.set_ylabel(r"\textbf{Pointing Error [m]}")
        ax_dist_mrad.set_ylabel(r"\textbf{Pointing Error [mrad]}")
        ax_dist_m.grid(True)

        handles_m, labels_m = ax_dist_m.get_legend_handles_labels()
        handles_a, labels_a = ax_dist_mrad.get_legend_handles_labels()
        ax_dist_m.legend(
            handles_m + handles_a, labels_m + labels_a, fontsize=8, loc="upper right"
        )

        fig.suptitle(rf"\textbf{{Kinematics Reconstruction with {label}}}", fontsize=14)

        filename = save_dir / f"{reconstruction}_error_analysis.pdf"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved kinematics reconstruction error analysis plot at: {filename}.")


def plot_kinematics_loss_history(
    results: dict[str, Any],
    save_dir: pathlib.Path,
    results_number: int,
) -> None:
    """
    Plot the kinematic reconstruction loss history.

    Parameters
    ----------
    results : dict[str, Any]
        Results of the study.
    save_dir : pathlib.Path
        Path to the location where the plots are saved.
    results_number : int
        Identifier of the results run.
    """
    reconstructions = [
        "kinematics_reconstruction_with_ideal_surfaces",
        "kinematics_reconstruction_with_reconstructed_surfaces",
    ]

    for reconstruction in reconstructions:
        loss_history = results[reconstruction]["loss_history"]
        epochs = np.arange(0, len(loss_history[0][0]["total_loss"]))

        fig, ax1 = plt.subplots(figsize=(6, 5))

        ax1.plot(
            epochs,
            loss_history[0][0]["total_loss"],
            label=r"Total Loss",
            color=plot_colors["darkblue"],
        )

        ax1.set_xlabel(r"Epoch")
        ax1.set_ylabel(r"Loss Terms")
        ax1.grid(True)
        ax1.legend(loc="upper right")

        ax1.set_title(r"\textbf{Loss History}", fontsize=13, ha="center")

        save_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            save_dir
            / f"run_{results_number}/kinematics_reconstruction_loss_history_{reconstruction}.pdf"
        )
        plt.tight_layout()
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved kinematics reconstructions loss history plot at: {filename}.")


def plot_kinematics_validation(
    results: dict[str, Any],
    save_dir: pathlib.Path,
    results_number: int,
) -> None:
    """
    Plot the surface reconstruction validation.

    Parameters
    ----------
    results : dict[str, Any]
        Results of the study.
    save_dir : pathlib.Path
        Path to the location where the plots are saved.
    results_number : int
        Identifier of the results run.
    """
    plot_data = results["kinematics_reconstruction_with_reconstructed_surfaces"][
        "validation_data"
    ]

    keys = list(plot_data.keys())

    # for k in keys:
    #     measured_flux = plot_data[k]["measured_flux"]
    #     artist_flux = plot_data[k]["artist_flux"]

    #     centers = utils.get_center_of_mass(
    #         bitmaps=torch.stack([measured_flux, artist_flux]),
    #     )

    # x = list(range(len(keys)))
    # plt.figure(figsize=(10, 5))
    # plt.scatter(x, kl_losses, label="KL Divergence")
    # plt.scatter(x, mse_losses, label="MSE")
    # plt.scatter(x, l1_norm_losses, label="L1 normalized")
    # plt.title("Loss comparison per heliostat")
    # plt.xlabel("Heliostat index")
    # plt.ylabel("Loss value")
    # plt.legend()
    # plt.tight_layout()

    # save_dir.mkdir(parents=True, exist_ok=True)
    # filename = (
    #     save_dir
    #     / f"run_{results_number}/kinematics_reconstruction_validation_loss.pdf"
    # )
    # plt.tight_layout()
    # plt.savefig(filename, dpi=300, bbox_inches="tight")
    # plt.clf()

    # print(f"Saved kinematics reconstruction validation loss at: {filename}.")

    selected_indices = list(range(0, len(keys), 1))
    n_rows = len(selected_indices)

    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(8, 3 * n_rows),
    )

    if n_rows == 1:
        axes = [axes]

    for row, idx in enumerate(selected_indices):
        k = keys[idx]

        measured_flux = plot_data[k]["measured_flux"]
        artist_flux = plot_data[k]["artist_flux"]

        centers = (
            utils.get_center_of_mass(
                bitmaps=torch.stack([measured_flux, artist_flux]),
            )
            .detach()
            .cpu()
        )

        ax1 = axes[row][0]
        ax2 = axes[row][1]

        im1 = ax1.imshow(measured_flux.detach().cpu())
        ax1.set_title(f"Measured {k}")
        ax1.scatter(x=centers[0, 0], y=centers[0, 1], c="black", marker="o", s=30)
        ax1.scatter(x=centers[1, 0], y=centers[1, 1], c="red", marker="x", s=30)
        ax1.axis("off")

        im2 = ax2.imshow(artist_flux.detach().cpu())
        ax2.set_title(f"Artist {k}")
        ax2.scatter(x=centers[0, 0], y=centers[0, 1], c="black", marker="o", s=30)
        ax2.scatter(x=centers[1, 0], y=centers[1, 1], c="red", marker="x", s=30)
        ax2.axis("off")

    save_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        save_dir
        / f"run_{results_number}/kinematics_reconstruction_validation_fluxes.pdf"
    )
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved kinematics reconstruction validation fluxes at: {filename}.")


def plot_model_reconstruction(
    results: dict[str, Any],
    save_dir: pathlib.Path,
    results_number: int,
) -> None:
    """
    Plot the aim point optimization flux results.

    Parameters
    ----------
    results : dict[str, Any]
        Results of the study.
    save_dir : pathlib.Path
        Path to the location where the plots are saved.
    results_number : int
        Identifier of the results run.
    """
    measured_flux = results["measured_flux"]
    ideal_model_flux = results["ideal_model"]["aim_point_plot"]
    reconstructed_surfaces_flux = results["surface_reconstruction"]["aim_point_plot"]
    reconstructed_kinematics_flux = results[
        "kinematics_reconstruction_with_ideal_surfaces"
    ]["aim_point_plot"]
    combined_reconstruction_flux = results[
        "kinematics_reconstruction_with_reconstructed_surfaces"
    ]["aim_point_plot"]

    measured_flux_normed = measured_flux * (
        combined_reconstruction_flux.sum() / measured_flux.sum()
    )

    kl_divergence = KLDivergenceLoss()
    kl_divs_1 = [
        "",
        kl_divergence(
            measured_flux_normed.unsqueeze(0),
            ideal_model_flux.unsqueeze(0),
            reduction_dimensions=(1, 2),
        ).item(),
        kl_divergence(
            measured_flux_normed.unsqueeze(0),
            reconstructed_surfaces_flux.unsqueeze(0),
            reduction_dimensions=(1, 2),
        ).item(),
        kl_divergence(
            measured_flux_normed.unsqueeze(0),
            reconstructed_kinematics_flux.unsqueeze(0),
            reduction_dimensions=(1, 2),
        ).item(),
        kl_divergence(
            measured_flux_normed.unsqueeze(0),
            combined_reconstruction_flux.unsqueeze(0),
            reduction_dimensions=(1, 2),
        ).item(),
    ]

    kl_divs_2 = [
        "",
        kl_divergence(
            ideal_model_flux.unsqueeze(0),
            measured_flux_normed.unsqueeze(0),
            reduction_dimensions=(1, 2),
        ).item(),
        kl_divergence(
            reconstructed_surfaces_flux.unsqueeze(0),
            measured_flux_normed.unsqueeze(0),
            reduction_dimensions=(1, 2),
        ).item(),
        kl_divergence(
            reconstructed_kinematics_flux.unsqueeze(0),
            measured_flux_normed.unsqueeze(0),
            reduction_dimensions=(1, 2),
        ).item(),
        kl_divergence(
            combined_reconstruction_flux.unsqueeze(0),
            measured_flux_normed.unsqueeze(0),
            reduction_dimensions=(1, 2),
        ).item(),
    ]

    mse_losses = [
        torch.nan,
        torch.nn.functional.mse_loss(ideal_model_flux, measured_flux_normed).item(),
        torch.nn.functional.mse_loss(
            reconstructed_surfaces_flux, measured_flux_normed
        ).item(),
        torch.nn.functional.mse_loss(
            reconstructed_kinematics_flux, measured_flux_normed
        ).item(),
        torch.nn.functional.mse_loss(
            combined_reconstruction_flux, measured_flux_normed
        ).item(),
    ]

    l_one_modified = [
        torch.nan,
        (
            torch.abs((ideal_model_flux - measured_flux_normed)).sum()
            / measured_flux_normed.sum()
        ).item(),
        (
            torch.abs((reconstructed_surfaces_flux - measured_flux_normed)).sum()
            / measured_flux_normed.sum()
        ).item(),
        (
            torch.abs((reconstructed_kinematics_flux - measured_flux_normed)).sum()
            / measured_flux_normed.sum()
        ).item(),
        (
            torch.abs((combined_reconstruction_flux - measured_flux_normed)).sum()
            / measured_flux_normed.sum()
        ).item(),
    ]

    images = [
        measured_flux_normed.cpu().detach(),
        ideal_model_flux.cpu().detach(),
        reconstructed_surfaces_flux.cpu().detach(),
        reconstructed_kinematics_flux.cpu().detach(),
        combined_reconstruction_flux.cpu().detach(),
    ]
    vmin = min(img.min() for img in images)
    vmax = max(img.max() for img in images)

    n_images = len(images)
    fig, axes = plt.subplots(
        1,
        n_images,
        figsize=(4 * n_images, 8),
        gridspec_kw={"bottom": 0.15, "top": 0.95, "wspace": 0.05},
    )
    if n_images == 1:
        axes = [axes]

    titles = [
        r"\textbf{Measured Flux}",
        r"\textbf{Ideal Model Flux}",
        r"\textbf{Reconstructed Surfaces}",
        r"\textbf{Reconstructed Kinematics}",
        r"\textbf{Combined Reconstructions}",
    ]

    # height, width = measured_flux.shape
    # points = {
    #     "focal spot aim point": 256
    #     - torch.tensor([104.7413, 139.8122], device="cuda:0"),
    #     "aim point 'measured'": 256
    #     - torch.tensor([112.9362, 111.4531], device="cuda:0"),
    # }

    # measured = images[0]
    # y_grid, x_grid = torch.meshgrid(torch.arange(256), torch.arange(256), indexing="ij")
    # x_com_measured = (x_grid * measured).sum() / measured.sum()
    # y_com_measured = (y_grid * measured).sum() / measured.sum()

    for idx, (ax, flux) in enumerate(zip(axes, images)):
        im = ax.imshow(flux, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis("off")
        ax.set_title(titles[idx], fontsize=13)
        # ax.scatter(
        #     x_com_measured,
        #     y_com_measured,
        #     s=80,
        #     marker="x",
        #     color="blue",
        #     label="center of mass reference",
        # )

        # y_grid, x_grid = torch.meshgrid(
        #     torch.arange(256), torch.arange(256), indexing="ij"
        # )
        # x_com = (x_grid * flux).sum() / flux.sum()
        # y_com = (y_grid * flux).sum() / flux.sum()
        # ax.scatter(
        #     x_com, y_com, s=80, marker="o", color="green", label="center of mass"
        # )

        # ax.scatter(
        #     256 / 2,
        #     256 / 2,
        #     s=40,
        #     marker="x",
        #     color="white",
        #     label="geometric center of bitmap",
        # )
        # ax.scatter(
        #     points["aim point 'measured'"][0].cpu(),
        #     points["aim point 'measured'"][1].cpu(),
        #     s=80,
        #     marker="x",
        #     color="white",
        #     label="aim point from protocol",
        # )

        ax.legend(fontsize=9, loc="upper right")

        annotation = f"Flux integral: {flux.sum():.2f}"
        if kl_divs_1[idx] != "":
            annotation += f"\nKL(M...): {kl_divs_1[idx]:.4f}"
        if kl_divs_2[idx] != "":
            annotation += f"\nKL(...M): {kl_divs_2[idx]:.4f}"
        if not torch.isnan(torch.tensor(mse_losses[idx])):
            annotation += f"\nMSE: {mse_losses[idx]:.4f}"
        if not torch.isnan(torch.tensor(l_one_modified[idx])):
            annotation += f"\nL1-mod: {l_one_modified[idx]:.4f}"

        ax.annotate(
            annotation,
            xy=(0.5, -0.05),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=12,
        )

    fig.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=14)

    fig.text(
        0.95,
        0.01,
        "M = Measured flux\nU = Unoptimized flux\nO = Optimized flux\nH = Homogeneous flux",
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=3),
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    filename = save_dir / f"run_{results_number}/model_reconstructions_flux.pdf"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved model reconstruction flux plot at: {filename}.")


def plot_aim_point_flux(
    results: dict[str, Any],
    save_dir: pathlib.Path,
    results_number: int,
) -> None:
    """
    Plot the aim point optimization flux results.

    Parameters
    ----------
    results : dict[str, Any]
        Results of the study.
    save_dir : pathlib.Path
        Path to the location where the plots are saved.
    results_number : int
        Identifier of the results run.
    """
    measured_flux = results["measured_flux"]
    homogeneous_distribution = results["homogeneous_distribution"]
    unoptimized_flux = results["aim_point_optimization_reconstructed_model"][
        "aim_point_plot"
    ][0]
    optimized_flux = results["aim_point_optimization_reconstructed_model"][
        "aim_point_plot"
    ][1]

    # Normalize measured and homogeneous fluxes
    measured_flux_normed = measured_flux * (
        unoptimized_flux.sum() / measured_flux.sum()
    )
    homogeneous_distribution_normed = homogeneous_distribution * (
        optimized_flux.sum() / homogeneous_distribution.sum()
    )

    # Compute KL divergences
    kl_divergence = KLDivergenceLoss()
    kl_div_measured_homogeneous = kl_divergence(
        homogeneous_distribution_normed.unsqueeze(0),
        measured_flux_normed.unsqueeze(0),
        reduction_dimensions=(1, 2),
    )
    kl_div_unoptimized_homogeneous = kl_divergence(
        homogeneous_distribution.unsqueeze(0),
        unoptimized_flux.unsqueeze(0),
        reduction_dimensions=(1, 2),
    )
    kl_div_optimized_homogeneous = kl_divergence(
        homogeneous_distribution.unsqueeze(0),
        optimized_flux.unsqueeze(0),
        reduction_dimensions=(1, 2),
    )
    improvement = (100 / unoptimized_flux.sum()) * optimized_flux.sum()

    images = [
        measured_flux_normed.cpu().detach(),
        unoptimized_flux.cpu().detach(),
        optimized_flux.cpu().detach(),
        homogeneous_distribution_normed.cpu().detach(),
    ]
    vmin = min(img.min() for img in images)
    vmax = max(img.max() for img in images)

    n_images = len(images)
    fig, axes = plt.subplots(
        1,
        n_images,
        figsize=(4 * n_images, 5),
        gridspec_kw={"bottom": 0.15, "top": 0.95, "wspace": 0.05},
    )
    if n_images == 1:
        axes = [axes]

    kl_divs = [
        rf"$\mathrm{{KL}}(\mathrm{{H}} \,\|\, \mathrm{{M}}) = {kl_div_measured_homogeneous.item():.4f}$",
        rf"$\mathrm{{KL}}(\mathrm{{H}} \,\|\, \mathrm{{U}}) = {kl_div_unoptimized_homogeneous.item():.4f}$",
        rf"$\mathrm{{KL}}(\mathrm{{H}} \,\|\, \mathrm{{O}}) = {kl_div_optimized_homogeneous.item():.4f}$",
        "",
    ]
    improvement_labels = [
        "",
        r"$\mathrm{100\%}$",
        rf"$\mathrm{{{improvement:.2f}\%}}$",
        "",
    ]
    titles = [
        r"\textbf{Baseline Aim Point}",
        r"\textbf{Reconstructed Model}",
        r"\textbf{Optimized Aim Points}",
        r"\textbf{Homogeneous Distribution}",
    ]

    for idx, (ax, flux) in enumerate(zip(axes, images)):
        im = ax.imshow(flux, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis("off")
        ax.set_title(titles[idx], fontsize=13)
        ax.annotate(
            f"Flux integral: {flux.sum():.2f}\n{kl_divs[idx]}\n{improvement_labels[idx]}",
            xy=(0.5, -0.05),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=12,
        )

    cbar_ax = fig.add_axes((0.15, 0.01, 0.7, 0.03))
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=14)

    fig.text(
        0.95,
        0.02,
        "M = Measured flux\nU = Unoptimized flux\nO = Optimized flux\nH = Homogeneous flux",
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=3),
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    filename = save_dir / f"run_{results_number}/aim_point_optimization_flux.pdf"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved aim point optimization flux plot at: {filename}.")


def plot_aim_point_loss_history(
    results: dict[str, Any],
    save_dir: pathlib.Path,
    results_number: int,
) -> None:
    """
    Plot the aim point optimization loss history.

    Parameters
    ----------
    results : dict[str, Any]
        Results of the study.
    save_dir : pathlib.Path
        Path to the location where the plots are saved.
    results_number : int
        Identifier of the results run.
    """
    loss_history = results["aim_point_optimization_reconstructed_model"]["loss_history"]
    epochs = np.arange(0, len(loss_history["total_loss"]))

    fig, ax1 = plt.subplots(figsize=(8, 5))

    (l1,) = ax1.plot(
        epochs,
        loss_history["total_loss"],
        label=r"Total Loss",
        color=plot_colors["darkblue"],
    )

    (l2,) = ax1.plot(
        epochs,
        loss_history["flux_loss"],
        label=r"KL-Divergence",
        color=plot_colors["blue_1"],
    )

    (l3,) = ax1.plot(
        epochs,
        loss_history["local_flux_constraint"],
        label=r"Local Flux Constraint",
        color=plot_colors["blue_2"],
    )

    (l4,) = ax1.plot(
        epochs,
        loss_history["intercept_constraint"],
        label=r"Intercept Constraint",
        color=plot_colors["blue_3"],
    )
    (l5,) = ax1.plot(
        epochs,
        loss_history["flux_integral_constraint"],
        label=r"Flux Integral Constraint",
        color=plot_colors["blue_4"],
    )

    ax1.set_xlabel(r"Epoch")
    ax1.set_ylabel(r"Loss Terms")
    ax1.grid(True)

    ax2 = ax1.twinx()
    (l6,) = ax2.plot(
        epochs,
        loss_history["flux_integral"],
        label=r"Flux integral %",
        color=plot_colors["darkred"],
    )

    ax2.set_ylabel(r"Flux integral \%")
    lines = [l1, l2, l3, l4, l5, l6]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")

    ax1.set_title(r"\textbf{Loss History}", fontsize=13, ha="center")

    save_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        save_dir / f"run_{results_number}/aim_point_optimization_loss_history.pdf"
    )
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved aim point optimization loss history plot at: {filename}.")


if __name__ == "__main__":
    """
    Generate plots based on the field optimization study.

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
    config: dict[str, Any] = {}
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            warnings.warn(f"Error parsing YAML file: {exc}.")
    else:
        warnings.warn(f"Configuration file not found at {config_path}. Using defaults.")

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
    args = parser.parse_args()

    device = get_device(torch.device(args.device))

    for case in ["baseline", "full_field"]:
        results_number = 5
        results_path = (
            pathlib.Path(args.results_dir) / case / f"results_{results_number}.pt"
        )

        plots_path = pathlib.Path(args.plots_dir) / case
        (plots_path / f"run_{results_number}").mkdir(parents=True, exist_ok=True)

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

        # results_ftp = torch.load(
        #     pathlib.Path(args.results_dir) / "ftp/full_field" / "results_0.pt",
        #     weights_only=False,
        #     map_location=device,
        # )

        # plot_heliostat_positions(
        #     results=results, results_ftp=results_ftp, save_dir=plots_path
        # )

        # plot_kinematics_reconstruction_flux(
        #     results=results, save_dir=plots_path, results_number=results_number
        # )
        # plot_kinematics_error_analysis(
        #     results=results,
        #     save_dir=plots_path,
        #     split_left=False,
        #     results_number=results_number,
        # )
        # plot_kinematics_loss_history(
        #     results=results, save_dir=plots_path, results_number=results_number
        # )
        # plot_kinematics_validation(
        #     results=results, save_dir=plots_path, results_number=results_number
        # )

        plot_surface_reconstruction_flux(
            results=results, save_dir=plots_path, results_number=results_number
        )
        plot_surface_error_analysis(
            results=results, save_dir=plots_path, results_number=results_number
        )
        plot_surface_loss_history(
            results=results, save_dir=plots_path, results_number=results_number
        )
        plot_surface_validation(
            results=results, save_dir=plots_path, results_number=results_number
        )

        plot_model_reconstruction(
            results=results, save_dir=plots_path, results_number=results_number
        )

        plot_aim_point_flux(
            results=results, save_dir=plots_path, results_number=results_number
        )
        plot_aim_point_loss_history(
            results=results, save_dir=plots_path, results_number=results_number
        )
