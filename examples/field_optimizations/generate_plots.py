import argparse
import pathlib
import warnings
from typing import Any

import h5py
import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

from artist.flux import bitmap
from artist.geometry import coordinates
from artist.optim.loss import KLDivergenceLoss
from artist.scenario.scenario import Scenario
from artist.util.env import get_device

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

plt.rcParams["text.usetex"] = False
plt.rcParams["text.latex.preamble"] = r"""
\usepackage{cmbright}
\setlength{\parindent}{0pt}
"""
cmap = "inferno"


def plot_heliostat_positions(
    scenario_dir: pathlib.Path, save_dir: pathlib.Path
) -> None:
    """
    Plot heliostat positions.

    Parameters
    ----------
    scenario_dir : pathlib.Path
        Path to the scenarios.
    save_dir : pathlib.Path
        Directory to save the plots.
    """
    scenario_path_full_field = scenario_dir / "ideal_full_field_scenario.h5"
    scenario_path_baseline = scenario_dir / "ideal_baseline_scenario.h5"

    with h5py.File(scenario_path_full_field) as scenario_file:
        scenario_full_field = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file,
            device=device,
        )
    positions_full_field = scenario_full_field.heliostat_field.heliostat_groups[
        0
    ].positions

    with h5py.File(scenario_path_baseline) as scenario_file:
        scenario_baseline = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file,
            device=device,
        )
    positions_baseline = scenario_baseline.heliostat_field.heliostat_groups[0].positions

    x = [pos[0].cpu() for pos in positions_baseline]
    y = [pos[1].cpu() for pos in positions_baseline]

    x_all = [pos[0].cpu() for pos in positions_full_field]
    y_all = [pos[1].cpu() for pos in positions_full_field]

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
    filename = save_dir / "heliostat_positions.png"
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
    col_labels = [
        "Calibration Flux",
        "Ideal\\\\Surfaces",
        "Reconstructed\\\\Surfaces",
    ]
    save_dir = save_dir / f"run_{results_number}"
    save_dir.mkdir(parents=True, exist_ok=True)

    flux_data_before = results["surface_reconstruction"]["flux_plot_data_before"]
    flux_data_after = results["surface_reconstruction"]["flux_plot_data_after"]

    testing_loss_pixel_1 = results["surface_reconstruction"]["loss_history"][0][0][0][
        "test_loss"
    ]["pixel_loss"]
    testing_loss_pixel_2 = results["surface_reconstruction"]["loss_history"][1][0][0][
        "test_loss"
    ]["pixel_loss"]
    testing_loss_pixel_3 = results["surface_reconstruction"]["loss_history"][2][0][0][
        "test_loss"
    ]["pixel_loss"]
    testing_errors_pixel = (
        torch.cat(
            [testing_loss_pixel_1, testing_loss_pixel_2, testing_loss_pixel_3], dim=0
        )
        .cpu()
        .detach()
        .numpy()
    )

    testing_loss_kl_div_1 = results["surface_reconstruction"]["loss_history"][0][0][0][
        "test_loss"
    ]["kl_div"]
    testing_loss_kl_div_2 = results["surface_reconstruction"]["loss_history"][1][0][0][
        "test_loss"
    ]["kl_div"]
    testing_loss_kl_div_3 = results["surface_reconstruction"]["loss_history"][2][0][0][
        "test_loss"
    ]["kl_div"]
    testing_errors_kl_div = (
        torch.cat(
            [testing_loss_kl_div_1, testing_loss_kl_div_2, testing_loss_kl_div_3], dim=0
        )
        .cpu()
        .detach()
        .numpy()
    )

    heliostat_names = list(flux_data_before.keys())

    for heliostat_idx, heliostat_name in enumerate(heliostat_names):
        if heliostat_idx == 10:
            break
        pixel_loss = testing_errors_pixel[heliostat_idx].item()
        kl_div = testing_errors_kl_div[heliostat_idx].item()
        measured_flux = flux_data_before[heliostat_name]["measured_flux"].detach().cpu()

        artist_flux_before = (
            flux_data_before[heliostat_name]["artist_flux"].detach().cpu()
        )

        artist_flux_after = (
            flux_data_after[heliostat_name]["artist_flux"].detach().cpu()
        )

        n_samples = min(2, measured_flux.shape[0])

        measured_flux = measured_flux[:n_samples]
        artist_flux_before = artist_flux_before[:n_samples]
        artist_flux_after = artist_flux_after[:n_samples]

        fig, axes = plt.subplots(
            n_samples,
            3,
            figsize=(9, 3 * n_samples),
        )

        if n_samples == 1:
            axes = axes[np.newaxis, :]

        for col_idx, label in enumerate(col_labels):
            axes[0, col_idx].set_title(
                rf"\textbf{{{label}}}",
                fontsize=13,
            )

        for sample_idx in range(n_samples):
            axes[sample_idx, 0].imshow(
                measured_flux[sample_idx],
                cmap=cmap,
            )
            axes[sample_idx, 1].imshow(
                artist_flux_before[sample_idx],
                cmap=cmap,
            )
            axes[sample_idx, 2].imshow(
                artist_flux_after[sample_idx],
                cmap=cmap,
            )
            axes[sample_idx, 0].set_ylabel(
                rf"\textbf{{Sample {sample_idx + 1}}}",
                rotation=90,
                fontsize=12,
                labelpad=15,
            )
            for col_idx in range(3):
                axes[sample_idx, col_idx].set_xticks([])
                axes[sample_idx, col_idx].set_yticks([])

        position = results["heliostat_positions"][heliostat_name]
        position_str = ", ".join(f"{coord:.2f}" for coord in position[:3])

        fig.suptitle(
            rf"\textbf{{Heliostat {heliostat_name}}}"
            + "\n"
            + rf"\textit{{ENU Position: {position_str}}}"
            + "\n"
            + rf"Test Loss Pixel: {pixel_loss:.4f}, "
            + rf"Test Loss KL Div: {kl_div:.4f}, ",
            fontsize=15,
        )

        filename = save_dir / f"surface_reconstruction_{heliostat_name}_flux.png"
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
    training_losses_list = results["surface_reconstruction"]["loss"]
    training_errors = (
        torch.min(torch.stack(training_losses_list, dim=0), dim=0).values.cpu().numpy()
    )

    testing_loss_pixel_1 = results["surface_reconstruction"]["loss_history"][0][0][0][
        "test_loss"
    ]["pixel_loss"]
    testing_loss_pixel_2 = results["surface_reconstruction"]["loss_history"][1][0][0][
        "test_loss"
    ]["pixel_loss"]
    testing_loss_pixel_3 = results["surface_reconstruction"]["loss_history"][2][0][0][
        "test_loss"
    ]["pixel_loss"]

    testing_errors_pixel = (
        torch.cat(
            [testing_loss_pixel_1, testing_loss_pixel_2, testing_loss_pixel_3], dim=0
        )
        .cpu()
        .numpy()
    )

    testing_loss_kl_div_1 = results["surface_reconstruction"]["loss_history"][0][0][0][
        "test_loss"
    ]["kl_div"]
    testing_loss_kl_div_2 = results["surface_reconstruction"]["loss_history"][1][0][0][
        "test_loss"
    ]["kl_div"]
    testing_loss_kl_div_3 = results["surface_reconstruction"]["loss_history"][2][0][0][
        "test_loss"
    ]["kl_div"]

    testing_errors_kl_div = (
        torch.cat(
            [testing_loss_kl_div_1, testing_loss_kl_div_2, testing_loss_kl_div_3], dim=0
        )
        .cpu()
        .numpy()
    )

    for errors, name in zip(
        [training_errors, testing_errors_pixel, testing_errors_kl_div],
        ["Training Loss", "Testing Pixel Loss", "Testing KL Divergence"],
    ):
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
            label="Histogram",
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

        ax1.set_xlabel(rf"\textbf{{{name}}}")
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
            save_dir
            / f"run_{results_number}/surface_reconstruction_error_analysis_{name.lower().replace(' ', '_')}.png"
        )
        fig.tight_layout()
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved surface reconstruction error analysis plot at: {filename}.")


def plot_kinematics_reconstruction_flux(
    results: dict[str, Any],
    save_dir: pathlib.Path,
    results_number: int,
) -> None:
    """
    Plot the kinematics reconstruction flux results.

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
        if reconstruction not in results:
            continue

        flux_data_before = results[reconstruction]["flux_plot_data_before"]
        flux_data_after = results[reconstruction]["flux_plot_data_after"]

        heliostat_names = list(flux_data_before.keys())

        for heliostat_idx, heliostat_name in enumerate(heliostat_names):
            if heliostat_idx == 10:
                break
            test_loss = results[reconstruction]["loss_history"][0][0]["test_loss"]
            pixel_loss = test_loss["pixel_loss"][heliostat_idx].detach().cpu().item()
            kl_div = test_loss["kl_div"][heliostat_idx].detach().cpu().item()
            focal_spot_loss = (
                test_loss["focal_spot_loss"][heliostat_idx].detach().cpu().item()
            )

            measured_flux = (
                flux_data_before[heliostat_name]["measured_flux"].detach().cpu()
            )

            artist_flux_before = (
                flux_data_before[heliostat_name]["artist_flux"].detach().cpu()
            )

            artist_flux_after = (
                flux_data_after[heliostat_name]["artist_flux"].detach().cpu()
            )

            n_samples = min(3, measured_flux.shape[0])

            measured_flux = measured_flux[:n_samples]
            artist_flux_before = artist_flux_before[:n_samples]
            artist_flux_after = artist_flux_after[:n_samples]

            fig, axes = plt.subplots(
                n_samples,
                3,
                figsize=(9, 3 * n_samples),
            )

            if n_samples == 1:
                axes = axes[np.newaxis, :]

            for col_idx, label in enumerate(col_labels):
                axes[0, col_idx].set_title(
                    rf"\textbf{{{label}}}",
                    fontsize=13,
                )

            for sample_idx in range(n_samples):
                axes[sample_idx, 0].imshow(
                    measured_flux[sample_idx],
                    cmap=cmap,
                )
                axes[sample_idx, 1].imshow(
                    artist_flux_before[sample_idx],
                    cmap=cmap,
                )
                axes[sample_idx, 2].imshow(
                    artist_flux_after[sample_idx],
                    cmap=cmap,
                )
                axes[sample_idx, 0].set_ylabel(
                    rf"\textbf{{Sample {sample_idx + 1}}}",
                    rotation=90,
                    fontsize=12,
                    labelpad=15,
                )
                for col_idx in range(3):
                    axes[sample_idx, col_idx].set_xticks([])
                    axes[sample_idx, col_idx].set_yticks([])

            position = results["heliostat_positions"][heliostat_name]
            position_str = ", ".join(f"{coord:.2f}" for coord in position[:3])

            fig.suptitle(
                rf"\textbf{{Heliostat {heliostat_name}}}"
                + "\n"
                + rf"\textit{{ENU Position: {position_str}}}"
                + "\n"
                + rf"\textit{{Test Loss Pixel: {pixel_loss:.4f}, "
                + rf"Test Loss KL Div: {kl_div:.4f}, "
                + rf"Test Loss Focal Spot: {focal_spot_loss:.4f}}}",
                fontsize=15,
            )

            filename = save_dir / f"{reconstruction}_{heliostat_name}_fluxes.png"
            fig.tight_layout()
            fig.savefig(filename, dpi=300)
            plt.close(fig)

            print(f"Saved kinematics reconstruction flux plot at: {filename}.")


def plot_kinematics_training_error_analysis(
    results: dict[str, Any],
    save_dir: pathlib.Path,
    angle: bool = True,
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
    angle : bool
        Loss in angles or meters.
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
        if reconstruction in results.keys():
            if angle:
                errors_mrad = (
                    results[reconstruction]["loss"].detach().cpu().numpy() * 1000
                )
                errors_m = (errors_mrad / 1000) * distances
            else:
                errors_m = results[reconstruction]["loss"].detach().cpu().numpy()
                errors_mrad = (errors_m / distances) * 1000
            fig = plt.figure(figsize=(10, 4))
            gs = GridSpec(
                1,
                2,
                figure=fig,
                width_ratios=[1, 1.3],
                wspace=0.3,
            )
            ax_kde_m = fig.add_subplot(gs[0, 0])

            ax_dist_m = fig.add_subplot(gs[0, 1])
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
                handles_m + handles_a,
                labels_m + labels_a,
                fontsize=8,
                loc="upper right",
            )

            fig.suptitle(
                rf"\textbf{{Kinematics Reconstruction with {label}}}", fontsize=14
            )

            filename = save_dir / f"{reconstruction}_training_error_analysis.png"
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close(fig)

            print(
                f"Saved kinematics reconstruction error analysis plot at: {filename}."
            )


def plot_kinematics_testing_error_analysis(
    results: dict[str, Any],
    save_dir: pathlib.Path,
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
        if reconstruction in results.keys():
            test_losses = results[reconstruction]["loss_history"][0][0]["test_loss"]
            pixel_loss = test_losses["pixel_loss"].detach().cpu().numpy()
            kl_div = test_losses["kl_div"].detach().cpu().numpy()
            focal_spot_loss = test_losses["focal_spot_loss"].detach().cpu().numpy()
            angle_loss = (focal_spot_loss / distances) * 1000

            for loss, loss_name in zip(
                [pixel_loss, kl_div, focal_spot_loss, angle_loss],
                [
                    "Pixel Loss",
                    "KL Divergence",
                    "Focal Spot Loss (m)",
                    "Focal Spot Loss (mrad)",
                ],
            ):
                fig = plt.figure(figsize=(10, 4))
                gs = GridSpec(
                    1,
                    2,
                    figure=fig,
                    width_ratios=[1, 1.3],
                    wspace=0.3,
                )

                ax_kde = fig.add_subplot(gs[0, 0])
                ax_dist = fig.add_subplot(gs[0, 1])

                x_max = max(loss)
                x_vals = np.linspace(0, x_max, 100)
                kde = gaussian_kde(loss, bw_method="scott")
                kde_vals = kde(x_vals)

                ax_kde.hist(
                    loss,
                    bins=25,
                    range=(0, x_max),
                    density=True,
                    alpha=0.3,
                    color=plot_colors["lightblue"],
                    label="Histogram",
                )
                ax_kde.plot(
                    x_vals,
                    kde_vals,
                    color=plot_colors["lightblue"],
                    label="KDE",
                )
                ax_kde.axvline(
                    np.mean(loss),
                    linestyle="--",
                    color=plot_colors["lightblue"],
                    label=f"Mean: {np.mean(loss):.3f}",
                )
                ax_kde.set_xlabel(rf"\textbf{{{loss_name}}}")
                ax_kde.set_ylabel(r"\textbf{Density}")
                ax_kde.grid(True)
                ax_kde.legend(fontsize=8)

                ax_dist.scatter(
                    distances,
                    loss,
                    marker="o",
                    color=plot_colors["lightblue"],
                    alpha=0.7,
                    label=f"{loss_name}",
                )

                x_fit = np.linspace(distances.min(), distances.max(), 200)
                fit = np.poly1d(np.polyfit(distances, loss, 1))

                ax_dist.plot(
                    x_fit, fit(x_fit), linestyle="--", color=plot_colors["lightblue"]
                )

                ax_dist.set_xlabel(r"\textbf{Heliostat Distance from Tower [m]}")
                ax_dist.set_ylabel(rf"\textbf{{{loss_name}}}")
                ax_dist.grid(True)

                handles, labels = ax_dist.get_legend_handles_labels()
                ax_dist.legend(
                    handles,
                    labels,
                    fontsize=8,
                    loc="upper right",
                )

                fig.suptitle(
                    rf"\textbf{{Kinematics Reconstruction with {label}, {loss_name}}}",
                    fontsize=14,
                )

                filename = (
                    save_dir
                    / f"{reconstruction}_testing_error_analysis_{loss_name.lower().replace(' ', '_')}.png"
                )
                fig.savefig(filename, dpi=300, bbox_inches="tight")
                plt.close(fig)

                print(
                    f"Saved kinematics reconstruction error analysis plot at: {filename}."
                )


def plot_model_reconstruction(
    results: dict[str, Any],
    save_dir: pathlib.Path,
    results_number: int,
    scenario_path: pathlib.Path,
    device: torch.device,
    baseline_aim_point: torch.Tensor,
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
    scenario_path : pathlib.Path
        Path to the scenario.
    device : torch.device
        Device for calculations.
    baseline_aim_point : torch.Tensor
        Aim point coordinate of the measurement.
    """
    with h5py.File(scenario_path) as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file,
            device=device,
        )
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
        measured_flux_normed,
        ideal_model_flux,
        reconstructed_surfaces_flux,
        reconstructed_kinematics_flux,
        combined_reconstruction_flux,
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

    bitmap_resolution = torch.tensor(
        [measured_flux_normed.shape[0], measured_flux_normed.shape[1]], device=device
    )

    centers_bitmaps = [bitmap.get_center_of_mass(img[None])[0] for img in images]
    _ = coordinates.bitmap_coordinates_to_target_coordinates(
        bitmap_coordinates=torch.stack(centers_bitmaps),
        bitmap_resolution=bitmap_resolution,
        solar_tower=scenario.solar_tower,
        target_area_indices=torch.full((len(images),), fill_value=3, device=device),
        device=device,
    )

    # baseline_aim_point_bitmap_coords = target_coordinates_to_bitmap_coordinates(
    #     world_coordinates=torch.tensor([baseline_aim_point], device=device),
    #     bitmap_resolution=bitmap_resolution,
    #     solar_tower=scenario.solar_tower,
    #     target_area_indices=torch.full((1,), fill_value=3, device=device),
    #     device=device
    # )

    colors = ["red", "cyan", "pink", "black", "green"]

    for idx, (ax, flux) in enumerate(zip(axes, images)):
        im = ax.imshow(flux.cpu().detach(), cmap=cmap, vmin=vmin, vmax=vmax)
        # ax.scatter(
        #     baseline_aim_point_bitmap_coords[0, 1].cpu().detach(),
        #     baseline_aim_point_bitmap_coords[0, 0].cpu().detach(),
        #     marker="x",
        #     s=120,
        #     edgecolors="black",
        #     linewidths=1.5,
        #     label="baseline aim point",
        # )
        for center_idx, ((cy, cx), title) in enumerate(zip(centers_bitmaps, titles)):
            ax.scatter(
                cy.cpu().detach(),
                cx.cpu().detach(),
                marker="x",
                s=120,
                c=colors[center_idx],
                linewidths=3,
                label=title,
            )
        ax.set_title(titles[idx], fontsize=13)
        if idx == 0:
            ax.legend(
                loc="upper right",
                fontsize=9,
                frameon=True,
            )

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
    filename = save_dir / f"run_{results_number}/model_reconstructions_flux.png"
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
    optimal_distribution = results["optimal_flux"]
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
    homogeneous_distribution_normed = optimal_distribution * (
        optimized_flux.sum() / optimal_distribution.sum()
    )

    # Compute KL divergences
    kl_divergence = KLDivergenceLoss()
    kl_div_measured_homogeneous = kl_divergence(
        homogeneous_distribution_normed.unsqueeze(0),
        measured_flux_normed.unsqueeze(0),
        reduction_dimensions=(1, 2),
    )
    kl_div_unoptimized_homogeneous = kl_divergence(
        optimal_distribution.unsqueeze(0),
        unoptimized_flux.unsqueeze(0),
        reduction_dimensions=(1, 2),
    )
    kl_div_optimized_homogeneous = kl_divergence(
        optimal_distribution.unsqueeze(0),
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
    filename = save_dir / f"run_{results_number}/aim_point_optimization_flux.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved aim point optimization flux plot at: {filename}.")


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
    # Locate this script and the repository root (two levels up).
    script_dir = pathlib.Path(__file__).resolve().parent
    default_config_path = script_dir / "config.yaml"
    project_root = script_dir.parent.parent

    def _make_abs(p: str | pathlib.Path) -> pathlib.Path:
        """Resolve a possibly‑relative path relative to the repository root (where YAML paths were written)."""
        p = pathlib.Path(p).expanduser()
        return p if p.is_absolute() else (project_root / p).resolve()

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
    # Resolve any directory defaults that are stored in the YAML relative to the repo root.
    results_dir_default = _make_abs(config.get("results_dir", "./results"))
    plots_dir_default = _make_abs(config.get("plots_dir", "./plots"))
    # The following entries are not used directly in this script, but we resolve them
    # here for completeness – they may be needed by other parts of the project.
    _ = _make_abs(
        config.get("measured_data_dir", "./examples/field_optimizations/measured_data")
    )
    _ = _make_abs(
        config.get(
            "data_for_stral_dir", "./examples/field_optimizations/data_for_stral"
        )
    )
    scenarios_dir_default = _make_abs(
        config.get("scenarios_dir", "./examples/field_optimizations/scenarios")
    )
    basic_config_default = config.get("basic_config", {})

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
        default=str(results_dir_default),
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        help="Path to save the plots.",
        default=str(plots_dir_default),
    )
    parser.add_argument(
        "--scenarios_dir",
        type=str,
        help="Path to the directory for saving the generated scenarios.",
        default=str(scenarios_dir_default),
    )
    parser.add_argument(
        "--basic_config",
        type=dict[Any],  # type: ignore[arg-type, misc]
        help="Config.",
        default=basic_config_default,
    )

    # Re-parse the full set of arguments.
    args = parser.parse_args()
    device = get_device(torch.device(args.device))

    # Convert CLI‑provided paths (which may be relative) to absolute ones.
    results_dir = _make_abs(args.results_dir)
    plots_dir = _make_abs(args.plots_dir)

    for case in ["baseline", "full_field"]:
        results_number = 0
        results_path = results_dir / case / f"results_{results_number}.pt"
        plots_path = plots_dir / case
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

        scenarios_dir = _make_abs(args.scenarios_dir)
        scenario_path = scenarios_dir / f"ideal_{case}_scenario.h5"
        basic_config = args.basic_config

        plot_heliostat_positions(scenario_dir=scenarios_dir, save_dir=plots_path)

        plot_kinematics_training_error_analysis(
            results=results,
            save_dir=plots_path,
            results_number=results_number,
        )
        plot_kinematics_testing_error_analysis(
            results=results,
            save_dir=plots_path,
            results_number=results_number,
        )
        plot_kinematics_reconstruction_flux(
            results=results, save_dir=plots_path, results_number=results_number
        )

        plot_model_reconstruction(
            results=results,
            save_dir=plots_path,
            results_number=results_number,
            scenario_path=scenario_path,
            device=device,
            baseline_aim_point=basic_config["baseline_aim_point"],
        )

        plot_surface_error_analysis(
            results=results, save_dir=plots_path, results_number=results_number
        )

        plot_surface_reconstruction_flux(
            results=results, save_dir=plots_path, results_number=results_number
        )

        plot_aim_point_flux(
            results=results, save_dir=plots_path, results_number=results_number
        )
