import json

import matplotlib.pyplot as plt
import torch
import os

from artist.data_loader import paint_loader
from artist.util import config_dictionary, utils


def calibration_path_to_sun_and_tower(paths, target_area_names, device):  # noqa: D103
    elevations = []
    azimuths = []
    targets = []

    for path in paths:
        with open(path, "r") as file:
            dict = json.load(file)

        elevations.append(dict[config_dictionary.paint_light_source_elevation])
        azimuths.append(dict[config_dictionary.paint_light_source_azimuth])

        targets.append(dict[config_dictionary.paint_calibration_target])

    sun_positions = utils.convert_3d_points_to_4d_format(
        paint_loader.azimuth_elevation_to_enu(
            azimuth=torch.tensor(azimuths),
            elevation=torch.tensor(elevations),
            device=device,
        ),
        device=device,
    )

    incident_rays = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device) - sun_positions

    return incident_rays, targets


def plot_surface_points(points, name):
    plt.clf()
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    for facet in points:
        x, y, z = (
            facet[:, 0].cpu().detach(),
            facet[:, 1].cpu().detach(),
            facet[:, 2].cpu().detach(),
        )
        sc = axes.scatter(x, y, c=z, cmap="viridis", s=5)

        plt.title("Points")
        plt.xlabel("X")
        plt.ylabel("Y")

    plt.colorbar(sc, label="z values")
    plt.tight_layout()
    plt.axis("equal")
    plt.savefig(f"2d_surface_{name}.png")
    plt.clf()
    plt.close()


def plot_normal_angle_map(surface_points, surface_normals, reference_direction, name):
    # Normalize vectors
    plt.clf()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    normals = (
        (
            surface_normals[..., :3]
            / torch.linalg.norm(surface_normals[..., :3], axis=-1, keepdims=True)
        )
        .cpu()
        .detach()
    )
    ref = (
        (reference_direction[..., :3] / torch.linalg.norm(reference_direction[..., :3]))
        .cpu()
        .detach()
    )

    for facet_points, facet_normals in zip(surface_points.cpu().detach(), normals):
        x, y, z = (
            facet_points[:, 0].cpu().detach(),
            facet_points[:, 1].cpu().detach(),
            facet_points[:, 2].cpu().detach(),
        )

        sc0 = axes[0].scatter(x, y, c=z, cmap="viridis", s=7)
        axes[0].set_title("Surface points")

        cos_theta = facet_normals @ ref
        angles = torch.arccos(torch.clip(cos_theta, -1.0, 1.0))

        angles = torch.clip(angles, -0.1, 0.1)

        sc1 = axes[1].scatter(x, y, c=angles, cmap="plasma", s=7)
        axes[1].set_title("Angle map normals")
    
    plt.tight_layout()
    # plt.axis('equal')
    plt.savefig(f"{name}.png")
    plt.clf()
    plt.close()


def plot_multiple_fluxes(reconstructed, references, name):
    fig1, axes1 = plt.subplots(nrows=reconstructed.shape[0], ncols=2, figsize=(24, 150))
    for i in range(reconstructed.shape[0]):
        axes1[i, 0].imshow(reconstructed[i].cpu().detach(), cmap="gray")
        axes1[i, 0].set_title(f"Reconstructed {i}")
        axes1[i, 0].axis("off")

        axes1[i, 1].imshow(references[i].cpu().detach(), cmap="gray")
        axes1[i, 1].set_title(f"Reference {i}")
        axes1[i, 1].axis("off")
    
    plt.tight_layout()
    plt.savefig(f"{name}.png")
    plt.clf()
    plt.close()
