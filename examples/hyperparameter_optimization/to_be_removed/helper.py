import matplotlib.pyplot as plt
import torch


def plot_normal_angle_map(surface_points, surface_normals, reference_direction, name):  # noqa: D103
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
        x, y, z = (  # noqa: F841
            facet_points[:, 0].cpu().detach(),
            facet_points[:, 1].cpu().detach(),
            facet_points[:, 2].cpu().detach(),
        )

        axes[0].set_title("Surface points")

        cos_theta = facet_normals @ ref
        angles = torch.arccos(torch.clip(cos_theta, -1.0, 1.0))

        angles = torch.clip(angles, -0.1, 0.1)

        axes[1].set_title("Angle map normals")

    plt.tight_layout()
    # plt.axis('equal')
    plt.savefig(f"2d_points_and_normals_{name}.png")
    plt.clf()


def plot_multiple_fluxes(reconstructed, references, name):  # noqa: D103
    fig1, axes1 = plt.subplots(nrows=reconstructed.shape[0], ncols=2, figsize=(24, 72))
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
