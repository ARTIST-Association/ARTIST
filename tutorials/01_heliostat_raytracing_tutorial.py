import math
import subprocess
from typing import Optional, Union

import h5py
import matplotlib.pyplot as plt
import torch
from matplotlib.pyplot import tight_layout

from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.scenario import Scenario

# If you have already generated the tutorial scenario yourself, you can leave this boolean as False. If not, set it to
# true and a pre-generated scenario file will be downloaded for this tutorial!
DOWNLOAD_DATA = False
scenario_name = "please/insert/the/path/to/the/scenario/here/name.h5"

if DOWNLOAD_DATA:
    url = "https://drive.google.com/uc?export=download&id=1WSJyabylkK8cMDgulymVP5nPNGNf_BzN"
    output_filename = "tutorial_scenario.h5"
    command = ["wget", "-O", output_filename, url]
    result = subprocess.run(command, capture_output=True, text=True)
    scenario_name = output_filename

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the scenario.
with h5py.File(scenario_name, "r") as f:
    example_scenario = Scenario.load_scenario_from_hdf5(scenario_file=f, device=device)

# Inspect the scenario.
print(example_scenario)
print(f"The light source is a {example_scenario.light_sources.light_source_list[0]}")
print(
    f"The receiver type is {example_scenario.receivers.receiver_list[0].receiver_type}"
)
single_heliostat = example_scenario.heliostats.heliostat_list[0]
print(f"The heliostat position is: {single_heliostat.position}")
print(f"The heliostat is aiming at: {single_heliostat.aim_point}")

# Define the incident ray direction for when the sun is in the south.
incident_ray_direction_south = torch.tensor([0.0, -1.0, 0.0, 0.0], device=device)

# Save original surface points.
original_surface_points, _ = single_heliostat.surface.get_surface_points_and_normals(
    device=device
)

# Align the heliostat.
single_heliostat.set_aligned_surface_with_incident_ray_direction(
    incident_ray_direction=incident_ray_direction_south, device=device
)

# Define colors for each facet.
colors = ["r", "g", "b", "y"]

# Create a 3D plot.
fig = plt.figure(figsize=(14, 6))  # Adjust figure size as needed.
gs = fig.add_gridspec(
    1, 2, width_ratios=[1, 1], wspace=0.3
)  # Adjust width_ratios and wspace as needed.

# Create subplots.
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")

# Plot each facet
for i in range(len(single_heliostat.surface.facets)):
    e_origin = original_surface_points[i, :, 0].cpu().detach().numpy()
    n_origin = original_surface_points[i, :, 1].cpu().detach().numpy()
    u_origin = original_surface_points[i, :, 2].cpu().detach().numpy()
    e_aligned = (
        single_heliostat.current_aligned_surface_points[i, :, 0].cpu().detach().numpy()
    )
    n_aligned = (
        single_heliostat.current_aligned_surface_points[i, :, 1].cpu().detach().numpy()
    )
    u_aligned = (
        single_heliostat.current_aligned_surface_points[i, :, 2].cpu().detach().numpy()
    )
    ax1.scatter(e_origin, n_origin, u_origin, color=colors[i], label=f"Facet {i+1}")
    ax2.scatter(e_aligned, n_aligned, u_aligned, color=colors[i], label=f"Facet {i+1}")

# Add labels.
ax1.set_xlabel("E")
ax1.set_ylabel("N")
ax1.set_zlabel("U")
ax2.set_xlabel("E")
ax2.set_ylabel("N")
ax2.set_zlabel("U")
ax1.set_zlim(-0.5, 0.5)
ax2.set_ylim(4.5, 5.5)
ax1.set_title("Original surface")
ax2.set_title("Aligned surface")

# Remove axis numbers to create a cleaner visualization.
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_zticks([])
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_zticks([])

# Create a single legend for both subplots.
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncols=4)


# Show the plot.
plt.show()

# Define the raytracer.
raytracer = HeliostatRayTracer(scenario=example_scenario, batch_size=100)

# Perform heliostat-based raytracing.
image_south = raytracer.trace_rays(
    incident_ray_direction=incident_ray_direction_south, device=device
)
image_south = raytracer.normalize_bitmap(image_south)

# Plot the result.
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(image_south.T.cpu().detach().numpy(), cmap="inferno")
tight_layout()


# Define helper functions to enable us to repeat the process!
def align_and_trace_rays(
    light_direction: torch.Tensor, device: Union[torch.device, str] = "cuda"
) -> torch.Tensor:
    """
    Align the heliostat and perform heliostat raytracing.

    Parameters
    ----------
    light_direction : torch.Tensor
        The direction of the incoming light on the heliostat.
    device : Union[torch.device, str]
        The device on which to initialize tensors (default: cuda).

    Returns
    -------
    torch.Tensor
        A tensor containing the distribution strengths used to generate the image on the receiver.
    """
    single_heliostat.set_aligned_surface_with_incident_ray_direction(
        incident_ray_direction=light_direction, device=device
    )
    return raytracer.normalize_bitmap(
        raytracer.trace_rays(incident_ray_direction=light_direction, device=device)
    )


def plot_multiple_images(
    *image_tensors: torch.Tensor, names: Optional[list[str]] = None
) -> None:
    """
    Plot multiple receiver raytracing images in a grid.

    This function is flexible and able to plot an arbitrary number of images depending on the number of image tensors
    provided. Note that the list of names must be the same length as the number of provided images, otherwise the images
    will be untitled.

    Parameters
    ----------
    image_tensors : torch.Tensor
        An arbitrary number of image tensors to be plotted.
    names : list[str], optional
        The names of the images to be plotted.
    """
    # Calculate the number of images and determine the size of the grid based on the number of images.
    n = len(image_tensors)
    grid_size = math.ceil(math.sqrt(n))

    # Create a subplot with the appropriate size.
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    # Flatten axes array for easy iteration if it's more than 1D.
    if grid_size > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # Plot each tensor.
    for i, image in enumerate(image_tensors):
        ax = axes[i]
        ax.imshow(image.T.cpu().detach().numpy(), cmap="inferno")
        if names is not None and i < len(names):
            ax.set_title(names[i])
        else:
            ax.set_title(f"Untitled Image {i+1}")

    # Hide unused subplots.
    for j in range(i + 1, grid_size * grid_size):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


# Consider multiple incident ray directions and plot the result.
# Define light directions.
incident_ray_direction_east = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
incident_ray_direction_west = torch.tensor([-1.0, 0.0, 0.0, 0.0], device=device)
incident_ray_direction_above = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device)

# Perform alignment and raytracing to generate flux density images.
image_east = align_and_trace_rays(
    light_direction=incident_ray_direction_east, device=device
)
image_west = align_and_trace_rays(
    light_direction=incident_ray_direction_west, device=device
)
image_above = align_and_trace_rays(
    light_direction=incident_ray_direction_above, device=device
)

# Plot the resulting images.
plot_multiple_images(
    image_south,
    image_east,
    image_west,
    image_above,
    names=["South", "East", "West", "Above"],
)
