import math
import subprocess
from typing import List, Optional

import h5py
import matplotlib.pyplot as plt
import torch
from matplotlib.pyplot import tight_layout

from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.scenario import Scenario

# If you have already generated the tutorial scenario yourself, you can leave this boolean as False. If not, set it to
# true and a pre-generated scenario file will be downloaded for this tutorial!
USE_DOWNLOADED_DATA = False

if USE_DOWNLOADED_DATA:
    url = "https://drive.google.com/uc?export=download&id=1C3mG_RpatF27rZaWyUWcyUVlcTadT8FH"
    output_filename = "tutorial_scenario.h5"
    command = ["wget", "-O", output_filename, url]
    result = subprocess.run(command, capture_output=True, text=True)


# Load the scenario.
with h5py.File("tutorial_scenario.h5", "r") as f:
    example_scenario = Scenario.load_scenario_from_hdf5(scenario_file=f)

# Inspect the secnario.
print(example_scenario)
print(f"The light source is a {example_scenario.light_sources.light_source_list[0]}")
print(
    f"The receiver type is {example_scenario.receivers.receiver_list[0].receiver_type}"
)
single_heliostat = example_scenario.heliostats.heliostat_list[0]
print(f"The heliostat position is: {single_heliostat.position}")
print(f"The heliostat is aiming at: {single_heliostat.aim_point}")

# Define the incident ray direction for when the sun is in the south.
incident_ray_direction_south = torch.tensor([0.0, -1.0, 0.0, 0.0])

# Save original surface points
original_surface_points, _ = single_heliostat.surface.get_surface_points_and_normals()

# Align the heliostat
single_heliostat.set_aligned_surface(
    incident_ray_direction=incident_ray_direction_south
)

# Define colors for each facet
colors = ["r", "g", "b", "y"]

# Create a 3D plot
fig = plt.figure(figsize=(14, 6))  # Adjust figure size as needed
gs = fig.add_gridspec(
    1, 2, width_ratios=[1, 1], wspace=0.3
)  # Adjust width_ratios and wspace as needed

# Create subplots
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")

# Plot each facet
for i in range(len(single_heliostat.surface.facets)):
    e_origin = original_surface_points[i, :, 0].detach().numpy()
    n_origin = original_surface_points[i, :, 1].detach().numpy()
    u_origin = original_surface_points[i, :, 2].detach().numpy()
    e_aligned = (
        single_heliostat.current_aligned_surface_points[i, :, 0].detach().numpy()
    )
    n_aligned = (
        single_heliostat.current_aligned_surface_points[i, :, 1].detach().numpy()
    )
    u_aligned = (
        single_heliostat.current_aligned_surface_points[i, :, 2].detach().numpy()
    )
    ax1.scatter(e_origin, n_origin, u_origin, color=colors[i], label=f"Facet {i+1}")
    ax2.scatter(e_aligned, n_aligned, u_aligned, color=colors[i], label=f"Facet {i+1}")

# Add labels
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

# Create a single legend for both subplots
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncols=4)


# Show the plot
plt.show()


# Calculate the preferred reflection direction.
single_heliostat.set_preferred_reflection_direction(rays=-incident_ray_direction_south)

# Inspect if the shape of the surface points and the preferred reflection direction is identical.
print(
    single_heliostat.current_aligned_surface_points.shape
    == single_heliostat.preferred_reflection_direction.shape
)

# Define the raytracer.
raytracer = HeliostatRayTracer(scenario=example_scenario, batch_size=100)

# Perform heliostat-based raytracing.
image_south = raytracer.trace_rays()
image_south = raytracer.normalize_bitmap(image_south)

# Plot the result.
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(image_south.T.detach().numpy(), cmap="inferno")
tight_layout()


# Define helper functions to enable us to repeat the process!
def align_reflect_and_trace_rays(light_direction: torch.Tensor) -> torch.Tensor:
    """
    Align the heliostat, calculate the preferred reflection direction, and perform heliostat raytracing.

    Parameters
    ----------
    light_direction : torch.Tensor
        The direction of the incoming light on the heliostat.

    Returns
    -------
    torch.Tensor
        A tensor containing the distribution strengths used to generate the image on the receiver.
    """
    single_heliostat.set_aligned_surface(incident_ray_direction=light_direction)
    single_heliostat.set_preferred_reflection_direction(rays=-light_direction)
    return raytracer.normalize_bitmap(raytracer.trace_rays())


def plot_multiple_images(
    *image_tensors: torch.Tensor, names: Optional[List[str]] = None
) -> None:
    """
    Plot multiple receiver raytracing images in a grid.

    This function is flexible and able to plot an arbitrary number of images depending on the number of image tensors provided.
    Note that the list of names must be the same length as the number of provided images, otherwise the images will be untitled.

    Parameters
    ----------
    image_tensors : torch.Tensor
        An arbitrary number of image tensors to be plotted.
    names : List[str], optional
        The names of the images to be plotted.
    """
    # Calculate the number of images and determine the size of the grid based on the number of images
    n = len(image_tensors)
    grid_size = math.ceil(math.sqrt(n))

    # Create a subplot with the appropriate size
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    # Flatten axes array for easy iteration if it's more than 1D
    if grid_size > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # Plot each tensor
    for i, image in enumerate(image_tensors):
        ax = axes[i]
        ax.imshow(image.T.detach().numpy(), cmap="inferno")
        if names is not None and i < len(names):
            ax.set_title(names[i])
        else:
            ax.set_title(f"Untitled Image {i+1}")

    # Hide unused subplots
    for j in range(i + 1, grid_size * grid_size):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


# Consider multiple incident ray directions and plot the result
# Define light directions.
incident_ray_direction_east = torch.tensor([1.0, 0.0, 0.0, 0.0])
incident_ray_direction_west = torch.tensor([-1.0, 0.0, 0.0, 0.0])
incident_ray_direction_above = torch.tensor([0.0, 0.0, 1.0, 0.0])

# Perform alignment and raytracing to generate flux density images.
image_east = align_reflect_and_trace_rays(light_direction=incident_ray_direction_east)
image_west = align_reflect_and_trace_rays(light_direction=incident_ray_direction_west)
image_above = align_reflect_and_trace_rays(light_direction=incident_ray_direction_above)

# Plot the resulting images
plot_multiple_images(
    image_south,
    image_east,
    image_west,
    image_above,
    names=["South", "East", "West", "Above"],
)
