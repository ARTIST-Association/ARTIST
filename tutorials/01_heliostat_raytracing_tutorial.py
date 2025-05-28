import math
import pathlib
from typing import Optional, Union

import h5py
import matplotlib.pyplot as plt
import torch
from matplotlib.pyplot import tight_layout

from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util import set_logger_config
from artist.util.scenario import Scenario

# If you have already generated the tutorial scenario yourself, you can use that scenario,
# create and use any custom scenario, or use one provided in the artist/tutorials/data/scenarios directory.
# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path("please/insert/the/path/to/the/scenario/here/scenario.h5")

# Set up logger.
set_logger_config()

# Set the device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the scenario.
with h5py.File(scenario_path) as scenario_path:
    scenario = Scenario.load_scenario_from_hdf5(
        scenario_file=scenario_path, device=device
    )

# Inspect the scenario.
print(scenario)
print(f"The light source is a {scenario.light_sources.light_source_list[0]}.")
print(f"The first target area is a {scenario.target_areas.names[0]}.")
print(
    f"The first heliostat in the first group in the field is heliostat {scenario.heliostat_field.heliostat_groups[0].names[0]}."
)
print(
    f"Heliostat {scenario.heliostat_field.heliostat_groups[0].names[0]} is located at: {scenario.heliostat_field.heliostat_groups[0].positions[0].tolist()}."
)
print(
    f"Heliostat {scenario.heliostat_field.heliostat_groups[0].names[0]} is aiming at: {scenario.heliostat_field.heliostat_groups[0].kinematic.aim_points[0].tolist()}."
)

# Let's say we only want to consider one Heliostat for the beginning.
# We will choose the first Heliostat, with index 0 by activating it.
active_heliostats_indices = torch.tensor([0], device=device)

# Each heliostat has an aim point, it makes sense to choose an aimpoint on one of the target areas.
# We select the first target area as the designated target for this heliostat.
target_area_indices = torch.tensor([0], device=device)

# Since we only have one helisotat we need to define a single incident ray direction.
# When the sun is directly in the south, the rays point directly to the north.
# Incident ray directions need to be normed.
incident_ray_directions = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device)

# Save the original surface points of the one active heliostat.
original_surface_points = scenario.heliostat_field.heliostat_groups[0].surface_points[
    active_heliostats_indices
]

# Align the heliostat(s).
scenario.heliostat_field.heliostat_groups[
    0
].align_surfaces_with_incident_ray_directions(
    aim_points=scenario.target_areas.centers[target_area_indices],
    incident_ray_directions=incident_ray_directions,
    device=device,
)

# Save the aligned surface points of the one active heliostat.
# The original surface points are saved for all heliostats, active or not.
# The aligned surface points are saved only for the active/current/aligned heliostats.
# That is why we do not need to select specific indices here.
aligned_surface_points = scenario.heliostat_field.heliostat_groups[
    0
].current_aligned_surface_points

# Let's plot the original and the aligned surface points.
# Define colors for each facet of the heliostat.
colors = ["r", "g", "b", "y"]

# Create a 3D plot.
fig = plt.figure(figsize=(14, 6))  # Adjust figure size as needed.
gs = fig.add_gridspec(
    1, 2, width_ratios=[1, 1], wspace=0.3
)  # Adjust width_ratios and wspace as needed.

# Create subplots.
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")

# Plot each facet of the first heliostat in the scenario.
number_of_facets = 4
number_of_surface_points_per_facet = original_surface_points.shape[1]
batch_size = number_of_surface_points_per_facet // number_of_facets
for i in range(number_of_facets):
    start = i * batch_size
    end = start + batch_size
    e_origin = original_surface_points[0, start:end, 0].cpu().detach().numpy()
    n_origin = original_surface_points[0, start:end, 1].cpu().detach().numpy()
    u_origin = original_surface_points[0, start:end, 2].cpu().detach().numpy()
    e_aligned = aligned_surface_points[0, start:end, 0].cpu().detach().numpy()
    n_aligned = aligned_surface_points[0, start:end, 1].cpu().detach().numpy()
    u_aligned = aligned_surface_points[0, start:end, 2].cpu().detach().numpy()
    ax1.scatter(e_origin, n_origin, u_origin, color=colors[i], label=f"Facet {i + 1}")
    ax2.scatter(
        e_aligned, n_aligned, u_aligned, color=colors[i], label=f"Facet {i + 1}"
    )

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

# Create a ray tracer.
ray_tracer = HeliostatRayTracer(
    scenario=scenario,
    heliostat_group=scenario.heliostat_field.heliostat_groups[0],
)

# Perform heliostat-based ray tracing.
image_south = ray_tracer.trace_rays(
    incident_ray_directions=incident_ray_directions,
    target_area_mask=target_area_indices,
    device=device,
)

# Plot the result.
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(image_south[0].cpu().detach().numpy(), cmap="inferno")
tight_layout()


# Define helper functions to enable us to repeat the process!
def align_and_trace_rays(
    light_direction: torch.Tensor,
    active_heliostats_indices: torch.Tensor,
    target_area_indices: torch.Tensor,
    device: Union[torch.device, str] = "cuda",
) -> torch.Tensor:
    """
    Align the heliostat and perform heliostat ray tracing.

    Parameters
    ----------
    light_directions : torch.Tensor
        The direction of the incoming light on the heliostat.
    active_heliostats_indices : torch.Tensor
        The indices of the active heliostats to be aligned.
    target_area_indices : torch.Tensor
        The indices of the target areas for each active heliostat.
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Returns
    -------
    torch.Tensor
        A tensor containing the distribution strengths used to generate the image on the receiver.
    """
    # Align all heliostats.
    scenario.heliostat_field.heliostat_groups[
        0
    ].align_surfaces_with_incident_ray_directions(
        aim_points=scenario.target_areas.centers[target_area_indices],
        incident_ray_directions=light_direction,
        device=device,
    )

    # Perform heliostat-based ray tracing.
    return ray_tracer.trace_rays(
        incident_ray_directions=light_direction,
        target_area_mask=target_area_indices,
        device=device,
    )


def plot_multiple_images(
    *image_tensors: torch.Tensor, names: Optional[list[str]] = None
) -> None:
    """
    Plot multiple receiver ray tracing images in a grid.

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
        ax.imshow(image.cpu().detach().numpy(), cmap="inferno")
        if names is not None and i < len(names):
            ax.set_title(names[i])
        else:
            ax.set_title(f"Untitled Image {i + 1}")

    # Hide unused subplots.
    for j in range(i + 1, grid_size * grid_size):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
    plt.savefig("tutorial3.png")


# Consider multiple incident ray directions and plot the result.
# Define light directions.
incident_ray_direction_east = torch.tensor([[-1.0, 0.0, 0.0, 0.0]], device=device)
incident_ray_direction_west = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
incident_ray_direction_above = torch.tensor([[0.0, 0.0, -1.0, 0.0]], device=device)

# Perform alignment and ray tracing to generate flux density images.
image_east = align_and_trace_rays(
    light_direction=incident_ray_direction_east,
    active_heliostats_indices=active_heliostats_indices,
    target_area_indices=target_area_indices,
    device=device,
)
image_west = align_and_trace_rays(
    light_direction=incident_ray_direction_west,
    active_heliostats_indices=active_heliostats_indices,
    target_area_indices=target_area_indices,
    device=device,
)
image_above = align_and_trace_rays(
    light_direction=incident_ray_direction_above,
    active_heliostats_indices=active_heliostats_indices,
    target_area_indices=target_area_indices,
    device=device,
)

# Plot the resulting images.
plot_multiple_images(
    image_south[0],
    image_east[0],
    image_west[0],
    image_above[0],
    names=["South", "East", "West", "Above"],
)
