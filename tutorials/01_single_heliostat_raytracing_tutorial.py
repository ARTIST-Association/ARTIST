"""Single heliostat ray tracing tutorial."""

import math
import pathlib

import h5py
import matplotlib.pyplot as plt
import torch
from matplotlib.pyplot import tight_layout

from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer
from artist.scenario.scenario import Scenario
from artist.util import indices, set_logger_config
from artist.util.env import get_device

# This is an introductory tutorial to look at some of the basic elements of ARTIST. Therefore, it is designed to only
# work with a scenario containing a single heliostat. Please use the "single_heliostat_scenario.h5" provided in the
# scenarios folder or create your own scenario that only contains a single heliostat.

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path(
    "please/insert/the/path/to/the/scenario/here/scenarios/single_heliostat_scenario.h5"
)

# Set up logger.
set_logger_config()

# Set the device.
device = get_device()

# Load the scenario.
with h5py.File(scenario_path) as scenario_path:
    scenario = Scenario.load_scenario_from_hdf5(
        scenario_file=scenario_path, device=device
    )

# Inspect the scenario.
print(scenario)
print(
    f"The light source is a {scenario.light_sources.light_source_list[indices.first_light_source].__class__.__name__}."
)
print(
    f"The target areas have the following index mapping: {scenario.solar_tower.target_name_to_index}."
)
print(
    f"The first heliostat in the first group in the field is {scenario.heliostat_field.heliostat_groups[indices.first_heliostat_group].names[indices.first_heliostat]}."
)
print(
    f"The location of {scenario.heliostat_field.heliostat_groups[indices.first_heliostat_group].names[indices.first_heliostat]} is: {scenario.heliostat_field.heliostat_groups[indices.first_heliostat_group].positions[indices.first_heliostat].tolist()}."
)

# We only consider one heliostat for the beginning.
# There is only one heliostat in the scenario. That is why the active_heliostat_mask has only one element.
# To activate a heliostat once, you write a 1 at the index of the heliostat you want to activate.
# In our case we write a 1 at index 0. To activate this heliostat twice (this would duplicate the heliostat) you would write a 2 at index 0.
active_heliostats_mask = torch.tensor([1], dtype=torch.int32, device=device)

# Activate the heliostat. Only activated heliostats will be aligned or ray-traced.
scenario.heliostat_field.heliostat_groups[
    indices.first_heliostat_group
].activate_heliostats(
    active_heliostats_mask=active_heliostats_mask,
    device=device,
)

# Each heliostat has an aim point. We choose an aim point on one of the target areas.
# Select the first target area as the designated target for this heliostat.
target_area_indices = torch.tensor([0], device=device)

# Use the center of the selected target area as the aim point.
aim_point = scenario.solar_tower.get_centers_of_target_areas(
    target_area_indices=target_area_indices, device=device
)
print(f"The initial aim point used for this raytracing is {aim_point.tolist()}.")

# Since we only have one heliostat we need to define a single incident ray direction.
# When the sun is directly in the south, the rays point directly to the north.
# Incident ray directions need to be normalized.
incident_ray_directions = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device)

# Save the original surface points of the one active heliostat.
original_surface_points = scenario.heliostat_field.heliostat_groups[0].surface_points

# Align the heliostat(s).
scenario.heliostat_field.heliostat_groups[
    indices.first_heliostat_group
].align_surfaces_with_incident_ray_directions(
    aim_points=aim_point,
    incident_ray_directions=incident_ray_directions,
    active_heliostats_mask=active_heliostats_mask,
    device=device,
)

# Save the aligned surface points of the one active heliostat.
# The original surface points are saved for all heliostats, active or not.
# The aligned surface points are saved only for the active/current/aligned heliostats.
# That is why we do not need to select specific indices here.
aligned_surface_points = scenario.heliostat_field.heliostat_groups[
    indices.first_heliostat_group
].active_surface_points

# Let's plot the original and the aligned surface points.
# Define colors for each facet of the heliostat.
colors = ["r", "g", "b", "y"]

# Create a 3D plot.
fig = plt.figure(figsize=(14, 6))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)

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
    e_origin = (
        original_surface_points[indices.first_heliostat, start:end, indices.e]
        .cpu()
        .detach()
        .numpy()
    )
    n_origin = (
        original_surface_points[indices.first_heliostat, start:end, indices.n]
        .cpu()
        .detach()
        .numpy()
    )
    u_origin = (
        original_surface_points[indices.first_heliostat, start:end, indices.u]
        .cpu()
        .detach()
        .numpy()
    )
    e_aligned = (
        aligned_surface_points[indices.first_heliostat, start:end, indices.e]
        .cpu()
        .detach()
        .numpy()
    )
    n_aligned = (
        aligned_surface_points[indices.first_heliostat, start:end, indices.n]
        .cpu()
        .detach()
        .numpy()
    )
    u_aligned = (
        aligned_surface_points[indices.first_heliostat, start:end, indices.u]
        .cpu()
        .detach()
        .numpy()
    )
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
plt.savefig("tut_1.png")


# Create a ray tracer.
ray_tracer = HeliostatRayTracer(
    scenario=scenario,
    heliostat_group=scenario.heliostat_field.heliostat_groups[
        indices.first_heliostat_group
    ],
)

# Perform heliostat-based ray tracing.
(
    image_south,
    _,
    _,
    _,
) = ray_tracer.trace_rays(
    incident_ray_directions=incident_ray_directions,
    active_heliostats_mask=active_heliostats_mask,
    target_area_indices=target_area_indices,
    device=device,
)

# Plot the result.
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(image_south[0].cpu().detach().numpy(), cmap="inferno")
tight_layout()
plt.savefig("tut_2.png")


# Define helper functions to enable us to repeat the process!
def align_and_trace_rays(
    light_direction: torch.Tensor,
    active_heliostats_mask: torch.Tensor,
    target_area_indices: torch.Tensor,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """
    Align the heliostat and perform heliostat ray tracing.

    Parameters
    ----------
    light_direction : torch.Tensor
        Direction of the incoming light on the heliostat.
    active_heliostats_mask : torch.Tensor
        A mask for the active heliostats.
    target_area_indices : torch.Tensor
        Indices of the target areas for each active heliostat.
    device : torch.device | str
        The device on which to initialize tensors (default is "cuda").

    Returns
    -------
    torch.Tensor
        Flux density distribution bitmaps per heliostat on the receiver.
    """
    # Activate heliostats.
    scenario.heliostat_field.heliostat_groups[
        indices.first_heliostat_group
    ].activate_heliostats(
        active_heliostats_mask=active_heliostats_mask,
        device=device,
    )

    # Align all heliostats.
    scenario.heliostat_field.heliostat_groups[
        indices.first_heliostat_group
    ].align_surfaces_with_incident_ray_directions(
        aim_points=scenario.solar_tower.get_centers_of_target_areas(
            target_area_indices=target_area_indices, device=device
        ),
        incident_ray_directions=light_direction,
        active_heliostats_mask=active_heliostats_mask,
        device=device,
    )

    # Perform heliostat-based ray tracing.
    (
        bitmaps,
        _,
        _,
        _,
    ) = ray_tracer.trace_rays(
        incident_ray_directions=light_direction,
        active_heliostats_mask=active_heliostats_mask,
        target_area_indices=target_area_indices,
        device=device,
    )
    return bitmaps


def plot_multiple_images(
    *image_tensors: torch.Tensor, names: list[str] | None = None
) -> None:
    """
    Plot multiple receiver ray tracing images in a grid.

    This function is flexible and able to plot an arbitrary number of images depending on the number of image tensors
    provided. Note that the list of names must be the same length as the number of provided images, otherwise the images
    will be untitled.

    Parameters
    ----------
    *image_tensors : torch.Tensor
        An arbitrary number of image tensors to be plotted.
    names : list[str] | None, optional
        Names of the images to be plotted (default is None).
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
    plt.savefig("tut_3.png")


# Consider multiple incident ray directions and plot the result.
# Define light directions.
incident_ray_direction_east = torch.tensor([[-1.0, 0.0, 0.0, 0.0]], device=device)
incident_ray_direction_west = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
incident_ray_direction_above = torch.tensor([[0.0, 0.0, -1.0, 0.0]], device=device)

# Perform alignment and ray tracing to generate flux density images.
image_east = align_and_trace_rays(
    light_direction=incident_ray_direction_east,
    active_heliostats_mask=active_heliostats_mask,
    target_area_indices=target_area_indices,
    device=device,
)
image_west = align_and_trace_rays(
    light_direction=incident_ray_direction_west,
    active_heliostats_mask=active_heliostats_mask,
    target_area_indices=target_area_indices,
    device=device,
)
image_above = align_and_trace_rays(
    light_direction=incident_ray_direction_above,
    active_heliostats_mask=active_heliostats_mask,
    target_area_indices=target_area_indices,
    device=device,
)

# Plot the resulting images.
plot_multiple_images(
    image_south[indices.first_heliostat],
    image_east[indices.first_heliostat],
    image_west[indices.first_heliostat],
    image_above[indices.first_heliostat],
    names=["South", "East", "West", "Above"],
)
