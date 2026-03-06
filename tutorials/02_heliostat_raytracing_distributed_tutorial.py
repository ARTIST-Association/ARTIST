import pathlib

import h5py
import torch
from matplotlib import pyplot as plt

from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.field.heliostat_group import HeliostatGroup
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, index_mapping, set_logger_config
from artist.util.environment_setup import get_device, setup_distributed_environment

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set up logger.
set_logger_config()

# Set device type.
device = get_device()

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path("please/insert/the/path/to/the/scenario/here/scenario.h5")

# Set the number of heliostat groups, this is needed for process group assignment.
number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
    scenario_path=scenario_path
)

with setup_distributed_environment(
    number_of_heliostat_groups=number_of_heliostat_groups,
    device=device,
) as ddp_setup:
    device = ddp_setup[config_dictionary.device]

    # Load the scenario.
    with h5py.File(scenario_path) as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file,
            device=device,
        )

    # Set a ray extinction factor responsible for global shading of rays.
    # 0.0 -> no global shading, 1.0 -> full global shading
    ray_extinction_factor = 0.0

    # Use a heliostat-target-light source mapping to specify
    # - which heliostat in your scenario should be activated,
    # - which incident ray direction each heliostat should receive for alignment, and
    # - on which target each heliostat will be ray-traced.
    # If no mapping is provided, all heliostats are selected. They will all receive the default incident ray direction
    # from a sun located directly in the south and be ray-traced on the first target found in the scenario.
    heliostat_target_light_source_mapping = None
    # If you want to customize the mapping, choose the following style: list[tuple[str, str, torch.Tensor]]
    # heliostat_target_light_source_mapping = [
    #     ("heliostat_1", "target_name_2", incident_ray_direction_tensor_1),
    #     ("heliostat_2", "target_name_2", incident_ray_direction_tensor_2),
    #     (...)
    # ]

    bitmap_resolution = torch.tensor([256, 256])

    combined_bitmaps_per_target = torch.zeros(
        (
            scenario.target_areas.number_of_target_areas,
            bitmap_resolution[index_mapping.unbatched_bitmap_e],
            bitmap_resolution[index_mapping.unbatched_bitmap_u],
        ),
        device=device,
    )

    # Since each heliostat group has its own kinematics and actuator types, the groups must be processed separately.
    # If a distributed environment exists, they can be processed in parallel; otherwise, the results for each heliostat
    # group are computed sequentially.
    # For blocking to work correctly, all heliostat groups have to be aligned before any group can be ray-traced.
    for heliostat_group_alignment in scenario.heliostat_field.heliostat_groups:
        # If no mapping from heliostats to target areas and incident ray direction is provided, the
        # ``scenario.index_mapping()`` method activates all heliostats. A default target area index and a default
        # incident ray direction can then be specified. If these are not provided either, all heliostats are assigned
        # to the first target area found in the scenario and receive an incident ray direction "north" (meaning the
        # light source position is directly in the south).
        (
            active_heliostats_mask,
            target_area_indices,
            incident_ray_directions,
        ) = scenario.index_mapping(
            heliostat_group=heliostat_group_alignment,
            string_mapping=heliostat_target_light_source_mapping,
            device=device,
        )

        # The ``active_heliostats_mask`` is a tensor that indicates which heliostats are active.
        # For each index, 0 indicates a deactivated heliostat and 1 an activated one.
        # An integer greater than 1 indicates that the heliostat in this index is considered multiple times.
        # The tensor has the shape [number_of_heliostats_in_group].
        heliostat_group_alignment.activate_heliostats(
            active_heliostats_mask=active_heliostats_mask, device=device
        )

        # Align heliostats.
        heliostat_group_alignment.align_surfaces_with_incident_ray_directions(
            aim_points=scenario.target_areas.centers[target_area_indices],
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_heliostats_mask,
            device=device,
        )

    # Raytracing happens only on one device for each group.
    for heliostat_group_index in ddp_setup[config_dictionary.groups_to_ranks_mapping][
        ddp_setup[config_dictionary.rank]
    ]:
        heliostat_group: HeliostatGroup = scenario.heliostat_field.heliostat_groups[
            heliostat_group_index
        ]
        if heliostat_group.active_heliostats_mask.sum() > 0:
            (
                active_heliostats_mask,
                target_area_indices,
                incident_ray_directions,
            ) = scenario.index_mapping(
                heliostat_group=heliostat_group,
                string_mapping=heliostat_target_light_source_mapping,
                device=device,
            )

            # Create a parallelized ray tracer.
            ray_tracer = HeliostatRayTracer(
                scenario=scenario,
                heliostat_group=heliostat_group,
                blocking_active=False,
                world_size=ddp_setup[config_dictionary.heliostat_group_world_size],
                rank=ddp_setup[config_dictionary.heliostat_group_rank],
                batch_size=heliostat_group.number_of_active_heliostats,
                random_seed=ddp_setup[config_dictionary.heliostat_group_rank],
                bitmap_resolution=bitmap_resolution,
            )

            # Perform heliostat-based ray tracing.
            bitmaps_per_heliostat = ray_tracer.trace_rays(
                incident_ray_directions=incident_ray_directions,
                active_heliostats_mask=active_heliostats_mask,
                target_area_indices=target_area_indices,
                device=device,
            )

            sample_indices_for_local_rank = ray_tracer.get_sampler_indices()
            # Plot the bitmaps of each single heliostat.
            for i in range(bitmaps_per_heliostat.shape[0]):
                expanded_names = [
                    name
                    for name, m in zip(heliostat_group.names, active_heliostats_mask)
                    for _ in range(m)
                ]
                plt.imshow(bitmaps_per_heliostat[i].cpu().detach(), cmap="gray")
                plt.axis("off")
                plt.title(
                    f"Heliostat: {expanded_names[sample_indices_for_local_rank[i]]}, Group: {heliostat_group_index}, Rank: {ddp_setup['rank']} Target: {scenario.target_areas.names[target_area_indices[i]]}"
                )
                plt.savefig(
                    f"bitmap_group_{heliostat_group_index}_on_rank_{ddp_setup['rank']}_sample_{i}_heliostat_{expanded_names[sample_indices_for_local_rank[i]]}.png"
                )

            # Get the flux distributions per target.
            bitmaps_per_target = ray_tracer.get_bitmaps_per_target(
                bitmaps_per_heliostat=bitmaps_per_heliostat,
                target_area_indices=target_area_indices[sample_indices_for_local_rank],
                device=device,
            )

            combined_bitmaps_per_target = (
                combined_bitmaps_per_target + bitmaps_per_target
            )

    # This nested reduction step could be skipped, since the reduction within the outer process group would handle it.
    # However, performing it here allows us to inspect the intermediate reduction results of the nested process group.
    if ddp_setup[config_dictionary.is_nested]:
        torch.distributed.all_reduce(
            combined_bitmaps_per_target,
            op=torch.distributed.ReduceOp.SUM,
            group=ddp_setup[config_dictionary.process_subgroup],
        )

        # Plot the combined bitmaps of heliostats on the same target reduced within each group.
        for target_area_index in range(scenario.target_areas.number_of_target_areas):
            plt.imshow(
                combined_bitmaps_per_target[target_area_index].cpu().detach(),
                cmap="gray",
            )
            plt.axis("off")
            plt.title(
                f"Reduced within group, Target area: {scenario.target_areas.names[target_area_index]}, Rank: {ddp_setup['rank']}"
            )
            plt.savefig(
                f"reduced_bitmap_on_rank_{ddp_setup['rank']}_on_{scenario.target_areas.names[target_area_index]}.png"
            )

    if ddp_setup[config_dictionary.is_distributed]:
        torch.distributed.all_reduce(
            combined_bitmaps_per_target, op=torch.distributed.ReduceOp.SUM
        )

    # Plot the final combined bitmaps of heliostats on the same target fully reduced.
    for target_area_index in range(scenario.target_areas.number_of_target_areas):
        plt.imshow(
            combined_bitmaps_per_target[target_area_index].cpu().detach(),
            cmap="gray",
        )
        plt.axis("off")
        plt.title(
            f"Final bitmap, Target area: {scenario.target_areas.names[target_area_index]}, Rank: {ddp_setup['rank']}"
        )
        plt.savefig(
            f"final_reduced_bitmap_on_rank_{ddp_setup['rank']}_on_{scenario.target_areas.names[target_area_index]}.png"
        )
