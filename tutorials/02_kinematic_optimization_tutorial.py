import pathlib

import h5py
import torch

from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.core.kinematic_optimizer import KinematicOptimizer
from artist.data_loader import paint_loader
from artist.scenario.scenario import Scenario
from artist.util import set_logger_config
from artist.util.environment_setup import get_device, setup_distributed_environment

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set up logger
set_logger_config()

# Set the device
device = get_device()

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path(
    "/workVERLEIHNIX/mb/ARTIST/tutorials/data/scenarios/test_scenario_paint_multiple_heliostat_groups.h5"
)
number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
    scenario_path=scenario_path
)

with setup_distributed_environment(
    number_of_heliostat_groups=number_of_heliostat_groups,
    device=device,
) as (
    device,
    is_distributed,
    is_nested,
    rank,
    world_size,
    process_subgroup,
    groups_to_ranks_mapping,
    heliostat_group_rank,
    heliostat_group_world_size,
):
    # Also specify the heliostats to be calibrated and the paths to your calibration-properties.json files.
    # Please follow the following style: list[tuple[str, list[pathlib.Path]]]
    heliostat_data_mapping = [
        (
            "AA39",
            [
                pathlib.Path(
                    "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/202558-calibration-properties.json"
                ),
                pathlib.Path(
                    "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/209075-calibration-properties.json"
                ),
                pathlib.Path(
                    "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/218385-calibration-properties.json"
                ),
                pathlib.Path(
                    "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/246955-calibration-properties.json"
                ),
                pathlib.Path(
                    "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/275564-calibration-properties.json"
                ),
            ],
            [
                pathlib.Path(
                    "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/202558-flux-centered.png"
                ),
                pathlib.Path(
                    "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/209075-flux-centered.png"
                ),
                pathlib.Path(
                    "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/218385-flux-centered.png"
                ),
                pathlib.Path(
                    "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/246955-flux-centered.png"
                ),
                # pathlib.Path(
                #     "please/insert/the/path/to/the/paint/data/here/calibration-properties.json"
                # ),
                # ....
            ],
        ),
        (
            "AA31",
            [
                pathlib.Path(
                    "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA31/125284-calibration-properties.json"
                ),
                pathlib.Path(
                    "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA31/126372-calibration-properties.json"
                ),
                pathlib.Path(
                    "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA31/219988-calibration-properties.json"
                ),
            ],
            [
                pathlib.Path(
                    "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA31/125284-flux-centered.png"
                ),
                pathlib.Path(
                    "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA31/126372-flux-centered.png"
                ),
                pathlib.Path(
                    "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA31/219988-flux-centered.png"
                ),
                # pathlib.Path(
                #      "please/insert/the/path/to/the/paint/data/here/calibration-properties.json"
                # ),
            ],
        ),
        # ...
    ]

    # Load the scenario.
    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file, device=device
        )

    for heliostat_group_index, heliostat_group in enumerate(
        scenario.heliostat_field.heliostat_groups
    ):
        # Load the calibration data.
        (
            focal_spots_calibration,
            incident_ray_directions_calibration,
            motor_positions_calibration,
            heliostats_mask_calibration,
            target_area_mask_calibration,
        ) = paint_loader.extract_paint_calibration_properties_data(
            heliostat_calibration_mapping=[
                (heliostat_name, calibration_properties_paths)
                for heliostat_name, calibration_properties_paths, _ in heliostat_data_mapping
                if heliostat_name in heliostat_group.names
            ],
            heliostat_names=heliostat_group.names,
            target_area_names=scenario.target_areas.names,
            power_plant_position=scenario.power_plant_position,
            device=device,
        )
        if heliostats_mask_calibration.sum() > 0:
            # Set up optimizer and scheduler.
            tolerance = 0.0005
            max_epoch = 1000
            initial_learning_rate = 0.0001

            use_ray_tracing = True
            if use_ray_tracing:
                motor_positions_calibration = None
                tolerance = 0.035
                max_epoch = 600
                initial_learning_rate = 0.0004

            optimizer = torch.optim.Adam(
                heliostat_group.kinematic.parameters(), lr=initial_learning_rate
            )

            # Create the kinematic optimizer.
            kinematic_optimizer = KinematicOptimizer(
                scenario=scenario,
                heliostat_group=heliostat_group,
                optimizer=optimizer,
            )

            # Calibrate the kinematic.
            kinematic_optimizer.optimize(
                focal_spots_calibration=focal_spots_calibration,
                incident_ray_directions=incident_ray_directions_calibration,
                active_heliostats_mask=heliostats_mask_calibration,
                target_area_mask_calibration=target_area_mask_calibration,
                motor_positions_calibration=motor_positions_calibration,
                tolerance=tolerance,
                max_epoch=max_epoch,
                num_log=max_epoch,
                device=device,
            )

    heliostat_target_light_source_mapping = [
        (
            "AA31",
            "solar_tower_juelich_upper",
            torch.tensor([0.6083, 0.2826, -0.7417, 0.0000], device=device),
        ),
        (
            "AA31",
            "solar_tower_juelich_lower",
            torch.tensor([-0.5491, 0.3197, -0.7722, 0.0000], device=device),
        ),
        (
            "AA31",
            "solar_tower_juelich_lower",
            torch.tensor([-0.6905, 0.2223, -0.6884, 0.0000], device=device),
        ),
        (
            "AA39",
            "multi_focus_tower",
            torch.tensor([-0.6568, 0.3541, -0.6658, 0.0000], device=device),
        ),
        (
            "AA39",
            "solar_tower_juelich_lower",
            torch.tensor([-0.0947, 0.4929, -0.8649, 0.0000], device=device),
        ),
        (
            "AA39",
            "multi_focus_tower",
            torch.tensor([0.0619, 0.4641, -0.8836, 0.0000], device=device),
        ),
        (
            "AA39",
            "solar_tower_juelich_lower",
            torch.tensor([-0.5211, 0.5949, -0.6121, 0.0000], device=device),
        ),
        (
            "AA39",
            "multi_focus_tower",
            torch.tensor([-0.2741, 0.4399, -0.8552, 0.0000], device=device),
        ),
    ]

    bitmap_resolution_e = 256
    bitmap_resolution_u = 256
    combined_bitmaps_per_target = torch.zeros(
        (
            scenario.target_areas.number_of_target_areas,
            bitmap_resolution_e,
            bitmap_resolution_u,
        ),
        device=device,
    )

    for group_index in groups_to_ranks_mapping[rank]:
        heliostat_group = scenario.heliostat_field.heliostat_groups[group_index]

        # If no mapping from heliostats to target areas to incident ray direction is provided, the scenario.index_mapping() method
        # activates all heliostats. It is possible to then provide a default target area index and a default incident ray direction
        # if those are not specified either all heliostats are assigned to the first target area found in the scenario with an
        # incident ray direction "north" (meaning the light source position is directly in the south) for all heliostats.
        (
            active_heliostats_mask,
            target_area_mask,
            incident_ray_directions,
        ) = scenario.index_mapping(
            heliostat_group=heliostat_group,
            string_mapping=heliostat_target_light_source_mapping,
            device=device,
        )

        # Align heliostats.
        heliostat_group.align_surfaces_with_incident_ray_directions(
            aim_points=scenario.target_areas.centers[target_area_mask],
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_heliostats_mask,
            device=device,
        )

        # Create a ray tracer.
        ray_tracer = HeliostatRayTracer(
            scenario=scenario,
            heliostat_group=heliostat_group,
            world_size=heliostat_group_world_size,
            rank=heliostat_group_rank,
            batch_size=4,
            random_seed=heliostat_group_rank,
            bitmap_resolution_e=bitmap_resolution_e,
            bitmap_resolution_u=bitmap_resolution_u,
        )

        # Perform heliostat-based ray tracing.
        bitmaps_per_heliostat = ray_tracer.trace_rays(
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_heliostats_mask,
            target_area_mask=target_area_mask,
            device=device,
        )

        import matplotlib.pyplot as plt

        for i in range(bitmaps_per_heliostat.shape[0]):
            fig, ax = plt.subplots()
            ax.imshow(bitmaps_per_heliostat[i].cpu().detach(), cmap="gray")
            ax.axis("off")
            ax.margins(0)
            ax.set_position([0, 0, 1, 1])
            plt.savefig(f"heliostat_{i}_artist.png", bbox_inches="tight", pad_inches=0)

        torch.save(bitmaps_per_heliostat.cpu().detach(), "heliostat_bitmaps.pt")

        bitmaps_per_target = ray_tracer.get_bitmaps_per_target(
            bitmaps_per_heliostat=bitmaps_per_heliostat,
            target_area_mask=target_area_mask,
            device=device,
        )

        combined_bitmaps_per_target = combined_bitmaps_per_target + bitmaps_per_target

    if is_nested:
        torch.distributed.all_reduce(
            combined_bitmaps_per_target,
            op=torch.distributed.ReduceOp.SUM,
            group=process_subgroup,
        )

    if is_distributed:
        torch.distributed.all_reduce(
            combined_bitmaps_per_target, op=torch.distributed.ReduceOp.SUM
        )
