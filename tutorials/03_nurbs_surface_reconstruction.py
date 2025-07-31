import pathlib

import h5py
import torch

from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.core.surface_reconstructor import SurfaceReconstructor
from artist.scenario.scenario import Scenario
from artist.util import set_logger_config, utils
from artist.util.environment_setup import get_device, setup_distributed_environment
from artist.util.nurbs import NURBSSurfaces
from tutorials import helper

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set up logger
set_logger_config()

# Set the device
device = get_device()

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path(
    "/workVERLEIHNIX/mb/ARTIST/tutorials/data/scenarios/test_scenario_paint_multiple_heliostat_groups_ideal_6_cp.h5"
)

# Also specify the heliostats to be calibrated and the paths to your measured flux density distributions.
# Please follow the following style: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
heliostat_data_mapping = [
    (
        "AA39",
        [
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/86520-calibration-properties.json"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/151375-calibration-properties.json"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/202558-calibration-properties.json"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/205363-calibration-properties.json"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/209075-calibration-properties.json"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/218385-calibration-properties.json"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/223788-calibration-properties.json"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/246955-calibration-properties.json"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/270398-calibration-properties.json"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/271633-calibration-properties.json"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/275564-calibration-properties.json"
            ),
            # ....
        ],
        [
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/86520-flux-centered.png"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/151375-flux-centered.png"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/202558-flux-centered.png"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/205363-flux-centered.png"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/209075-flux-centered.png"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/218385-flux-centered.png"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/223788-flux-centered.png"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/246955-flux-centered.png"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/270398-flux-centered.png"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/271633-flux-centered.png"
            ),
            pathlib.Path(
                "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/275564-flux-centered.png"
            ),
            # ....
        ],
    ),
]

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
    # Load the scenario.
    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file, device=device
        )

    for heliostat_group_index in groups_to_ranks_mapping[rank]:
        # If there are more ranks than heliostat groups, some ranks will be left idle.
        if rank < scenario.heliostat_field.number_of_heliostat_groups:
            heliostat_group = scenario.heliostat_field.heliostat_groups[
                heliostat_group_index
            ]

            # Set parameters.
            scenario.light_sources.light_source_list[0].number_of_rays = 200
            tolerance = 0.00005
            max_epoch = 4000
            initial_learning_rate = 1e-6
            number_of_surface_points = torch.tensor([80, 80], device=device)
            resolution = torch.tensor([256, 256], device=device)

            # Create the surface reconstructor.
            surface_reconstructor = SurfaceReconstructor(
                scenario=scenario,
                heliostat_group=heliostat_group,
                heliostat_data_mapping=heliostat_data_mapping,
                number_of_surface_points=number_of_surface_points,
                resolution=resolution,
                initial_learning_rate=initial_learning_rate,
                tolerance=tolerance,
                max_epoch=max_epoch,
                num_log=max_epoch,
                device=device,
            )

            surface_reconstructor.reconstruct_surfaces(
                heliostat_group_index, device=device
            )

            #### TODO delete everything from here on
            if heliostat_group_index == 1:
                paths = [
                    pathlib.Path(
                        "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/86520-calibration-properties.json"
                    ),
                    pathlib.Path(
                        "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/151375-calibration-properties.json"
                    ),
                    pathlib.Path(
                        "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/202558-calibration-properties.json"
                    ),
                    pathlib.Path(
                        "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/205363-calibration-properties.json"
                    ),
                    pathlib.Path(
                        "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/209075-calibration-properties.json"
                    ),
                    pathlib.Path(
                        "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/218385-calibration-properties.json"
                    ),
                    pathlib.Path(
                        "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/223788-calibration-properties.json"
                    ),
                    pathlib.Path(
                        "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/246955-calibration-properties.json"
                    ),
                    pathlib.Path(
                        "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/270398-calibration-properties.json"
                    ),
                    pathlib.Path(
                        "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/271633-calibration-properties.json"
                    ),
                    pathlib.Path(
                        "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/275564-calibration-properties.json"
                    ),
                ]

                incident_ray_directions, targets = (
                    helper.calibration_path_to_sun_and_tower(
                        paths=paths,
                        target_area_names=scenario.target_areas.names,
                        device=device,
                    )
                )

                # If you want to customize the mapping, choose the following style: list[tuple[str, str, torch.Tensor]]
                heliostat_target_light_source_mapping = [
                    ("AA39", targets[0], incident_ray_directions[0]),
                    ("AA39", targets[1], incident_ray_directions[1]),
                    ("AA39", targets[2], incident_ray_directions[2]),
                    ("AA39", targets[3], incident_ray_directions[3]),
                    ("AA39", targets[4], incident_ray_directions[4]),
                    ("AA39", targets[5], incident_ray_directions[5]),
                    ("AA39", targets[6], incident_ray_directions[6]),
                    ("AA39", targets[7], incident_ray_directions[7]),
                    ("AA39", targets[8], incident_ray_directions[8]),
                    ("AA39", targets[9], incident_ray_directions[9]),
                    ("AA39", targets[10], incident_ray_directions[10]),
                    (
                        "AA39",
                        "receiver",
                        torch.tensor([0.0, 1.0, 0.0, 0.0], device=device),
                    ),
                ]

                (
                    active_heliostats_mask,
                    target_area_mask,
                    incident_ray_directions,
                ) = scenario.index_mapping(
                    heliostat_group=heliostat_group,
                    string_mapping=heliostat_target_light_source_mapping,
                    device=device,
                )

                heliostat_group.activate_heliostats(
                    active_heliostats_mask=active_heliostats_mask, device=device
                )

                nurbs = NURBSSurfaces(
                    degrees=heliostat_group.nurbs_degrees,
                    control_points=heliostat_group.active_nurbs_control_points,
                    uniform=True,
                    device=device,
                )

                evaluation_points = (
                    utils.create_nurbs_evaluation_grid(
                        number_of_evaluation_points=number_of_surface_points,
                        device=device,
                    )
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(
                        len(heliostat_target_light_source_mapping),
                        4,
                        -1,
                        -1,
                    )
                )

                calc_points, calc_normals = nurbs.calculate_surface_points_and_normals(
                    evaluation_points=evaluation_points, device=device
                )

                heliostat_group.active_surface_points = calc_points.reshape(
                    len(heliostat_target_light_source_mapping), -1, 4
                )
                heliostat_group.active_surface_normals = calc_normals.reshape(
                    len(heliostat_target_light_source_mapping), -1, 4
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
                    batch_size=20,
                    random_seed=heliostat_group_rank,
                    bitmap_resolution=torch.tensor([256, 256], device=device),
                )

                # Perform heliostat-based ray tracing.
                bitmaps_per_heliostat = ray_tracer.trace_rays(
                    incident_ray_directions=incident_ray_directions,
                    active_heliostats_mask=active_heliostats_mask,
                    target_area_mask=target_area_mask,
                    device=device,
                )

                import matplotlib.pyplot as plt

                plt.clf()
                name = f"reconstructed_group_{heliostat_group_index}_cp_{heliostat_group.active_nurbs_control_points.shape[2]}_{max_epoch}"
                helper.plot_multiple_fluxes(
                    bitmaps_per_heliostat,
                    torch.zeros_like(bitmaps_per_heliostat),
                    name=name,
                )
