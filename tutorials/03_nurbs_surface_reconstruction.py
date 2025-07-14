import pathlib

import h5py
import torch

from artist.core.surface_reconstructor import SurfaceReconstructor
from artist.data_loader import flux_distribution_loader, paint_loader
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

# Also specify the heliostats to be calibrated and the paths to your measured flux density distributions.
# Please follow the following style: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
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
            )
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
    
    for heliostat_group_index, heliostat_group in enumerate(
        scenario.heliostat_field.heliostat_groups
    ):
        # Load the measured flux density distribution data.
        measured_flux_density_distributions = flux_distribution_loader.load_flux_from_png(
            heliostat_flux_path_mapping=[
                (heliostat_name, png_paths)
                for heliostat_name, _, png_paths in heliostat_data_mapping
                if heliostat_name in heliostat_group.names
            ],
        heliostat_names=heliostat_group.names,
        device=device
        )
        (
            _,
            incident_ray_directions_reconstruction,
            _,
            heliostats_mask_reconstruction,
            target_area_mask_reconstruction,
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

        if heliostats_mask_reconstruction.sum() > 0:
            tolerance = 0.0005
            max_epoch = 1000
            initial_learning_rate = 0.0001

            # Create the surface reconstructor.
            surface_reconstructor = SurfaceReconstructor(
                scenario=scenario,
                heliostat_group=heliostat_group,
            )

            surface_reconstructor.reconstruct_surfaces(
                flux_distributions_measured=measured_flux_density_distributions,
                number_of_evaluation_points=torch.tensor([50, 50], device=device),
                incident_ray_directions=incident_ray_directions_reconstruction,
                active_heliostats_mask=heliostats_mask_reconstruction,
                target_area_mask=target_area_mask_reconstruction,
                initial_learning_rate=initial_learning_rate,
                tolerance=tolerance,
                max_epoch=max_epoch,
                num_log=max_epoch,
                device=device
            )
