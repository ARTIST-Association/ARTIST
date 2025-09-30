import logging
import pathlib

import h5py
import paint.util.paint_mappings as paint_mappings
import torch
from matplotlib import pyplot as plt

from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.core.kinematic_calibrator import KinematicCalibrator
from artist.core.loss_functions import FocalSpotLoss
from artist.data_parser.calibration_data_parser import CalibrationDataParser
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, set_logger_config
from artist.util.environment_setup import get_device, setup_distributed_environment

torch.manual_seed(7)
torch.cuda.manual_seed(7)

#############################################################################################################
# Define helper functions for the plots.
# Skip to line 112 for the tutorial code.
#############################################################################################################


def create_flux_plots(data_parser: CalibrationDataParser, name: str) -> None:
    """
    Create data to plot the heliostat fluxes.

    Parameters
    ----------
    data_parser : CalibrationDataParser
            The data parser used to load calibration data from files.
    name : str
        The name for the plots.
    """
    for heliostat_group_index, heliostat_group in enumerate(
        scenario.heliostat_field.heliostat_groups
    ):
        (
            _,
            _,
            incident_ray_directions,
            _,
            active_heliostats_mask,
            target_area_mask,
        ) = data_parser.parse_data_for_reconstruction(
            heliostat_data_mapping=heliostat_data_mapping,
            heliostat_group=heliostat_group,
            scenario=scenario,
            bitmap_resolution=torch.tensor([256, 256]),
            device=device,
        )

        # Activate heliostats.
        heliostat_group.activate_heliostats(
            active_heliostats_mask=active_heliostats_mask,
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
            batch_size=heliostat_group.number_of_active_heliostats,
            bitmap_resolution=torch.tensor([256, 256], device=device),
        )

        # Perform heliostat-based ray tracing.
        bitmaps_per_heliostat = ray_tracer.trace_rays(
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_heliostats_mask,
            target_area_mask=target_area_mask,
            device=device,
        )

        # Create the plots.
        for heliostat_index in range(heliostat_group.number_of_active_heliostats):
            repeated_names = [
                s
                for s, n in zip(heliostat_group.names, active_heliostats_mask.tolist())
                for _ in range(n)
            ]
            plt.imshow(
                bitmaps_per_heliostat[heliostat_index].cpu().detach(), cmap="gray"
            )
            plt.title(f"Heliostat {repeated_names[heliostat_index]} {name} calibration")
            plt.axis("off")
            plt.savefig(
                f"heliostat_{repeated_names[heliostat_index]}_{name}_calibration.png"
            )


#############################################################################################################
# Tutorial
#############################################################################################################

# Set up logger.
set_logger_config()
log = logging.getLogger(__name__)

# Set the device.
device = get_device()

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path("please/insert/the/path/to/the/scenario/here/scenario.h5")

# Also specify the heliostats to be calibrated and the paths to your calibration-properties.json files.
# Please use the following style: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
heliostat_data_mapping = [
    (
        "heliostat_name_1",
        [
            pathlib.Path(
                "please/insert/the/path/to/the/paint/data/here/calibration-properties.json"
            ),
            # ....
        ],
        [
            pathlib.Path("please/insert/the/path/to/the/paint/data/here/flux.png"),
            # ....
        ],
    ),
    (
        "heliostat_name_2",
        [
            pathlib.Path(
                "please/insert/the/path/to/the/paint/data/here/calibration-properties.json"
            ),
            # ....
        ],
        [
            pathlib.Path("please/insert/the/path/to/the/paint/data/here/flux.png"),
            # ....
        ],
    ),
    # ...
]

data_parser = PaintCalibrationDataParser(
    sample_limit=2, centroid_extraction_method=paint_mappings.UTIS_KEY
)

# Create dict for the data source name and the heliostat_data_mapping.
data: dict[
    str,
    CalibrationDataParser | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
] = {
    config_dictionary.data_parser: data_parser,
    config_dictionary.heliostat_data_mapping: heliostat_data_mapping,
}

number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
    scenario_path=scenario_path
)

with setup_distributed_environment(
    number_of_heliostat_groups=number_of_heliostat_groups,
    device=device,
) as ddp_setup:
    device = ddp_setup[config_dictionary.device]

    # Load the scenario.
    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file, device=device
        )

    # Configure the learning rate scheduler. The example scheduler parameter dict includes
    # example parameters for all three possible schedulers.
    scheduler = (
        config_dictionary.exponential
    )  # exponential, cyclic or reduce_on_plateau
    scheduler_parameters = {
        config_dictionary.gamma: 0.9,
        config_dictionary.min: 1e-6,
        config_dictionary.max: 1e-3,
        config_dictionary.step_size_up: 500,
        config_dictionary.reduce_factor: 0.3,
        config_dictionary.patience: 10,
        config_dictionary.threshold: 1e-3,
        config_dictionary.cooldown: 10,
    }

    # Set optimization parameters.
    optimization_configuration = {
        config_dictionary.initial_learning_rate: 0.0005,
        config_dictionary.tolerance: 0.0005,
        config_dictionary.max_epoch: 70,
        config_dictionary.log_step: 3,
        config_dictionary.early_stopping_delta: 1e-4,
        config_dictionary.early_stopping_patience: 200,
        config_dictionary.scheduler: scheduler,
        config_dictionary.scheduler_parameters: scheduler_parameters,
    }

    create_flux_plots(data_parser=data_parser, name="before")

    # Set calibration method.
    kinematic_calibration_method = config_dictionary.kinematic_calibration_raytracing

    # Create the kinematic optimizer.
    kinematic_calibrator = KinematicCalibrator(
        ddp_setup=ddp_setup,
        scenario=scenario,
        data=data,
        optimization_configuration=optimization_configuration,
        calibration_method=kinematic_calibration_method,
    )

    # Uncomment for calibration with raytracing:
    loss_definition = FocalSpotLoss(scenario=scenario)
    # Uncomment for calibration with motor positions.
    # loss_definition = VectorLoss()

    # Calibrate the kinematic.
    final_loss_per_heliostat = kinematic_calibrator.calibrate(
        loss_definition=loss_definition, device=device
    )

# Inspect the synchronized loss per heliostat. Heliostats that have not been optimized have an infinite loss.
print(f"rank {ddp_setup['rank']}, final loss per heliostat {final_loss_per_heliostat}")

create_flux_plots(data_parser=data_parser, name="after")
