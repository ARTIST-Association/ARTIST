import pathlib

import h5py
import torch

from artist.util import paint_loader, set_logger_config
from artist.util.kinematic_optimizer import KinematicOptimizer
from artist.util.scenario import Scenario

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set up logger
set_logger_config()

# Set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path("tutorials/data/scenarios/test_scenario_paint_four_heliostats.h5")

# Also specify the heliostats to be calibrated and the paths to your calibration-properties.json files.
heliostat_calibration_mapping = [
    ("AA39", 
     [pathlib.Path("/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/202558-calibration-properties.json"),
      pathlib.Path("/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/209075-calibration-properties.json"),
      pathlib.Path("/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/218385-calibration-properties.json"),
      pathlib.Path("/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/275564-calibration-properties.json"),
     ]
    ),
    ("AA31", 
     [pathlib.Path("/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA31/125284-calibration-properties.json"),
      pathlib.Path("/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA31/126372-calibration-properties.json"),
      pathlib.Path("/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA31/219988-calibration-properties.json"),
     ]
    ),
]

# Load the scenario.
with h5py.File(scenario_path, "r") as scenario_file:
    scenario = Scenario.load_scenario_from_hdf5(
        scenario_file=scenario_file, device=device
    )

for heliostat_group_index, heliostat_group in enumerate(scenario.heliostat_field.heliostat_groups):

    # Load the calibration data.
    (
        calibration_heliostat_names,
        calibration_target_names,
        centers_calibration_images,
        sun_positions,
        all_calibration_motor_positions,
    ) = paint_loader.extract_paint_calibration_data(
        heliostat_calibration_mapping=[(heliostat_name, paths) for heliostat_name, paths in heliostat_calibration_mapping if heliostat_name in heliostat_group.names],
        scenario=scenario,
        device=device,
    )
    heliostat_index_map = {name: index for index, name in enumerate(heliostat_group.names)}
    heliostat_indices = [heliostat_index_map[name] for name in calibration_heliostat_names]
    target_area_index_map = {target_area_name: index for index, target_area_name in enumerate(scenario.target_areas.names)}
    target_area_indices = torch.tensor([target_area_index_map[target_area_name] for target_area_name in calibration_target_names])

    # create calibration group
    heliostat_group_class = type(heliostat_group)
    calibration_group = heliostat_group_class(
        names=calibration_heliostat_names,
        positions=heliostat_group.positions[heliostat_indices],
        aim_points=scenario.target_areas.centers[target_area_indices],
        surface_points=heliostat_group.surface_points[heliostat_indices],
        surface_normals=heliostat_group.surface_normals[heliostat_indices],
        initial_orientations=heliostat_group.initial_orientations[heliostat_indices],
        kinematic_deviation_parameters=heliostat_group.kinematic_deviation_parameters[heliostat_indices],
        actuator_parameters=heliostat_group.actuator_parameters[heliostat_indices],
        device=device
    )

    # The incident ray direction needs to be normed.
    incident_ray_directions = (
        torch.tensor([0.0, 0.0, 0.0, 1.0], device=device) - sun_positions
    )

    # Select the kinematic parameters to be optimized and calibrated.
    optimizable_parameters = [
        calibration_group.kinematic_deviation_parameters.requires_grad_(),
        calibration_group.actuator_parameters.requires_grad_(),
    ]

    # Set up optimizer and scheduler.
    tolerance = 0.0005
    max_epoch = 1000
    initial_learning_rate = 0.0001
    learning_rate_factor = 0.1
    learning_rate_patience = 20
    learning_rate_threshold = 0.1

    use_ray_tracing = False
    if use_ray_tracing:
        all_calibration_motor_positions = None
        tolerance = 0.035
        max_epoch = 10000
        initial_learning_rate = 0.002
        learning_rate_factor = 0.1
        learning_rate_patience = 50
        learning_rate_threshold = 50

    optimizer = torch.optim.Adam(optimizable_parameters, lr=initial_learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=learning_rate_factor,
        patience=learning_rate_patience,
        threshold=learning_rate_threshold,
        threshold_mode="abs",
    )

    # Create the kinematic optimizer.
    kinematic_optimizer = KinematicOptimizer(
        scenario=scenario,
        calibration_group=calibration_group,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # Calibrate the kinematic.
    calibrated_kinematic_deviation_parameters, calibrated_actuator_parameters = kinematic_optimizer.optimize(
        tolerance=tolerance,
        max_epoch=max_epoch,
        centers_calibration_images=centers_calibration_images,
        incident_ray_directions=incident_ray_directions,
        target_area_indices=target_area_indices,
        motor_positions=all_calibration_motor_positions,
        num_log=max_epoch,
        device=device,
    )



