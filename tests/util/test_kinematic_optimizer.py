import pathlib

import h5py
import pytest
import torch

from artist import ARTIST_ROOT
from artist.util import config_dictionary, paint_loader, set_logger_config
from artist.util.kinematic_optimizer import KinematicOptimizer
from artist.util.scenario import Scenario

# Set up logger.
set_logger_config()


@pytest.mark.parametrize(
    "optimizer_method, tolerance, max_epoch, initial_lr",
    [
        (
            "use_motor_positions",
            0.0005,
            15,
            0.001,
        ),
        (
            "use_raytracing",
            0.005,
            15,
            0.001,
        ),
    ],
)
def test_kinematic_optimizer(
    optimizer_method: str,
    tolerance: float,
    max_epoch: int,
    initial_lr: float,
    device: torch.device,
) -> None:
    """
    Test the kinematic optimization methods.

    Parameters
    ----------
    optimizer_method : str
        The name of the optimizer method.
    tolerance : float
        Tolerance for the optimizer.
    max_epoch : int
        The maximum amount of epochs for the optimization loop.
    initial_lr : float
        The initial learning rate.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    scenario_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/data/scenarios/test_scenario_paint_multiple_heliostats.h5"
    )

    heliostat_calibration_mapping = [
        (
            "AA39",
            [
                pathlib.Path(ARTIST_ROOT)
                / "tests/data/field_data/AA39-calibration-properties_1.json",
                pathlib.Path(ARTIST_ROOT)
                / "tests/data/field_data/AA39-calibration-properties_2.json",
            ],
        ),
        (
            "AA31",
            [
                pathlib.Path(ARTIST_ROOT)
                / "tests/data/field_data/AA31-calibration-properties_1.json"
            ],
        ),
    ]

    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file, device=device
        )

    for heliostat_group_index, heliostat_group in enumerate(
        scenario.heliostat_field.heliostat_groups
    ):
        # Load the calibration data.
        (
            calibration_heliostat_names,
            calibration_target_names,
            centers_calibration_images,
            sun_positions,
            all_calibration_motor_positions,
        ) = paint_loader.extract_paint_calibration_data(
            heliostat_calibration_mapping=[
                (heliostat_name, paths)
                for heliostat_name, paths in heliostat_calibration_mapping
                if heliostat_name in heliostat_group.names
            ],
            power_plant_position=scenario.power_plant_position,
            device=device,
        )
        heliostat_index_map = {
            name: index for index, name in enumerate(heliostat_group.names)
        }
        heliostat_indices = [
            heliostat_index_map[name] for name in calibration_heliostat_names
        ]
        target_area_index_map = {
            target_area_name: index
            for index, target_area_name in enumerate(scenario.target_areas.names)
        }
        target_area_indices = torch.tensor(
            [
                target_area_index_map[target_area_name]
                for target_area_name in calibration_target_names
            ]
        )

        # create calibration group
        heliostat_group_class = type(heliostat_group)
        calibration_group = heliostat_group_class(
            names=calibration_heliostat_names,
            positions=heliostat_group.positions[heliostat_indices],
            aim_points=scenario.target_areas.centers[target_area_indices],
            surface_points=heliostat_group.surface_points[heliostat_indices],
            surface_normals=heliostat_group.surface_normals[heliostat_indices],
            initial_orientations=heliostat_group.initial_orientations[
                heliostat_indices
            ],
            kinematic_deviation_parameters=heliostat_group.kinematic_deviation_parameters[
                heliostat_indices
            ],
            actuator_parameters=heliostat_group.actuator_parameters[heliostat_indices],
            device=device,
        )

        incident_ray_directions = (
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=device) - sun_positions
        )

        optimizable_parameters = [
            calibration_group.kinematic_deviation_parameters.requires_grad_(),
            calibration_group.actuator_parameters.requires_grad_(),
        ]

        optimizer = torch.optim.Adam(optimizable_parameters, lr=initial_lr)

        # Create alignment optimizer.
        kinematic_optimizer = KinematicOptimizer(
            scenario=scenario,
            calibration_group=calibration_group,
            optimizer=optimizer,
        )

        if optimizer_method == config_dictionary.optimizer_use_raytracing:
            all_calibration_motor_positions = None

        calibrated_kinematic_deviation_parameters, calibrated_actuator_parameters = (
            kinematic_optimizer.optimize(
                tolerance=tolerance,
                max_epoch=max_epoch,
                centers_calibration_images=centers_calibration_images,
                incident_ray_directions=incident_ray_directions,
                target_area_indices=target_area_indices,
                motor_positions=all_calibration_motor_positions,
                num_log=max_epoch,
                device=device,
            )
        )

    expected_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/data/expected_optimized_kinematic_parameters"
        / f"{optimizer_method}_{device.type}.pt"
    )

    expected = torch.load(expected_path, map_location=device, weights_only=True)

    torch.testing.assert_close(
        calibrated_kinematic_deviation_parameters,
        expected["kinematic_deviations"],
        atol=5e-2,
        rtol=5e-2,
    )
    torch.testing.assert_close(
        calibrated_actuator_parameters,
        expected["actuator_parameters"],
        atol=5e-2,
        rtol=5e-2,
    )
