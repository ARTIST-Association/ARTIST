import pathlib
from typing import Any, Callable

import h5py
import pytest
import torch

from artist import ARTIST_ROOT
from artist.core import loss_functions
from artist.core.kinematic_calibrator import KinematicCalibrator
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, set_logger_config

# Set up logger.
set_logger_config()


@pytest.mark.parametrize(
    "calibration_method, tolerance, max_epoch, initial_lr, loss_function",
    [
        (
            config_dictionary.kinematic_calibration_motor_positions,
            0.0005,
            15,
            0.001,
            loss_functions.vector_loss,
        ),
        (
            config_dictionary.kinematic_calibration_raytracing,
            0.0005,
            15,
            0.0001,
            loss_functions.focal_spot_loss,
        ),
    ],
)
def test_kinematic_calibrator(
    calibration_method: str,
    tolerance: float,
    max_epoch: int,
    initial_lr: float,
    loss_function: Callable[..., torch.Tensor],
    ddp_setup_for_testing: dict[str, Any],
    device: torch.device,
) -> None:
    """
    Test the kinematic calibration methods.

    Parameters
    ----------
    calibration_method : str
        The name of the calibration method.
    tolerance : float
        Tolerance for the optimizer.
    max_epoch : int
        The maximum amount of epochs for the optimization loop.
    initial_lr : float
        The initial learning rate.
    loss_function : Callable[..., torch.Tensor]
        A callable function that computes the loss. It accepts predictions and targets
        and optionally other keyword arguments and return a tensor with loss values.
    ddp_setup_for_testing : dict[str, Any]
        Information about the distributed environment, process_groups, devices, ranks, world_Size, heliostat group to ranks mapping.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    scheduler_parameters = {
        config_dictionary.gamma: 0.9,
    }

    optimization_configuration = {
        config_dictionary.initial_learning_rate: initial_lr,
        config_dictionary.tolerance: tolerance,
        config_dictionary.max_epoch: max_epoch,
        config_dictionary.num_log: 1,
        config_dictionary.early_stopping_delta: 1e-4,
        config_dictionary.early_stopping_patience: 10,
        config_dictionary.scheduler: config_dictionary.exponential,
        config_dictionary.scheduler_parameters: scheduler_parameters,
    }

    scenario_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/data/scenarios/test_scenario_paint_four_heliostats.h5"
    )

    heliostat_data_mapping = [
        (
            "AA39",
            [
                pathlib.Path(ARTIST_ROOT)
                / "tests/data/field_data/AA39-calibration-properties_1.json",
                pathlib.Path(ARTIST_ROOT)
                / "tests/data/field_data/AA39-calibration-properties_2.json",
            ],
            [
                pathlib.Path(ARTIST_ROOT)
                / "tests/data/field_data/AA39-flux_centered_1.png",
                pathlib.Path(ARTIST_ROOT)
                / "tests/data/field_data/AA39-flux-centered_2.png",
            ],
        ),
        (
            "AA31",
            [
                pathlib.Path(ARTIST_ROOT)
                / "tests/data/field_data/AA31-calibration-properties_1.json"
            ],
            [
                pathlib.Path(ARTIST_ROOT)
                / "tests/data/field_data/AA31-flux-centered_1.png"
            ],
        ),
    ]

    data: dict[str, str | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]] = {
        config_dictionary.data_source: config_dictionary.paint,
        config_dictionary.heliostat_data_mapping: heliostat_data_mapping,
    }

    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file, device=device
        )

    ddp_setup_for_testing[config_dictionary.device] = device
    ddp_setup_for_testing[config_dictionary.groups_to_ranks_mapping] = {0: [0, 1]}

    # Create the kinematic optimizer.
    kinematic_calibrator = KinematicCalibrator(
        ddp_setup=ddp_setup_for_testing,
        scenario=scenario,
        data=data,
        optimization_configuration=optimization_configuration,
        calibration_method=calibration_method,
    )

    # Calibrate the kinematic.
    _ = kinematic_calibrator.calibrate(loss_function=loss_function, device=device)

    for index, heliostat_group in enumerate(scenario.heliostat_field.heliostat_groups):
        expected_path = (
            pathlib.Path(ARTIST_ROOT)
            / "tests/data/expected_optimized_kinematic_parameters"
            / f"{calibration_method}_group_{index}_{device.type}.pt"
        )

        expected = torch.load(expected_path, map_location=device, weights_only=True)

        torch.testing.assert_close(
            heliostat_group.kinematic.deviation_parameters,
            expected["kinematic_deviations"],
            atol=5e-2,
            rtol=5e-2,
        )
        torch.testing.assert_close(
            heliostat_group.kinematic.actuators.actuator_parameters,
            expected["actuator_parameters"],
            atol=5e-2,
            rtol=5e-2,
        )
