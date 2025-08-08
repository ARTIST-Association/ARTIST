import pathlib

import h5py
import pytest
import torch

from artist import ARTIST_ROOT
from artist.core.kinematic_optimizer import KinematicOptimizer
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, set_logger_config

# Set up logger.
set_logger_config()


@pytest.mark.parametrize(
    "optimizer_method, tolerance, max_epoch, initial_lr",
    [
        (
            config_dictionary.kinematic_calibration_motor_positions,
            0.0005,
            15,
            0.001,
        ),
        (
            config_dictionary.kinematic_calibration_raytracing,
            0.0005,
            15,
            0.0001,
        ),
    ],
)
def test_kinematic_optimizer(
    optimizer_method: str,
    tolerance: float,
    max_epoch: int,
    initial_lr: float,
    ddp_setup_for_testing: dict[str, torch.device | bool | int | torch.distributed.ProcessGroup | dict[int, list[int]] | None],
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
    ddp_setup_for_testing : dict[str, torch.device | bool | int | torch.distributed.ProcessGroup | dict[int, list[int]] | None]
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

    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file, device=device
        )

    ddp_setup_for_testing["device"] = device
    ddp_setup_for_testing["groups_to_ranks_mapping"] = {0: [0]}

    # Create the kinematic optimizer.
    kinematic_optimizer = KinematicOptimizer(
        ddp_setup=ddp_setup_for_testing,
        scenario=scenario,
        heliostat_data_mapping=heliostat_data_mapping,
        calibration_method=optimizer_method,
        initial_learning_rate=initial_lr,
        tolerance=tolerance,
        max_epoch=max_epoch,
        num_log=max_epoch,
        device=device,
    )

    # Calibrate the kinematic.
    kinematic_optimizer.optimize(device=device)

    expected_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/data/expected_optimized_kinematic_parameters"
        / f"{optimizer_method}_group_0_{device.type}.pt"
    )

    expected = torch.load(expected_path, map_location=device, weights_only=True)

    torch.testing.assert_close(
        scenario.heliostat_field.heliostat_groups[0].kinematic.deviation_parameters,
        expected["kinematic_deviations"],
        atol=5e-2,
        rtol=5e-2,
    )
    torch.testing.assert_close(
        scenario.heliostat_field.heliostat_groups[0].kinematic.actuators.actuator_parameters,
        expected["actuator_parameters"],
        atol=5e-2,
        rtol=5e-2,
    )
