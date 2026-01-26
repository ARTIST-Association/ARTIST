import pathlib
from typing import Any

import h5py
import pytest
import torch

from artist import ARTIST_ROOT
from artist.core.loss_functions import FocalSpotLoss, KLDivergenceLoss, Loss
from artist.core.motor_position_optimizer import MotorPositionsOptimizer
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary


@pytest.fixture
def focal_spot() -> torch.Tensor:
    """
    Use a focal spot as target in the loss function.

    Returns
    -------
    torch.Tensor
        The desired focal spot.
        Tensor of shape [4].
    """
    ground_truth = torch.tensor([1.1493, -0.5030, 57.0474, 1.0000])

    return ground_truth


@pytest.fixture
def distribution(device: torch.device) -> torch.Tensor:
    """
    Use a distribution as target in the loss function.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    torch.Tensor
        The desired distribution.
        Tensor of shape [bitmap_resolution_e, bitmap_resolution_u].
    """
    distribution_path_group_1 = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/data/expected_optimized_motor_positions"
        / "distribution.pt"
    )

    ground_truth = torch.load(
        distribution_path_group_1, map_location=device, weights_only=True
    )

    return ground_truth


@pytest.mark.parametrize(
    "loss_class, ground_truth_fixture_name, scheduler",
    [
        (FocalSpotLoss, "focal_spot", config_dictionary.cyclic),
        (KLDivergenceLoss, "distribution", config_dictionary.reduce_on_plateau),
    ],
)
def test_motor_positions_optimizer(
    loss_class: Loss,
    ground_truth_fixture_name: torch.Tensor,
    scheduler: str,
    request: pytest.FixtureRequest,
    ddp_setup_for_testing: dict[str, Any],
    device: torch.device,
) -> None:
    """
    Test the motor positions optimizer.

    Parameters
    ----------
    loss_class : Loss
        The loss class.
    ground_truth_fixture_name : str
        A fixture to retrieve the ground truth.
    scheduler : str
        The scheduler to be used.
    request : pytest.FixtureRequest
        The pytest fixture used to consider different test cases.
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
        config_dictionary.min: 1e-3,
        config_dictionary.max: 2e-3,
        config_dictionary.step_size_up: 100,
        config_dictionary.reduce_factor: 0.9,
        config_dictionary.patience: 100,
        config_dictionary.threshold: 1e-3,
        config_dictionary.cooldown: 20,
    }

    optimization_configuration = {
        config_dictionary.initial_learning_rate: 1e-5,
        config_dictionary.tolerance: 0.0005,
        config_dictionary.max_epoch: 21,
        config_dictionary.batch_size: 50,
        config_dictionary.log_step: 0,
        config_dictionary.early_stopping_delta: 1e-4,
        config_dictionary.early_stopping_patience: 20,
        config_dictionary.early_stopping_window: 10,
        config_dictionary.scheduler: scheduler,
        config_dictionary.scheduler_parameters: scheduler_parameters,
    }

    scenario_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/data/scenarios/test_scenario_paint_four_heliostats.h5"
    )

    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file, device=device
        )

    ddp_setup_for_testing[config_dictionary.device] = device
    ddp_setup_for_testing[config_dictionary.groups_to_ranks_mapping] = {0: [0, 1]}

    # Create the motor positions optimizer.
    motor_positions_optimizer = MotorPositionsOptimizer(
        ddp_setup=ddp_setup_for_testing,
        scenario=scenario,
        optimization_configuration=optimization_configuration,
        incident_ray_direction=torch.tensor([0.0, 1.0, 0.0, 0.0], device=device),
        target_area_index=1,
        ground_truth=request.getfixturevalue(ground_truth_fixture_name).to(device),
        bitmap_resolution=torch.tensor([256, 256], device=device),
        device=device,
    )

    loss_definition = (
        FocalSpotLoss(scenario=scenario)
        if loss_class is FocalSpotLoss
        else KLDivergenceLoss()
    )

    # Optimize the motor positions.
    _ = motor_positions_optimizer.optimize(
        loss_definition=loss_definition, device=device
    )

    for index, heliostat_group in enumerate(scenario.heliostat_field.heliostat_groups):
        expected_path = (
            pathlib.Path(ARTIST_ROOT)
            / "tests/data/expected_optimized_motor_positions"
            / f"{ground_truth_fixture_name}_group_{index}_{device.type}.pt"
        )

        expected = torch.load(expected_path, map_location=device, weights_only=True)

        torch.testing.assert_close(
            heliostat_group.kinematic.motor_positions,
            expected,
            atol=5e-3,
            rtol=5e-3,
        )
