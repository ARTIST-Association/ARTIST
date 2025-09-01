import pathlib
from typing import Any

import h5py
import pytest
import torch

from artist import ARTIST_ROOT
from artist.core.loss_functions import FocalSpotLoss, KLDivergenceLoss
from artist.core.motor_position_optimizer import MotorPositionsOptimizer
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, utils


@pytest.fixture
def focal_spot() -> torch.Tensor:
    """
    Use a focal spot as target in the loss function.

    Returns
    -------
    optimization_goal : torch.Tensor
        The desired focal spot.
        Tensor of shape [4] or.
    loss_function : Callable[..., torch.Tensor],
        A callable function that computes the loss. It accepts predictions and targets
        and optionally other keyword arguments and return a tensor with loss values.
    """
    ground_truth = torch.tensor([1.1493, -0.5030, 57.0474, 1.0000])

    return ground_truth


@pytest.fixture
def distribution() -> torch.Tensor:
    """
    Use a distribution as target in the loss function.

    Returns
    -------
    optimization_goal : torch.Tensor
        The desired distribution.
        Tensor of shape [bitmap_resolution_e, bitmap_resolution_u].
    loss_function : Callable[..., torch.Tensor],
        A callable function that computes the loss. It accepts predictions and targets
        and optionally other keyword arguments and return a tensor with loss values.
    """
    e_trapezoid = utils.trapezoid_distribution(
        total_width=256, slope_width=30, plateau_width=180
    )
    u_trapezoid = utils.trapezoid_distribution(
        total_width=256, slope_width=30, plateau_width=180
    )
    ground_truth = u_trapezoid.unsqueeze(1) * e_trapezoid.unsqueeze(0)

    return ground_truth


@pytest.mark.parametrize(
    "ground_truth, early_stopping_delta",
    [
        ("focal_spot", 1e-4),
        ("distribution", 1.0),
    ],
)
def test_motor_positions_optimizer(
    ground_truth: torch.Tensor,
    early_stopping_delta: float,
    request: pytest.FixtureRequest,
    ddp_setup_for_testing: dict[str, Any],
    device: torch.device,
) -> None:
    """
    Test the motor positions optimizer.

    Parameters
    ----------
    loss_fixture_name : str
        The fixture determining the loss function and optimization target.
    early_stopping_delta : float
        The minimum required improvement to prevent early stopping.
    request : pytest.FixtureRequest
        The pytest fixture used to consider different test cases.
    optimizer_method : str
        The method used for optimization. The motor positions can be optimized to aim at a
        specific coordinate or to match a specific distribution.
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
    }

    optimization_configuration = {
        config_dictionary.initial_learning_rate: 1e-3,
        config_dictionary.tolerance: 0.0005,
        config_dictionary.max_epoch: 15,
        config_dictionary.num_log: 1,
        config_dictionary.early_stopping_delta: early_stopping_delta,
        config_dictionary.early_stopping_patience: 13,
        config_dictionary.scheduler: config_dictionary.cyclic,
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
        ground_truth=request.getfixturevalue(ground_truth).to(device),
        bitmap_resolution=torch.tensor([256, 256], device=device),
        device=device,
    )

    if ground_truth == "focal_spot":
        loss_definition = FocalSpotLoss(
            scenario=scenario,
        )
    if ground_truth == "distribution":
        loss_definition = KLDivergenceLoss()

    # Optimize the motor positions.
    _ = motor_positions_optimizer.optimize(
        loss_definition=loss_definition, device=device
    )

    for index, heliostat_group in enumerate(scenario.heliostat_field.heliostat_groups):
        expected_path = (
            pathlib.Path(ARTIST_ROOT)
            / "tests/data/expected_optimized_motor_positions"
            / f"{ground_truth}_group_{index}_{device.type}.pt"
        )

        expected = torch.load(expected_path, map_location=device, weights_only=True)

        torch.testing.assert_close(
            heliostat_group.kinematic.motor_positions,
            expected,
            atol=5e-3,
            rtol=5e-3,
        )
