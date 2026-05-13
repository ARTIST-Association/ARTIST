import pathlib

import h5py
import pytest
import torch

from artist import ARTIST_ROOT
from artist.optim import MotorPositionsOptimizer
from artist.optim.loss import FocalSpotLoss, KLDivergenceLoss, Loss
from artist.scenario import Scenario
from artist.util import constants
from artist.util.env import DdpSetup


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
    ground_truth = torch.tensor([1.5, -3.2357, 37.0, 1.0])

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
    path = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/data/expected_optimized_motor_positions"
        / "distribution.pt"
    )

    ground_truth = torch.load(path, map_location=device, weights_only=True)

    # Scale bitmap intensity to account for the dni and ray magnitude set in this test.
    return ground_truth * 19400


@pytest.mark.parametrize(
    "loss_class, ground_truth_fixture_name, early_stopping_window, scheduler",
    [
        (FocalSpotLoss, "focal_spot", 50, constants.cyclic),
        (KLDivergenceLoss, "distribution", 50, constants.reduce_on_plateau),
        (KLDivergenceLoss, "distribution", 10, constants.reduce_on_plateau),
    ],
)
def test_motor_positions_optimizer(
    loss_class: Loss,
    ground_truth_fixture_name: str,
    early_stopping_window: int,
    scheduler: str,
    request: pytest.FixtureRequest,
    ddp_setup_for_testing: DdpSetup,
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
    early_stopping_window : int
        Number of epochs used to estimate loss trend.
    scheduler : str
        The scheduler to be used.
    request : pytest.FixtureRequest
        The pytest fixture used to consider different test cases.
    ddp_setup_for_testing : DdpSetup
        Information about the distributed environment, process_groups, devices, ranks, world_size, heliostat group to ranks mapping.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    scheduler_dict = {
        constants.scheduler_type: scheduler,
        constants.lr_min: 1e-3,
        constants.lr_max: 2e-3,
        constants.step_size_up: 100,
        constants.reduce_factor: 0.9,
        constants.patience: 100,
        constants.threshold: 1e-3,
        constants.cooldown: 20,
    }
    optimizer_dict = {
        constants.initial_learning_rate: 1e-3,
        constants.tolerance: 0.0005,
        constants.max_epoch: 50,
        constants.batch_size: 50,
        constants.log_step: 1,
        constants.early_stopping_delta: 1.0,
        constants.early_stopping_patience: 2,
        constants.early_stopping_window: early_stopping_window,
    }
    constraint_dict = {
        constants.rho_flux_integral: 1.0,
        constants.rho_local_flux: 1.0,
        constants.rho_intercept: 1.0,
        constants.max_flux_density: 1000000,
    }
    # Combine configurations.
    optimization_configuration = {
        constants.optimization: optimizer_dict,
        constants.scheduler: scheduler_dict,
        constants.constraints: constraint_dict,
    }
    scenario_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/data/scenarios/test_scenario_paint_four_heliostats.h5"
    )

    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file, device=device
        )

    ddp_setup_for_testing["device"] = device

    # Create the motor positions optimizer.
    motor_positions_optimizer = MotorPositionsOptimizer(
        ddp_setup=ddp_setup_for_testing,
        scenario=scenario,
        optimization_configuration=optimization_configuration,
        incident_ray_direction=torch.tensor([0.0, 1.0, 0.0, 0.0], device=device),
        target_area_index=1,
        ground_truth=request.getfixturevalue(ground_truth_fixture_name).to(device),
        dni=800,
        bitmap_resolution=torch.tensor([256, 256], device=device),
        device=device,
    )

    loss_definition = (
        FocalSpotLoss(scenario=scenario)
        if loss_class is FocalSpotLoss
        else KLDivergenceLoss()
    )

    # Optimize the motor positions.
    _, _, _, _, _ = motor_positions_optimizer.optimize(
        loss_definition=loss_definition, device=device
    )

    for index, heliostat_group in enumerate(scenario.heliostat_field.heliostat_groups):
        expected_path = (
            pathlib.Path(ARTIST_ROOT)
            / "tests/data/expected_optimized_motor_positions"
            / f"{ground_truth_fixture_name}_group_{index}_{early_stopping_window}_{device.type}.pt"
        )

        expected = torch.load(expected_path, map_location=device, weights_only=True)

        torch.testing.assert_close(
            heliostat_group.kinematics.motor_positions,
            expected,
            atol=5e-3,
            rtol=5e-2,
        )
