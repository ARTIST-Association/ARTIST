import pathlib
from typing import Callable

import h5py
import pytest
import torch

from artist import ARTIST_ROOT
from artist.core import loss_functions
from artist.core.motor_position_optimizer import MotorPositionsOptimizer
from artist.scenario.scenario import Scenario
from artist.util import utils
from artist.util.environment_setup import DistributedEnvironmentTypedDict


@pytest.fixture
def focal_spot_loss() -> tuple[torch.Tensor, Callable[..., torch.Tensor]]:
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
    optimization_goal = torch.tensor([1.1493, -0.5030, 57.0474, 1.0000])
    loss_function = loss_functions.focal_spot_loss

    return optimization_goal, loss_function


@pytest.fixture
def distribution_loss():
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
    optimization_goal = u_trapezoid.unsqueeze(1) * e_trapezoid.unsqueeze(0)
    loss_function = loss_functions.distribution_loss_kl_divergence

    return optimization_goal, loss_function


@pytest.mark.parametrize("loss_fixture_name", ["focal_spot_loss", "distribution_loss"])
def test_motor_positions_optimizer(
    loss_fixture_name: str,
    request: pytest.FixtureRequest,
    ddp_setup_for_testing: DistributedEnvironmentTypedDict,
    device: torch.device,
) -> None:
    """
    Test the motor positions optimizer.

    Parameters
    ----------
    loss_fixture_name : str
        The fixture determining the loss function and optimization target.
    request : pytest.FixtureRequest
        The pytest fixture used to consider different test cases.
    optimizer_method : str
        The method used for optimization. The motor positions can be optimized to aim at a
        specific coordinate or to match a specific distribution.
    ddp_setup_for_testing : DistributedEnvironmentTypedDict
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

    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file, device=device
        )

    ddp_setup_for_testing["device"] = device
    ddp_setup_for_testing["groups_to_ranks_mapping"] = {0: [0]}

    # Set optimizer paramteres.
    initial_learning_rate = 1e-3
    max_epoch = 5

    # Create the motor positions optimizer.
    motor_positions_optimizer = MotorPositionsOptimizer(
        ddp_setup=ddp_setup_for_testing,
        scenario=scenario,
        incident_ray_direction=torch.tensor([0.0, 1.0, 0.0, 0.0], device=device),
        target_area_index=1,
        optimization_goal=request.getfixturevalue(loss_fixture_name)[0],
        bitmap_resolution=torch.tensor([256, 256], device=device),
        initial_learning_rate=initial_learning_rate,
        max_epoch=max_epoch,
        num_log=max_epoch,
        device=device,
    )

    # Optimize the motor positions.
    motor_positions_optimizer.optimize(
        loss_function=request.getfixturevalue(loss_fixture_name)[1], device=device
    )

    for index, heliostat_group in enumerate(scenario.heliostat_field.heliostat_groups):
        expected_path = (
            pathlib.Path(ARTIST_ROOT)
            / "tests/data/expected_optimized_motor_positions"
            / f"{request.getfixturevalue(loss_fixture_name)[1].__name__}_group_{index}_{device.type}.pt"
        )

        expected = torch.load(expected_path, map_location=device, weights_only=True)

        torch.testing.assert_close(
            heliostat_group.kinematic.motor_positions,
            expected,
            atol=5e-4,
            rtol=5e-4,
        )
