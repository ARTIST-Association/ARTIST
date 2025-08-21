import pathlib

import h5py
import pytest
import torch

from artist import ARTIST_ROOT
from artist.core import loss_functions
from artist.core.motor_position_optimizer import MotorPositionsOptimizer
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, utils
from artist.util.environment_setup import DistributedEnvironmentTypedDict


@pytest.mark.parametrize(
    "optimizer_method",
    [
        (
            config_dictionary.optimization_to_distribution
        ),
        (
            config_dictionary.optimization_to_focal_spot
        )
    ],
)
def test_motor_positions_optimizer(
    optimizer_method: str,
    ddp_setup_for_testing: DistributedEnvironmentTypedDict,
    device: torch.device,
) -> None:
    """
    Test the motor positions optimizer.

    Parameters
    ----------
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

    if optimizer_method == config_dictionary.optimization_to_focal_spot:
        optimization_goal = torch.tensor(
            [1.1493, -0.5030, 57.0474, 1.0000], device=device
        )
        loss_function = loss_functions.focal_spot_loss
        
    if optimizer_method == config_dictionary.optimization_to_distribution:
        e_trapezoid = utils.trapezoid_distribution(
            total_width=256, slope_width=30, plateau_width=180, device=device
        )
        u_trapezoid = utils.trapezoid_distribution(
            total_width=256, slope_width=30, plateau_width=180, device=device
        )
        optimization_goal = u_trapezoid.unsqueeze(1) * e_trapezoid.unsqueeze(0)
        loss_function = loss_functions.distribution_loss_kl_divergence

    # Set optimizer paramteres.
    initial_learning_rate = 1e-3
    max_epoch = 5

    # Create the motor positions optimizer.
    motor_positions_optimizer = MotorPositionsOptimizer(
        ddp_setup=ddp_setup_for_testing,
        scenario=scenario,
        incident_ray_direction=torch.tensor([0.0, 1.0, 0.0, 0.0], device=device),
        target_area_index=1,
        optimization_goal=optimization_goal,
        bitmap_resolution=torch.tensor([256, 256], device=device),
        initial_learning_rate=initial_learning_rate,
        max_epoch=max_epoch,
        num_log=max_epoch,
        device=device,
    )

    # Optimize the motor positions.
    motor_positions_optimizer.optimize(
        loss_function=loss_function,
        device=device
    )

    for index, heliostat_group in enumerate(scenario.heliostat_field.heliostat_groups):
        expected_path = (
            pathlib.Path(ARTIST_ROOT)
            / "tests/data/expected_optimized_motor_positions"
            / f"{optimizer_method}_group_{index}_{device.type}.pt"
        )
 
        expected = torch.load(expected_path, map_location=device, weights_only=True)
        
        torch.testing.assert_close(
            heliostat_group.kinematic.motor_positions,
            expected,
            atol=5e-4,
            rtol=5e-4,
        )
