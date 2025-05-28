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

    for index, heliostat_group in enumerate(scenario.heliostat_field.heliostat_groups):
        (
            focal_spots_calibration,
            incident_ray_directions_calibration,
            motor_positions_calibration,
            heliostats_mask_calibration,
            target_area_mask_calibration,
        ) = paint_loader.extract_paint_calibration_data(
            heliostat_calibration_mapping=[
                (heliostat_name, paths)
                for heliostat_name, paths in heliostat_calibration_mapping
                if heliostat_name in heliostat_group.names
            ],
            heliostat_names=heliostat_group.names,
            target_area_names=scenario.target_areas.names,
            power_plant_position=scenario.power_plant_position,
            device=device,
        )

        # Select the kinematic parameters to be optimized and calibrated.
        optimizable_parameters = [
            heliostat_group.kinematic_deviation_parameters.requires_grad_(),
            heliostat_group.actuator_parameters.requires_grad_(),
        ]

        optimizer = torch.optim.Adam(optimizable_parameters, lr=initial_lr)

        # Create the kinematic optimizer.
        kinematic_optimizer = KinematicOptimizer(
            scenario=scenario,
            heliostat_group=heliostat_group,
            optimizer=optimizer,
        )

        if optimizer_method == config_dictionary.optimizer_use_raytracing:
            motor_positions_calibration = None

        # Calibrate the kinematic.
        kinematic_optimizer.optimize(
            focal_spots_calibration=focal_spots_calibration,
            incident_ray_directions=incident_ray_directions_calibration,
            active_heliostats_mask=heliostats_mask_calibration,
            target_area_mask_calibration=target_area_mask_calibration,
            motor_positions_calibration=motor_positions_calibration,
            tolerance=tolerance,
            max_epoch=max_epoch,
            num_log=max_epoch,
            device=device,
        )

        expected_path = (
            pathlib.Path(ARTIST_ROOT)
            / "tests/data/expected_optimized_kinematic_parameters"
            / f"{optimizer_method}_group{index}_{device.type}.pt"
        )

        expected = torch.load(expected_path, map_location=device, weights_only=True)

        torch.testing.assert_close(
            heliostat_group.kinematic_deviation_parameters,
            expected["kinematic_deviations"],
            atol=5e-2,
            rtol=5e-2,
        )
        torch.testing.assert_close(
            heliostat_group.actuator_parameters,
            expected["actuator_parameters"],
            atol=5e-2,
            rtol=5e-2,
        )
