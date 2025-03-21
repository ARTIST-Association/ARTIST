import pathlib

import h5py
import pytest
import torch

from artist import ARTIST_ROOT
from artist.util import config_dictionary, paint_loader, set_logger_config
from artist.util.alignment_optimizer import KinematicOptimizer
from artist.util.scenario import Scenario

# Set up logger.
set_logger_config()


@pytest.mark.parametrize(
    "optimizer_method, scenario_name, calibration_file, tolerance, max_epoch, initial_lr, lr_factor, lr_patience, lr_threshold",
    [
        (
            "use_motor_positions",
            "test_scenario_paint_single_heliostat",
            "AA39-calibration-properties",
            1e-7,
            150,
            0.01,
            0.1,
            20,
            0.1,
        ),
        (
            "use_raytracing",
            "test_scenario_paint_single_heliostat",
            "AA39-calibration-properties",
            1e-7,
            27,
            0.0002,
            0.1,
            18,
            0.1,
        ),
    ],
)
def test_alignment_optimizer_methods(
    optimizer_method: str,
    scenario_name: str,
    calibration_file: str,
    tolerance: float,
    max_epoch: int,
    initial_lr: float,
    lr_factor: float,
    lr_patience: int,
    lr_threshold: float,
    device: torch.device,
) -> None:
    """
    Test the alignemnt optimization methods.

    Parameters
    ----------
    optimizer_method : str
        The name of the optimizer method.
    scenario_name : str
        The name of the test scenario.
    calibration_file : str
        The file containing calibration data.
    tolerance : float
        Tolerance for the optimizer.
    max_epoch : int
        The maximum amount of epochs for the optimization loop.
    initial_lr : float
        The initial learning rate.
    lr_factor : float
        The scheduler factor.
    lr_patience : int
        The scheduler patience.
    lr_threshold : float
        The scheduler threshold.
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
        pathlib.Path(ARTIST_ROOT) / f"tests/data/scenarios/{scenario_name}.h5"
    )
    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file, device=device
        )

    optimizable_parameters = [
        scenario.heliostat_field.all_kinematic_deviation_parameters.requires_grad_(),
        scenario.heliostat_field.all_actuator_parameters.requires_grad_(),
    ]

    optimizer = torch.optim.Adam(optimizable_parameters, lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
        threshold=lr_threshold,
        threshold_mode="abs",
    )

    # Load the calibration data.
    calibration_properties_paths = [
        (pathlib.Path(ARTIST_ROOT) / f"tests/data/field_data/{calibration_file}.json")
    ]

    (
        calibration_target_names,
        center_calibration_images,
        sun_positions,
        all_calibration_motor_positions,
    ) = paint_loader.extract_paint_calibration_data(
        calibration_properties_paths=calibration_properties_paths,
        power_plant_position=scenario.power_plant_position,
        device=device,
    )

    incident_ray_directions = (
        torch.tensor([0.0, 0.0, 0.0, 1.0], device=device) - sun_positions
    )

    # Create alignment optimizer.
    alignment_optimizer = KinematicOptimizer(
        scenario=scenario,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    if optimizer_method == config_dictionary.optimizer_use_raytracing:
        all_calibration_motor_positions = None

    alignment_optimizer.optimize(
        tolerance=tolerance,
        max_epoch=max_epoch,
        center_calibration_images=center_calibration_images,
        incident_ray_directions=incident_ray_directions,
        calibration_target_names=calibration_target_names,
        motor_positions=all_calibration_motor_positions,
        num_log=10,
        device=device,
    )

    expected_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/data/expected_optimized_alignment_parameters"
        / f"{optimizer_method}_{device.type}.pt"
    )
    expected = torch.load(expected_path, map_location=device, weights_only=True)

    torch.testing.assert_close(
        scenario.heliostat_field.all_kinematic_deviation_parameters,
        expected["kinematic_deviations"],
        atol=5e-2,
        rtol=5e-2,
    )
    torch.testing.assert_close(
        scenario.heliostat_field.all_actuator_parameters,
        expected["actuator_parameters"],
        atol=5e-2,
        rtol=5e-2,
    )
