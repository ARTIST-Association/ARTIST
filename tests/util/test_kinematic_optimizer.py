import pathlib

import h5py
import pytest
import torch

from artist import ARTIST_ROOT
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util import config_dictionary, paint_loader, set_logger_config
from artist.util.kinematic_optimizer import KinematicOptimizer
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
            "test_scenario_paint_multiple_heliostats",
            "AA39-calibration-properties",
            1e-7,
            30,
            0.0005,
            0.1,
            18,
            0.1,
        ),
    ],
)
def test_kinematic_optimizer(
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
    Test the kinematic optimization methods.

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

    # Create a calibration scenario from the original scenario.
    # It contains a single heliostat, chosen by its index.
    calibration_scenario = scenario.create_calibration_scenario(
        heliostat_index=0, device=device
    )

    optimizable_parameters = [
        calibration_scenario.heliostat_field.all_kinematic_deviation_parameters.requires_grad_(),
        calibration_scenario.heliostat_field.all_actuator_parameters.requires_grad_(),
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
        power_plant_position=calibration_scenario.power_plant_position,
        device=device,
    )

    incident_ray_directions = (
        torch.tensor([0.0, 0.0, 0.0, 1.0], device=device) - sun_positions
    )

    # Create alignment optimizer.
    alignment_optimizer = KinematicOptimizer(
        scenario=calibration_scenario,
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
        / "tests/data/expected_optimized_kinematic_parameters"
        / f"{optimizer_method}_{device.type}.pt"
    )
    expected = torch.load(expected_path, map_location=device, weights_only=True)

    torch.testing.assert_close(
        calibration_scenario.heliostat_field.all_kinematic_deviation_parameters,
        expected["kinematic_deviations"],
        atol=5e-2,
        rtol=5e-2,
    )
    torch.testing.assert_close(
        calibration_scenario.heliostat_field.all_actuator_parameters,
        expected["actuator_parameters"],
        atol=5e-2,
        rtol=5e-2,
    )

    # Also assert if the align with motor position method works as expected.
    if optimizer_method == config_dictionary.optimizer_use_motor_positions:
        calibration_scenario.heliostat_field.align_surfaces_with_motor_positions(
            motor_positions=all_calibration_motor_positions, device=device
        )

        ray_tracer = HeliostatRayTracer(scenario=calibration_scenario)

        final_bitmap = ray_tracer.trace_rays(
            incident_ray_direction=incident_ray_directions.to(device),
            target_area=calibration_scenario.get_target_area(
                calibration_target_names[0]
            ),
            device=device,
        )

        expected_path = (
            pathlib.Path(ARTIST_ROOT)
            / "tests/data/expected_bitmaps_integration"
            / f"motor_position_alignment_{device.type}.pt"
        )

        expected = torch.load(expected_path, map_location=device, weights_only=True)
        torch.testing.assert_close(final_bitmap, expected, atol=5e-4, rtol=5e-4)
