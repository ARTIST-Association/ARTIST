import os
import pathlib

import h5py
import psutil
import pytest
import torch
from pytest_mock import MockerFixture

from artist import ARTIST_ROOT
from artist.field.tower_target_area import TargetArea
from artist.field.tower_target_area_array import TargetAreaArray
from artist.scenario import Scenario
from artist.util import config_dictionary, paint_loader, set_logger_config, utils
from artist.util.alignment_optimizer import AlignmentOptimizer

# Set up logger.
set_logger_config()


@pytest.mark.parametrize(
    "optimizer_method, scenario_name, calibration_file, tolerance, max_epoch, initial_lr, lr_factor, lr_patience, lr_threshold",
    [
        (
            "use_motor_positions",
            "test_scenario_paint_single_heliostat",
            "calibration-properties",
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
            "calibration-properties",
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

    # The distributed environment is setup and destroyed using a Generator object.
    environment_generator = utils.setup_distributed_environment(device=device)

    is_distributed, rank, world_size = next(environment_generator)

    if device.type == "cuda" and is_distributed:
        gpu_count = torch.cuda.device_count()
        device_id = rank % gpu_count
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)

    if device.type == "cpu":
        num_cores = os.cpu_count()
        if isinstance(num_cores, int):
            cores_per_rank = num_cores // world_size
        start_core = rank * cores_per_rank
        end_core = start_core + cores_per_rank - 1
        process = psutil.Process(os.getpid())
        process.cpu_affinity(list(range(start_core, end_core + 1)))

    scenario_path = (
        pathlib.Path(ARTIST_ROOT) / f"tests/data/scenarios/{scenario_name}.h5"
    )
    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file, device=device
        )

    optimizable_parameters = utils.get_rigid_body_kinematic_parameters_from_scenario(
        kinematic=scenario.heliostats.heliostat_list[0].kinematic
    )

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
    calibration_properties_path = (
        pathlib.Path(ARTIST_ROOT) / f"tests/data/paint/AA39/{calibration_file}.json"
    )

    (
        calibration_target_name,
        center_calibration_image,
        incident_ray_direction,
        motor_positions,
    ) = paint_loader.extract_paint_calibration_data(
        calibration_properties_path=calibration_properties_path,
        power_plant_position=scenario.power_plant_position,
        device=device,
    )

    # Create alignment optimizer.
    alignment_optimizer = AlignmentOptimizer(
        scenario=scenario,
        optimizer=optimizer,
        scheduler=scheduler,
        world_size=world_size,
        rank=rank,
        batch_size=1000,
        is_distributed=is_distributed,
    )

    if optimizer_method == config_dictionary.optimizer_use_raytracing:
        motor_positions = None

    optimized_parameters, _ = alignment_optimizer.optimize(
        tolerance=tolerance,
        max_epoch=max_epoch,
        center_calibration_image=center_calibration_image,
        incident_ray_direction=incident_ray_direction,
        calibration_target_name=calibration_target_name,
        motor_positions=motor_positions,
        num_log=10,
        device=device,
    )

    expected_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/data/expected_optimized_alignment_parameters"
        / f"{optimizer_method}_{device.type}.pt"
    )
    expected = torch.load(expected_path, map_location=device, weights_only=True)

    torch.testing.assert_close(optimized_parameters, expected, atol=5e-2, rtol=5e-2)


def test_raytracing_exception(mocker: MockerFixture, device: torch.device) -> None:
    """
    Test raytracing alignment optimization with faulty calibration target.

    Parameters
    ----------
    mocker : MockerFixture
        A pytest-mock fixture used to create mock objects.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    mock_alignment_optimizer = mocker.MagicMock(spec=AlignmentOptimizer)

    mock_scenario = mocker.MagicMock(spec=Scenario)
    mock_alignment_optimizer.scenario = mock_scenario
    mock_target_areas = mocker.MagicMock(spec=TargetAreaArray)
    mock_alignment_optimizer.scenario.target_areas = mock_target_areas
    mock_target_area = mocker.MagicMock(spec=TargetArea)
    mock_target_area.name = "receiver"
    mock_alignment_optimizer.scenario.target_areas.target_area_list = [mock_target_area]

    mock_alignment_optimizer._optimize_kinematic_parameters_with_raytracing = (
        AlignmentOptimizer._optimize_kinematic_parameters_with_raytracing.__get__(
            mock_alignment_optimizer, AlignmentOptimizer
        )
    )

    with pytest.raises(KeyError) as exc_info:
        mock_alignment_optimizer._optimize_kinematic_parameters_with_raytracing(
            tolerance=0.0,
            max_epoch=0,
            center_calibration_image=torch.tensor([0.0]),
            incident_ray_direction=torch.tensor([0.0]),
            calibration_target_name="invalid_calibration_target_name",
            device=device,
        )
    assert "The specified calibration target is not included in the scenario!" in str(
        exc_info.value
    )
