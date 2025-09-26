import pathlib
from typing import Any

import h5py
import paint.util.paint_mappings as paint_mappings
import pytest
import torch

from artist import ARTIST_ROOT
from artist.core.kinematic_calibrator import KinematicCalibrator
from artist.core.loss_functions import FocalSpotLoss, Loss, VectorLoss
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, set_logger_config

# Set up logger.
set_logger_config()


@pytest.mark.parametrize(
    "calibration_method, initial_learning_rate, loss_class, data_source, early_stopping_delta, centroid_extraction_method",
    [
        (
            config_dictionary.kinematic_calibration_motor_positions,
            0.001,
            VectorLoss,
            "paint",
            1e-4,
            paint_mappings.UTIS_KEY,
        ),
        (
            config_dictionary.kinematic_calibration_raytracing,
            0.0001,
            FocalSpotLoss,
            "paint",
            1e-4,
            paint_mappings.UTIS_KEY,
        ),
        (
            config_dictionary.kinematic_calibration_motor_positions,
            0.0001,
            VectorLoss,
            "invalid",
            1e-4,
            paint_mappings.UTIS_KEY,
        ),
        (
            config_dictionary.kinematic_calibration_raytracing,
            0.0001,
            FocalSpotLoss,
            "invalid",
            1e-4,
            paint_mappings.UTIS_KEY,
        ),
        (
            config_dictionary.kinematic_calibration_motor_positions,
            0.0001,
            VectorLoss,
            "paint",
            1.0,
            paint_mappings.UTIS_KEY,
        ),
        (
            config_dictionary.kinematic_calibration_raytracing,
            0.0001,
            FocalSpotLoss,
            "paint",
            1.0,
            paint_mappings.UTIS_KEY,
        ),
        (
            config_dictionary.kinematic_calibration_raytracing,
            0.0001,
            FocalSpotLoss,
            "paint",
            1.0,
            "invalid",
        ),
    ],
)
def test_kinematic_calibrator(
    calibration_method: str,
    initial_learning_rate: float,
    loss_class: Loss,
    data_source: str,
    early_stopping_delta: float,
    centroid_extraction_method: str,
    ddp_setup_for_testing: dict[str, Any],
    device: torch.device,
) -> None:
    """
    Test the kinematic calibration methods.

    Parameters
    ----------
    calibration_method : str
        The name of the calibration method.
    initial_learning_rate : float
        The initial learning rate.
    loss_class : Loss
        The loss class.
    data_source : str
        The name of the data source.
    early_stopping_delta : float
        The minimum required improvement to prevent early stopping.
    centroid_extraction_method : str
        The method used to extract the focal spot centroids.
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
        config_dictionary.gamma: 0.9,
    }

    optimization_configuration = {
        config_dictionary.initial_learning_rate: initial_learning_rate,
        config_dictionary.tolerance: 0.0005,
        config_dictionary.max_epoch: 15,
        config_dictionary.log_step: 0,
        config_dictionary.early_stopping_delta: early_stopping_delta,
        config_dictionary.early_stopping_patience: 13,
        config_dictionary.scheduler: config_dictionary.exponential,
        config_dictionary.scheduler_parameters: scheduler_parameters,
    }

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

    data: dict[str, str | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]] = {
        config_dictionary.data_source: data_source,
        config_dictionary.heliostat_data_mapping: heliostat_data_mapping,
    }

    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file, device=device
        )

    ddp_setup_for_testing[config_dictionary.device] = device
    ddp_setup_for_testing[config_dictionary.groups_to_ranks_mapping] = {0: [0, 1]}

    # Create the kinematic optimizer.
    if centroid_extraction_method == "invalid":
        with pytest.raises(ValueError) as exc_info:
            _ = KinematicCalibrator(
                ddp_setup=ddp_setup_for_testing,
                scenario=scenario,
                data=data,
                optimization_configuration=optimization_configuration,
                calibration_method=calibration_method,
                centroid_extraction_method=centroid_extraction_method,
            )

            assert (
                f"The selected centroid extraction method {centroid_extraction_method} is not yet supported. Please use either {paint_mappings.UTIS_KEY} or {paint_mappings.HELIOS_KEY}!"
                in str(exc_info.value)
            )
    else:
        kinematic_calibrator = KinematicCalibrator(
            ddp_setup=ddp_setup_for_testing,
            scenario=scenario,
            data=data,
            optimization_configuration=optimization_configuration,
            calibration_method=calibration_method,
            centroid_extraction_method=centroid_extraction_method,
        )

        loss_definition = (
            FocalSpotLoss(scenario=scenario)
            if loss_class is FocalSpotLoss
            else VectorLoss()
        )

        # Calibrate the kinematic.
        if data_source == "invalid":
            with pytest.raises(ValueError) as exc_info:
                _ = kinematic_calibrator.calibrate(
                    loss_definition=loss_definition, device=device
                )

                assert (
                    f"There is no data loader for the data source: {data_source}. Please use PAINT data instead."
                    in str(exc_info.value)
                )
        else:
            _ = kinematic_calibrator.calibrate(
                loss_definition=loss_definition, device=device
            )

            for index, heliostat_group in enumerate(
                scenario.heliostat_field.heliostat_groups
            ):
                expected_path = (
                    pathlib.Path(ARTIST_ROOT)
                    / "tests/data/expected_optimized_kinematic_parameters"
                    / f"{calibration_method}_group_{index}_{device.type}.pt"
                )

                expected = torch.load(
                    expected_path, map_location=device, weights_only=True
                )

                torch.testing.assert_close(
                    heliostat_group.kinematic.deviation_parameters,
                    expected["kinematic_deviations"],
                    atol=5e-2,
                    rtol=5e-2,
                )
                torch.testing.assert_close(
                    heliostat_group.kinematic.actuators.actuator_parameters,
                    expected["actuator_parameters"],
                    atol=5e-2,
                    rtol=5e-2,
                )
