import pathlib

import h5py
import paint.util.paint_mappings as paint_mappings
import pytest
import torch

from artist import ARTIST_ROOT
from artist.io import CalibrationDataParser, PaintCalibrationDataParser
from artist.optim import KinematicsReconstructor
from artist.optim.loss import FocalSpotLoss
from artist.scenario import Scenario
from artist.util import constants
from artist.util.env import DdpSetup


@pytest.mark.parametrize(
    "reconstruction_method, data_parser, centroid_extraction_method, early_stopping_window, scheduler",
    [
        # Test normal behavior.
        (
            constants.kinematics_reconstruction_raytracing,
            PaintCalibrationDataParser(),
            paint_mappings.UTIS_KEY,
            50,
            constants.exponential,
        ),
        # Test early stopping.
        (
            constants.kinematics_reconstruction_raytracing,
            PaintCalibrationDataParser(),
            paint_mappings.UTIS_KEY,
            10,
            constants.reduce_on_plateau,
        ),
        # Test invalid centroid extraction.
        (
            constants.kinematics_reconstruction_raytracing,
            PaintCalibrationDataParser(),
            "invalid",
            10,
            constants.reduce_on_plateau,
        ),
        # Test invalid reconstruction method.
        (
            "invalid",
            PaintCalibrationDataParser(),
            paint_mappings.UTIS_KEY,
            10,
            constants.reduce_on_plateau,
        ),
        # Test invalid parser.
        (
            constants.kinematics_reconstruction_raytracing,
            CalibrationDataParser(),
            paint_mappings.UTIS_KEY,
            10,
            constants.reduce_on_plateau,
        ),
    ],
)
def test_kinematics_reconstructor(
    reconstruction_method: str,
    data_parser: CalibrationDataParser,
    centroid_extraction_method: str,
    early_stopping_window: int,
    scheduler: str,
    ddp_setup_for_testing: DdpSetup,
    device: torch.device,
) -> None:
    """
    Test the kinematics reconstruction methods.

    Parameters
    ----------
    reconstruction_method : str
        The name of the reconstruction method.
    data_parser : CalibrationDataParser
        The data parser used to load calibration data from files.
    centroid_extraction_method : str
        The method used to extract the focal spot centroids.
    early_stopping_window : int
        Early stopping window size.
    scheduler : str
        The scheduler to be used.
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
        constants.gamma: 0.99,
        constants.lr_min: 1e-4,
        constants.reduce_factor: 0.9,
        constants.patience: 100,
        constants.threshold: 1e-3,
        constants.cooldown: 20,
    }
    optimizer_dict = {
        constants.initial_learning_rate_rotation_deviation: 1e-4,
        constants.initial_learning_rate_initial_angles: 1e-3,
        constants.initial_learning_rate_initial_stroke_length: 1e-2,
        constants.tolerance: 0.0005,
        constants.max_epoch: 250,
        constants.batch_size: 50,
        constants.log_step: 1,
        constants.early_stopping_delta: 1.0,
        constants.early_stopping_patience: 2,
        constants.early_stopping_window: early_stopping_window,
    }
    optimization_configuration = {
        constants.optimization: optimizer_dict,
        constants.scheduler: scheduler_dict,
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
                pathlib.Path(ARTIST_ROOT) / "tests/data/field_data/AA39-flux_1.png",
                pathlib.Path(ARTIST_ROOT) / "tests/data/field_data/AA39-flux_2.png",
            ],
        ),
        (
            "AA31",
            [
                pathlib.Path(ARTIST_ROOT)
                / "tests/data/field_data/AA31-calibration-properties_1.json",
                pathlib.Path(ARTIST_ROOT)
                / "tests/data/field_data/AA31-calibration-properties_2.json",
            ],
            [
                pathlib.Path(ARTIST_ROOT) / "tests/data/field_data/AA31-flux_1.png",
                pathlib.Path(ARTIST_ROOT) / "tests/data/field_data/AA31-flux_2.png",
            ],
        ),
    ]

    if centroid_extraction_method == "invalid":
        with pytest.raises(ValueError) as exc_info:
            data_parser = (
                PaintCalibrationDataParser(
                    centroid_extraction_method=centroid_extraction_method
                )
                if isinstance(data_parser, PaintCalibrationDataParser)
                else CalibrationDataParser()
            )
            assert (
                f"The selected centroid extraction method {centroid_extraction_method} is not yet supported. Please use either {paint_mappings.UTIS_KEY} or {paint_mappings.HELIOS_KEY}!"
                in str(exc_info.value)
            )
    else:
        data: dict[
            str,
            CalibrationDataParser
            | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
        ] = {
            constants.data_parser: data_parser,
            constants.heliostat_data_mapping: heliostat_data_mapping,
        }

        with h5py.File(scenario_path, "r") as scenario_file:
            scenario = Scenario.load_scenario_from_hdf5(
                scenario_file=scenario_file, device=device
            )

        ddp_setup_for_testing["device"] = device

        if reconstruction_method == "invalid":
            with pytest.raises(ValueError) as exc_info:
                _ = KinematicsReconstructor(
                    ddp_setup=ddp_setup_for_testing,
                    scenario=scenario,
                    data=data,
                    optimization_configuration=optimization_configuration,
                    reconstruction_method=reconstruction_method,
                )
                assert (
                    f"ARTIST currently only supports the {constants.kinematics_reconstruction_raytracing} reconstruction method. The reconstruction method {reconstruction_method} is not recognized. Please select another reconstruction method and try again!"
                    in str(exc_info.value)
                )
        else:
            kinematics_reconstructor = KinematicsReconstructor(
                ddp_setup=ddp_setup_for_testing,
                scenario=scenario,
                data=data,
                optimization_configuration=optimization_configuration,
                reconstruction_method=reconstruction_method,
            )

            loss_definition = FocalSpotLoss(scenario=scenario)

            # Reconstruct the kinematics.
            if not isinstance(data_parser, PaintCalibrationDataParser):
                with pytest.raises(NotImplementedError) as exc_info:
                    _ = kinematics_reconstructor.reconstruct_kinematics(
                        loss_definition=loss_definition, device=device
                    )

                    assert "Must be overridden!" in str(exc_info.value)
            else:
                _ = kinematics_reconstructor.reconstruct_kinematics(
                    loss_definition=loss_definition, device=device
                )

                for index, heliostat_group in enumerate(
                    scenario.heliostat_field.heliostat_groups
                ):
                    expected_path = (
                        pathlib.Path(ARTIST_ROOT)
                        / "tests/data/expected_reconstructed_kinematics_parameters"
                        / f"group_{index}_{early_stopping_window}_{device.type}.pt"
                    )

                    expected = torch.load(
                        expected_path, map_location=device, weights_only=True
                    )

                    tol = 1e-6 + 5e-3 * torch.maximum(
                        heliostat_group.kinematics.rotation_deviation_parameters.abs(),
                        expected["rotation_deviations"].abs(),
                    )
                    diff = (
                        heliostat_group.kinematics.rotation_deviation_parameters
                        - expected["rotation_deviations"]
                    ).abs()

                    assert torch.all(diff <= tol)

                    tol = 1e-6 + 5e-3 * torch.maximum(
                        heliostat_group.kinematics.actuators.optimizable_parameters.abs(),
                        expected["optimizable_parameters"].abs(),
                    )
                    diff = (
                        heliostat_group.kinematics.actuators.optimizable_parameters
                        - expected["optimizable_parameters"]
                    ).abs()

                    assert torch.all(diff <= tol)
