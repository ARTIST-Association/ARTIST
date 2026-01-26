import pathlib
from typing import Any

import h5py
import pytest
import torch

from artist import ARTIST_ROOT
from artist.core.loss_functions import KLDivergenceLoss, Loss, PixelLoss
from artist.core.regularizers import IdealSurfaceRegularizer, SmoothnessRegularizer
from artist.core.surface_reconstructor import SurfaceReconstructor
from artist.data_parser.calibration_data_parser import CalibrationDataParser
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary


@pytest.mark.parametrize(
    "loss_class, data_parser",
    [
        (KLDivergenceLoss, PaintCalibrationDataParser()),
        (PixelLoss, PaintCalibrationDataParser()),
        (PixelLoss, CalibrationDataParser()),
    ],
)
def test_surface_reconstructor(
    loss_class: Loss,
    data_parser: CalibrationDataParser | PaintCalibrationDataParser,
    ddp_setup_for_testing: dict[str, Any],
    device: torch.device,
) -> None:
    """
    Test the surface reconstructor.

    Parameters
    ----------
    loss_class : Loss
        The loss class.
    data_parser : CalibrationDataParser
        The data parser used to load calibration data from files.
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
        config_dictionary.min: 1e-6,
        config_dictionary.reduce_factor: 0.8,
        config_dictionary.patience: 10,
        config_dictionary.threshold: 1e-4,
        config_dictionary.cooldown: 5,
    }

    # Configure regularizers and their weights.
    ideal_surface_regularizer = IdealSurfaceRegularizer(
        weight=1.0, reduction_dimensions=(1,)
    )
    smoothness_regularizer = SmoothnessRegularizer(
        weight=1.0, reduction_dimensions=(1,)
    )

    regularizers = [
        ideal_surface_regularizer,
        smoothness_regularizer,
    ]

    optimization_configuration = {
        config_dictionary.initial_learning_rate: 1e-4,
        config_dictionary.tolerance: 5e-4,
        config_dictionary.max_epoch: 15,
        config_dictionary.batch_size: 30,
        config_dictionary.log_step: 0,
        config_dictionary.early_stopping_delta: 1e-4,
        config_dictionary.early_stopping_patience: 20,
        config_dictionary.early_stopping_window: 10,
        config_dictionary.scheduler: config_dictionary.reduce_on_plateau,
        config_dictionary.scheduler_parameters: scheduler_parameters,
        config_dictionary.regularizers: regularizers,
    }

    scenario_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/data/scenarios/test_scenario_paint_four_heliostats_ideal.h5"
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
                / "tests/data/field_data/AA39-flux-centered_1.png",
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

    data: dict[
        str,
        CalibrationDataParser
        | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    ] = {
        config_dictionary.data_parser: data_parser,
        config_dictionary.heliostat_data_mapping: heliostat_data_mapping,
    }

    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file, device=device
        )

    ddp_setup_for_testing[config_dictionary.device] = device
    ddp_setup_for_testing[config_dictionary.groups_to_ranks_mapping] = {0: [0, 1]}
    ddp_setup_for_testing[config_dictionary.ranks_to_groups_mapping] = {
        0: [0],
        1: [0],
    }

    # Create the surface reconstructor.
    surface_reconstructor = SurfaceReconstructor(
        ddp_setup=ddp_setup_for_testing,
        scenario=scenario,
        data=data,
        optimization_configuration=optimization_configuration,
        device=device,
    )

    loss_definition = (
        PixelLoss(scenario=scenario) if loss_class is PixelLoss else KLDivergenceLoss()
    )

    if not isinstance(data_parser, PaintCalibrationDataParser):
        with pytest.raises(NotImplementedError) as exc_info:
            _ = surface_reconstructor.reconstruct_surfaces(
                loss_definition=loss_definition, device=device
            )

            assert "Must be overridden!" in str(exc_info.value)
    else:
        old_state = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)
        try:
            _ = surface_reconstructor.reconstruct_surfaces(
                loss_definition=loss_definition, device=device
            )
        finally:
            torch.use_deterministic_algorithms(old_state)

        for index, heliostat_group in enumerate(
            scenario.heliostat_field.heliostat_groups
        ):
            loss_name = "pixel_loss" if loss_class is PixelLoss else "kl_divergence"
            expected_path = (
                pathlib.Path(ARTIST_ROOT)
                / "tests/data/expected_reconstructed_surfaces"
                / f"{loss_name}_group_{index}_{device.type}.pt"
            )

            expected = torch.load(expected_path, map_location=device, weights_only=True)

            torch.testing.assert_close(
                heliostat_group.nurbs_control_points,
                expected,
                atol=5e-3,
                rtol=5e-3,
            )


def test_lock_control_points_on_outer_edges(
    device: torch.device,
) -> None:
    """
    Test the outer control points lock function.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    test_gradients = torch.zeros((1, 1, 4, 4, 3), device=device)
    origin_offsets_e = torch.linspace(-5, 5, test_gradients.shape[2], device=device)
    origin_offsets_n = torch.linspace(-5, 5, test_gradients.shape[3], device=device)

    test_gradients_e, test_gradients_n = torch.meshgrid(
        origin_offsets_e, origin_offsets_n, indexing="ij"
    )

    test_gradients[:, :, :, :, 0] = test_gradients_e
    test_gradients[:, :, :, :, 1] = test_gradients_n
    test_gradients[:, :, :, :, 2] = torch.full_like(test_gradients_e, 5, device=device)

    locked_gradients = SurfaceReconstructor.lock_control_points_on_outer_edges(
        gradients=test_gradients, device=device
    )

    expected_gradients = torch.tensor(
        [
            [
                [
                    [
                        [0.0000, 0.0000, 5.0000],
                        [0.0000, 0.0000, 5.0000],
                        [0.0000, 0.0000, 5.0000],
                        [0.0000, 0.0000, 5.0000],
                    ],
                    [
                        [0.0000, 0.0000, 5.0000],
                        [-1.6667, -1.6667, 5.0000],
                        [-1.6667, 1.6667, 5.0000],
                        [0.0000, 0.0000, 5.0000],
                    ],
                    [
                        [0.0000, 0.0000, 5.0000],
                        [1.6667, -1.6667, 5.0000],
                        [1.6667, 1.6667, 5.0000],
                        [0.0000, 0.0000, 5.0000],
                    ],
                    [
                        [0.0000, 0.0000, 5.0000],
                        [0.0000, 0.0000, 5.0000],
                        [0.0000, 0.0000, 5.0000],
                        [0.0000, 0.0000, 5.0000],
                    ],
                ]
            ]
        ],
        device=device,
    )

    torch.testing.assert_close(
        locked_gradients,
        expected_gradients,
        atol=5e-2,
        rtol=5e-2,
    )
