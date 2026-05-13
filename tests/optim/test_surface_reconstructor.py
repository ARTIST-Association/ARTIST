import pathlib

import h5py
import pytest
import torch

from artist import ARTIST_ROOT
from artist.io import CalibrationDataParser, PaintCalibrationDataParser
from artist.optim import SurfaceReconstructor
from artist.optim.loss import KLDivergenceLoss, Loss, PixelLoss
from artist.scenario import Scenario
from artist.util import constants
from artist.util.env import DdpSetup


@pytest.mark.parametrize(
    "loss_class, early_stopping_window, data_parser, scheduler",
    [
        (
            KLDivergenceLoss,
            40,
            PaintCalibrationDataParser(),
            constants.reduce_on_plateau,
        ),
        (PixelLoss, 20, PaintCalibrationDataParser(), constants.cyclic),
        (PixelLoss, 10, CalibrationDataParser(), constants.cyclic),
    ],
)
def test_surface_reconstructor(
    loss_class: Loss,
    early_stopping_window: int,
    data_parser: CalibrationDataParser | PaintCalibrationDataParser,
    scheduler: str,
    ddp_setup_for_testing: DdpSetup,
    device: torch.device,
) -> None:
    """
    Test the surface reconstructor.

    Parameters
    ----------
    loss_class : Loss
        The loss class.
    early_stopping_window : int
        Number of epochs used to estimate loss trend.
    data_parser : CalibrationDataParser
        The data parser used to load calibration data from files.
    scheduler : str
        Scheduler name.
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

    optimizer_dict = {
        constants.initial_learning_rate: 1e-4,
        constants.tolerance: 5e-4,
        constants.max_epoch: 50,
        constants.batch_size: 30,
        constants.log_step: 0,
        constants.early_stopping_delta: 1.0,
        constants.early_stopping_patience: 2,
        constants.early_stopping_window: early_stopping_window,
    }
    scheduler_dict = {
        constants.scheduler_type: scheduler,
        constants.lr_min: 1e-6,
        constants.lr_max: 1e-3,
        constants.step_size_up: 500,
        constants.reduce_factor: 0.8,
        constants.patience: 10,
        constants.threshold: 1e-4,
        constants.cooldown: 5,
    }
    constraint_dict = {
        constants.rho_flux_integral: 1.0,
        constants.energy_tolerance: 0.01,
        constants.weight_smoothness: 0.005,
        constants.weight_ideal_surface: 0.005,
    }
    optimization_configuration = {
        constants.optimization: optimizer_dict,
        constants.scheduler: scheduler_dict,
        constants.constraints: constraint_dict,
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
                / "tests/data/field_data/AA31-calibration-properties_1.json",
                pathlib.Path(ARTIST_ROOT)
                / "tests/data/field_data/AA31-calibration-properties_2.json",
            ],
            [
                pathlib.Path(ARTIST_ROOT)
                / "tests/data/field_data/AA31-flux-centered_1.png",
                pathlib.Path(ARTIST_ROOT)
                / "tests/data/field_data/AA31-flux-centered_2.png",
            ],
        ),
    ]

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
            scenario_file=scenario_file,
            change_number_of_control_points_per_facet=torch.tensor(
                [7, 7], device=device
            ),
            device=device,
        )

    ddp_setup_for_testing["device"] = device

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
                / f"{loss_name}_group_{index}_{early_stopping_window}_{device.type}.pt"
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
