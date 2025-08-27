import pathlib
from typing import Any, Callable

import h5py
import pytest
import torch

from artist import ARTIST_ROOT
from artist.core import loss_functions
from artist.core.surface_reconstructor import SurfaceReconstructor
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary


@pytest.mark.parametrize(
    "loss_function",
    [
        (loss_functions.distribution_loss_kl_divergence),
        (loss_functions.pixel_loss),
    ],
)
def test_surface_reconstructor(
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ddp_setup_for_testing: dict[str, Any],
    device: torch.device,
) -> None:
    """
    Test the surface reconstructor.

    Parameters
    ----------
    loss_function : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        A callable function that computes the loss. It accepts predictions and targets
        and optionally other keyword arguments and return a tensor with loss values.
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

    # Configure regularizers and their weights.
    ideal_surface_regularizer = {
        config_dictionary.regularization_callable: config_dictionary.vector_loss,
        config_dictionary.weight: 0.5,
        config_dictionary.regularizers_parameters: None,
    }
    total_variation_regularizer_points = {
        config_dictionary.regularization_callable: config_dictionary.total_variation_loss,
        config_dictionary.weight: 0.5,
        config_dictionary.regularizers_parameters: {
            config_dictionary.number_of_neighbors: 64,
            config_dictionary.sigma: 1e-3,
        },
    }
    total_variation_regularizer_normals = {
        config_dictionary.regularization_callable: config_dictionary.total_variation_loss,
        config_dictionary.weight: 0.5,
        config_dictionary.regularizers_parameters: {
            config_dictionary.number_of_neighbors: 64,
            config_dictionary.sigma: 1e-3,
        },
    }
    regularizers = {
        config_dictionary.ideal_surface_loss: ideal_surface_regularizer,
        config_dictionary.total_variation_loss_points: total_variation_regularizer_points,
        config_dictionary.total_variation_loss_normals: total_variation_regularizer_normals,
    }

    optimization_configuration = {
        config_dictionary.initial_learning_rate: 1e-4,
        config_dictionary.tolerance: 5e-4,
        config_dictionary.max_epoch: 15,
        config_dictionary.num_log: 1,
        config_dictionary.early_stopping_delta: 1e-4,
        config_dictionary.early_stopping_patience: 10,
        config_dictionary.scheduler: config_dictionary.exponential,
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

    data: dict[str, str | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]] = {
        config_dictionary.data_source: config_dictionary.paint,
        config_dictionary.heliostat_data_mapping: heliostat_data_mapping,
    }
    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file, device=device
        )

    ddp_setup_for_testing[config_dictionary.device] = device
    ddp_setup_for_testing[config_dictionary.groups_to_ranks_mapping] = {0: [0, 1]}

    # Create the surface reconstructor.
    surface_reconstructor = SurfaceReconstructor(
        ddp_setup=ddp_setup_for_testing,
        scenario=scenario,
        data=data,
        optimization_configuration=optimization_configuration,
        device=device,
    )

    _ = surface_reconstructor.reconstruct_surfaces(
        loss_function=loss_function, device=device
    )

    for index, heliostat_group in enumerate(scenario.heliostat_field.heliostat_groups):
        expected_path = (
            pathlib.Path(ARTIST_ROOT)
            / "tests/data/expected_reconstructed_surfaces"
            / f"{loss_function.__name__}_group_{index}_{device.type}.pt"
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
