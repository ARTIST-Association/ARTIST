import pathlib
from typing import Callable

import h5py
import pytest
import torch

from artist import ARTIST_ROOT
from artist.core import loss_functions
from artist.core.surface_reconstructor import SurfaceReconstructor
from artist.scenario.scenario import Scenario
from artist.util.environment_setup import DistributedEnvironmentTypedDict


@pytest.mark.parametrize(
    "loss_function",
    [
        (loss_functions.distribution_loss_kl_divergence),
        (loss_functions.pixel_loss),
    ],
)
def test_surface_reconstructor(
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ddp_setup_for_testing: DistributedEnvironmentTypedDict,
    device: torch.device,
) -> None:
    """
    Test the surface reconstructor.

    Parameters
    ----------
    loss_function : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        A callable function that computes the loss. It accepts predictions and targets
        and optionally other keyword arguments and return a tensor with loss values.
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

    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file, device=device
        )

    ddp_setup_for_testing["device"] = device
    ddp_setup_for_testing["groups_to_ranks_mapping"] = {0: [0]}

    # Create the surface reconstructor.
    surface_reconstructor = SurfaceReconstructor(
        ddp_setup=ddp_setup_for_testing,
        scenario=scenario,
        heliostat_data_mapping=heliostat_data_mapping,
        max_epoch=2,
        num_log=1,
        device=device,
    )

    surface_reconstructor.reconstruct_surfaces(
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
            atol=5e-5,
            rtol=5e-5,
        )


def test_fixate_control_points_on_outer_edges(
    device: torch.device,
) -> None:
    """
    Test the outer control points fixation function.

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

    fixed_gradients = SurfaceReconstructor.fixate_control_points_on_outer_edges(
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
        fixed_gradients,
        expected_gradients,
        atol=5e-2,
        rtol=5e-2,
    )


def test_total_variation_loss(
    device: torch.device,
) -> None:
    """
    Test the total variation loss function.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    rows = torch.linspace(0, 4 * 3.1415, 120, device=device)
    columns = torch.linspace(0, 4 * 3.1415, 120, device=device)
    x, y = torch.meshgrid(rows, columns, indexing="ij")

    # Smooth surface with waves.
    z_values_smooth = 0.5 * torch.sin(x) + 0.5 * torch.cos(y)

    # Irregular surface = smooth surface with waves and random noise.
    noise = torch.randn_like(z_values_smooth, device=device) * 0.5
    z_irregular = z_values_smooth + noise

    coordinates_smooth = torch.stack(
        [x.flatten(), y.flatten(), z_values_smooth.flatten()], dim=1
    ).unsqueeze(0)
    coordinates_irregular = torch.stack(
        [x.flatten(), y.flatten(), z_irregular.flatten()], dim=1
    ).unsqueeze(0)

    surfaces = (
        torch.cat([coordinates_smooth, coordinates_irregular], dim=0)
        .unsqueeze(1)
        .expand(2, 4, -1, 3)
    )

    # Calculate total variation loss
    loss = SurfaceReconstructor.total_variation_loss_per_surface(
        surfaces=surfaces, number_of_neighbors=10, radius=0.5, sigma=1.0, device=device
    )

    torch.testing.assert_close(
        loss,
        torch.tensor(
            [
                [0.043646000326, 0.043646000326, 0.043646000326, 0.043646000326],
                [0.564661264420, 0.564661264420, 0.564661264420, 0.564661264420],
            ],
            device=device,
        ),
        atol=5e-2,
        rtol=5e-2,
    )
