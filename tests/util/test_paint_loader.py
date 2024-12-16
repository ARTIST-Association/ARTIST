import pathlib
from typing import Any

import pytest
import torch

from artist import ARTIST_ROOT
from artist.util import paint_loader
from artist.util.configuration_classes import ActuatorListConfig, KinematicConfig


@pytest.mark.parametrize(
    "file_path, expected",
    [
        (
            pathlib.Path(ARTIST_ROOT) / "tests/data/calibration_properties.json",
            "multi_focus_tower",
        )
    ],
)
def test_extract_paint_calibration_target_name(
    file_path: pathlib.Path,
    expected: str,
) -> None:
    """
    Test the function to load the calibration target name from ``PAINT`` calibration data.

    Parameters
    ----------
    file_path : pathlib.Path
        The path to the calibration file.
    expected : str
        The expected extracted data.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    calibration_target_name = paint_loader.extract_paint_calibration_target_name(
        calibration_properties_path=file_path
    )

    assert calibration_target_name == expected


@pytest.mark.parametrize(
    "file_path, power_plant_position, expected_list",
    [
        (
            pathlib.Path(ARTIST_ROOT) / "tests/data/calibration_properties.json",
            torch.tensor([50.91342112259258, 6.387824755874856, 87.0]),
            [
                torch.tensor(
                    [-18.332265853882, -2.928076744080, 52.565364837646, 1.000000000000]
                ),
                torch.tensor(
                    [-0.596934497356, -0.437113255262, 0.672756373882, 0.000000000000]
                ),
                torch.tensor([24282, 43957]),
            ],
        )
    ],
)
def test_extract_paint_calibration_data(
    file_path: pathlib.Path,
    power_plant_position: torch.Tensor,
    expected_list: list[torch.Tensor],
    device: torch.device,
) -> None:
    """
    Test the functino to extract calibration data from ``PAINT`` calibration data.

    Parameters
    ----------
    file_path : pathlib.Path
        The path to the calibration file.
    power_plant_position : torch.Tensor
        The power plant position.
    expected_list : list[torch.Tensor]
        The expected extracted data.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    extracted_list = list(
        paint_loader.extract_paint_calibration_data(
            calibration_properties_path=file_path,
            power_plant_position=power_plant_position.to(device),
            device=device,
        )
    )

    for actual, expected in zip(extracted_list, expected_list):
        torch.testing.assert_close(actual, expected.to(device), atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize(
    "file_path, target_name, expected_list",
    [
        (
            pathlib.Path(ARTIST_ROOT) / "tests/data/tower.json",
            "multi_focus_tower",
            [
                torch.tensor(
                    [50.913421122593, 6.387824755875, 87.000000000000],
                    dtype=torch.float64,
                ),
                "planar",
                torch.tensor(
                    [-17.604515075684, -2.744643926620, 51.979751586914, 1.000000000000]
                ),
                torch.tensor([0, 1, 0, 0]),
                torch.tensor(5.411863327026),
                torch.tensor(6.387498855591),
            ],
        )
    ],
)
def test_extract_paint_tower_measurements(
    file_path: pathlib.Path,
    target_name: str,
    expected_list: list[Any],
    device: torch.device,
) -> None:
    """
    Test the tower measurement loader for ``PAINT`` data.

    Parameters
    ----------
    file_path : pathlib.Path
        The path to the tower file.
    target_name : str
        The name of the target on the tower.
    expected_list : list[Any]
        The expected extracted data.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    extracted_list = list(
        paint_loader.extract_paint_tower_measurements(
            tower_measurements_path=file_path, target_name=target_name, device=device
        )
    )

    for actual, expected in zip(extracted_list, expected_list):
        if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
            torch.testing.assert_close(
                actual, expected.to(device), atol=5e-4, rtol=5e-4
            )
        else:
            assert actual == expected


@pytest.mark.parametrize(
    "file_path, power_plant_position, expected_list",
    [
        (
            pathlib.Path(ARTIST_ROOT) / "tests/data/heliostat_properties.json",
            torch.tensor([50.91342112259258, 6.387824755874856, 87.0]),
            [
                torch.tensor(
                    [11.738112449646, 24.784408569336, 1.794999957085, 1.000000000000]
                ),
                KinematicConfig,
                ActuatorListConfig,
            ],
        )
    ],
)
def test_extract_paint_heliostat_properties(
    file_path: pathlib.Path,
    power_plant_position: torch.Tensor,
    expected_list: list[Any],
    device: torch.device,
) -> None:
    """
    Test the heliostat properties loader for ``PAINT`` data.

    Parameters
    ----------
    file_path : pathlib.Path
        The path to the heliostat file.
    power_plant_position : torch.Tensor
        The power plant position.
    expected_list : list[Any]
        The expected extracted data.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    extracted_list = list(
        paint_loader.extract_paint_heliostat_properties(
            heliostat_properties_path=file_path,
            power_plant_position=power_plant_position.to(device),
            device=device,
        )
    )

    for actual, expected in zip(extracted_list, expected_list):
        if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
            torch.testing.assert_close(
                actual, expected.to(device), atol=5e-4, rtol=5e-4
            )
        else:
            assert isinstance(actual, expected)
