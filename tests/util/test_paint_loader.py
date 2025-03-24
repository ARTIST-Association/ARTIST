import pathlib
from typing import Any

import pytest
import torch

from artist import ARTIST_ROOT
from artist.util import paint_loader
from artist.util.configuration_classes import (
    HeliostatListConfig,
    PowerPlantConfig,
    PrototypeConfig,
    TargetAreaListConfig,
)


@pytest.mark.parametrize(
    "file_path, power_plant_position, expected_list",
    [
        (
            [
                pathlib.Path(ARTIST_ROOT)
                / "tests/data/field_data/AA39-calibration-properties.json"
            ],
            torch.tensor([50.91342112259258, 6.387824755874856, 87.0]),
            [
                ["multi_focus_tower"],
                torch.tensor(
                    [[-17.403167724609, -2.928076744080, 50.741085052490, 1.0]]
                ),
                torch.tensor([[-0.721040308475, -0.524403691292, 0.452881515026, 1.0]]),
                torch.tensor([[26370.0, 68271.0]]),
            ],
        )
    ],
)
def test_extract_paint_calibration_data(
    file_path: list[pathlib.Path],
    power_plant_position: torch.Tensor,
    expected_list: list[torch.Tensor],
    device: torch.device,
) -> None:
    """
    Test the function to extract calibration data from ``PAINT`` calibration data.

    Parameters
    ----------
    file_path : list[pathlib.Path]
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
            calibration_properties_paths=file_path,
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
            for actual_element, expected_element in zip(actual, expected):
                assert actual_element == expected_element


@pytest.mark.parametrize(
    "file_path, expected_list",
    [
        (
            pathlib.Path(ARTIST_ROOT) / "tests/data/field_data/tower-measurements.json",
            [PowerPlantConfig, TargetAreaListConfig],
        )
    ],
)
def test_extract_paint_tower_measurements(
    file_path: pathlib.Path,
    expected_list: list[Any],
    device: torch.device,
) -> None:
    """
    Test the tower measurement loader for ``PAINT`` data.

    Parameters
    ----------
    file_path : pathlib.Path
        The path to the tower file.
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
            tower_measurements_path=file_path, device=device
        )
    )

    for actual, expected in zip(extracted_list, expected_list):
        assert isinstance(actual, expected)


@pytest.mark.parametrize(
    "heliostat_and_deflectometry_paths, power_plant_position, aim_point, max_epochs_for_surface_training, expected_list",
    [
        (
            [
                (
                    "heliostat_1",
                    pathlib.Path(ARTIST_ROOT)
                    / "tests/data/field_data/AA39-heliostat-properties.json",
                    pathlib.Path(ARTIST_ROOT)
                    / "tests/data/field_data/AA39-deflectometry.h5",
                )
            ],
            torch.tensor([50.91342112259258, 6.387824755874856, 87.0]),
            torch.tensor(
                [3.860326111317e-02, -5.029551386833e-01, 5.522674942017e01, 1.0]
            ),
            2,
            [HeliostatListConfig, PrototypeConfig],
        )
    ],
)
def test_extract_paint_heliostats(
    heliostat_and_deflectometry_paths: list[tuple[str, pathlib.Path, pathlib.Path]],
    power_plant_position: torch.Tensor,
    aim_point: torch.Tensor,
    max_epochs_for_surface_training: int,
    expected_list: list[Any],
    device: torch.device,
) -> None:
    """
    Test the heliostat properties loader for ``PAINT`` data.

    Parameters
    ----------
    heliostat_and_deflectometry_paths : tuple[str, pathlib.Path, pathlib.Path]
        Name of the heliostat and a pair of heliostat properties and deflectometry file paths.
    power_plant_position : torch.Tensor
        The position of the power plant in latitude, longitude and elevation.
    aim_point : torch.Tensor
        The default aim point for the heliostats (Should ideally be on a receiver).
    max_epochs_for_surface_training : int
        The maximum amount of epochs for fitting the NURBS.
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
        paint_loader.extract_paint_heliostats(
            heliostat_and_deflectometry_paths=heliostat_and_deflectometry_paths,
            power_plant_position=power_plant_position.to(device),
            aim_point=aim_point.to(device),
            max_epochs_for_surface_training=max_epochs_for_surface_training,
            device=device,
        )
    )

    for actual, expected in zip(extracted_list, expected_list):
        assert isinstance(actual, expected)
