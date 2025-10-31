import pathlib
from typing import Any

import paint.util.paint_mappings as paint_mappings
import pytest
import torch
from _pytest.monkeypatch import MonkeyPatch

from artist import ARTIST_ROOT
from artist.data_parser import paint_scenario_parser
from artist.scenario.configuration_classes import (
    HeliostatListConfig,
    PowerPlantConfig,
    PrototypeConfig,
    TargetAreaListConfig,
)

torch.manual_seed(7)
torch.cuda.manual_seed(7)


@pytest.mark.parametrize(
    "file_path, expected_types, expected_power_plant_position, expected_receiver_properties",
    [
        (
            pathlib.Path(ARTIST_ROOT) / "tests/data/field_data/tower-measurements.json",
            [PowerPlantConfig, TargetAreaListConfig],
            torch.tensor(
                [50.913421122593, 6.387824755875, 87.000000000000], dtype=torch.float64
            ),
            [
                "receiver",
                "convex_cylinder",
                torch.tensor(
                    [
                        3.860326111317e-02,
                        -5.029551386833e-01,
                        5.522674942017e01,
                        1.000000000000e00,
                    ]
                ),
                torch.tensor(
                    [[0.000000000000, 0.906307816505, -0.422618269920, 0.000000000000]]
                ),
                torch.tensor(4.528313636780),
                torch.tensor(5.218500137329),
            ],
        )
    ],
)
def test_extract_paint_tower_measurements(
    file_path: pathlib.Path,
    expected_types: list[Any],
    expected_power_plant_position: torch.Tensor,
    expected_receiver_properties: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test the tower measurement loader for ``PAINT`` data.

    Parameters
    ----------
    file_path : pathlib.Path
        The path to the tower file.
    expected_types : list[Any]
        The expected extracted data types.
    expected_power_plant_position : torch.Tensor
        The expected power plant position.
    expected_receiver_properties : torch.Tensor
        The expected receiver properties.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    extracted_list = list(
        paint_scenario_parser.extract_paint_tower_measurements(
            tower_measurements_path=file_path, device=device
        )
    )

    assert isinstance(extracted_list[0], expected_types[0])
    assert isinstance(extracted_list[1], expected_types[1])

    torch.testing.assert_close(
        extracted_list[0].power_plant_position, expected_power_plant_position.to(device)
    )
    assert (
        extracted_list[1].target_area_list[3].target_area_key
        == expected_receiver_properties[0]
    )
    assert (
        extracted_list[1].target_area_list[3].geometry
        == expected_receiver_properties[1]
    )
    torch.testing.assert_close(
        extracted_list[1].target_area_list[3].center,
        expected_receiver_properties[2].to(device),
    )
    torch.testing.assert_close(
        extracted_list[1].target_area_list[3].normal_vector,
        expected_receiver_properties[3].to(device),
    )
    torch.testing.assert_close(
        extracted_list[1].target_area_list[3].plane_e,
        expected_receiver_properties[4].to(device),
    )
    torch.testing.assert_close(
        extracted_list[1].target_area_list[3].plane_u,
        expected_receiver_properties[5].to(device),
    )


@pytest.mark.parametrize(
    "heliostat_paths, power_plant_position, expected_types, expected_heliostat",
    [
        (
            [
                (
                    "heliostat_1",
                    pathlib.Path(ARTIST_ROOT)
                    / "tests/data/field_data/AA39-heliostat-properties.json",
                )
            ],
            torch.tensor([50.91342112259258, 6.387824755874856, 87.0]),
            [HeliostatListConfig, PrototypeConfig],
            [
                torch.tensor(
                    [11.664672851562, 24.570718765259, 1.688941955566, 1.000000000000]
                ),
                2,
                "linear",
                torch.tensor(154166.671875000000),
                torch.tensor([0, 0, 1, 0]),
                torch.tensor(
                    [
                        [
                            8.024845719337e-01,
                            0.000000000000e00,
                            -4.984567873180e-03,
                            0.000000000000e00,
                        ],
                        [
                            1.956921187229e-05,
                            6.374921798706e-01,
                            3.150522708893e-03,
                            0.000000000000e00,
                        ],
                    ]
                ),
                torch.tensor([-0.802500069141, -0.436184167862,  0.000000000000])
            ],
        ),
    ],
)
def test_extract_paint_heliostats_ideal_surface(
    heliostat_paths: list[tuple[str, pathlib.Path]],
    power_plant_position: torch.Tensor,
    expected_types: list[Any],
    expected_heliostat: list[Any],
    device: torch.device,
) -> None:
    """
    Test the heliostat extraction for ``PAINT`` data.

    Parameters
    ----------
    heliostat_paths : tuple[str, pathlib.Path]
        Name of the heliostat and a heliostat properties file path.
    power_plant_position : torch.Tensor
        The position of the power plant in latitude, longitude and elevation.
    expected_types : list[Any]
        The expected extracted data types.
    expected_heliostat : list[Union[torch.Tensor, int, str]],
        The expected extracted heliostat data.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    extracted_list = list(
        paint_scenario_parser.extract_paint_heliostats_ideal_surface(
            paths=heliostat_paths,
            power_plant_position=power_plant_position.to(device),
            number_of_nurbs_control_points=torch.tensor([20, 20], device=device),
            device=device,
        )
    )

    assert isinstance(extracted_list[0], expected_types[0])
    assert isinstance(extracted_list[1], expected_types[1])

    assert extracted_list[0].heliostat_list[0].name == heliostat_paths[0][0]
    torch.testing.assert_close(
        extracted_list[0].heliostat_list[0].position, expected_heliostat[0].to(device)
    )
    assert (
        len(extracted_list[0].heliostat_list[0].actuators.actuator_list)
        == expected_heliostat[1]
    )
    assert (
        extracted_list[0].heliostat_list[0].actuators.actuator_list[0].type
        == expected_heliostat[2]
    )
    torch.testing.assert_close(
        extracted_list[0]
        .heliostat_list[0]
        .actuators.actuator_list[0]
        .parameters.increment,
        expected_heliostat[3].to(device),
    )
    torch.testing.assert_close(
        extracted_list[0].heliostat_list[0].kinematic.initial_orientation,
        expected_heliostat[4].to(device),
    )
    torch.testing.assert_close(
        extracted_list[0].heliostat_list[0].surface.facet_list[0].canting,
        expected_heliostat[5].to(device),
    )
    torch.testing.assert_close(
        extracted_list[0].heliostat_list[0].surface.facet_list[0].control_points[0, 3],
        expected_heliostat[6].to(device),
    )


@pytest.mark.parametrize(
    "heliostat_and_deflectometry_paths, power_plant_position, max_epochs_for_surface_training, expected_types, expected_heliostat",
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
            2,
            [HeliostatListConfig, PrototypeConfig],
            [
                torch.tensor(
                    [11.664672851562, 24.570718765259, 1.688941955566, 1.000000000000]
                ),
                2,
                "linear",
                torch.tensor(154166.671875000000),
                torch.tensor([0, 0, 1, 0]),
                torch.tensor(
                    [
                        [
                            8.024845719337e-01,
                            0.000000000000e00,
                            -4.984567873180e-03,
                            0.000000000000e00,
                        ],
                        [
                            1.956921187229e-05,
                            6.374921798706e-01,
                            3.150522708893e-03,
                            0.000000000000e00,
                        ],
                    ]
                ),
                torch.tensor([-1.606200933456, 0.212735980749, 0.040915220976]),
            ],
        ),
    ],
)
def test_extract_paint_heliostats_fitted_surface(
    heliostat_and_deflectometry_paths: list[tuple[str, pathlib.Path, pathlib.Path]],
    power_plant_position: torch.Tensor,
    max_epochs_for_surface_training: int,
    expected_types: list[Any],
    expected_heliostat: list[Any],
    device: torch.device,
) -> None:
    """
    Test the heliostat extraction for ``PAINT`` data with fitted surfaces.

    Parameters
    ----------
    heliostat_and_deflectometry_paths : tuple[str, pathlib.Path, pathlib.Path]
        Name of the heliostat and a pair of heliostat properties and deflectometry file paths.
    power_plant_position : torch.Tensor
        The position of the power plant in latitude, longitude and elevation.
    max_epochs_for_surface_training : int
        The maximum amount of epochs for fitting the NURBS.
    expected_types : list[Any]
        The expected extracted data types.
    expected_heliostat : list[torch.Tensor| int| str],
        The expected extracted heliostat data.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    optimizer = torch.optim.Adam([torch.empty(1, requires_grad=True)], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.2,
        patience=50,
        threshold=1e-7,
        threshold_mode="abs",
    )

    extracted_list = list(
        paint_scenario_parser.extract_paint_heliostats_fitted_surface(
            paths=heliostat_and_deflectometry_paths,
            power_plant_position=power_plant_position.to(device),
            number_of_nurbs_control_points=torch.tensor([20, 20], device=device),
            nurbs_fit_max_epoch=max_epochs_for_surface_training,
            nurbs_fit_optimizer=optimizer,
            nurbs_fit_scheduler=scheduler,
            device=device,
        )
    )

    assert isinstance(extracted_list[0], expected_types[0])
    assert isinstance(extracted_list[1], expected_types[1])

    assert (
        extracted_list[0].heliostat_list[0].name
        == heliostat_and_deflectometry_paths[0][0]
    )
    torch.testing.assert_close(
        extracted_list[0].heliostat_list[0].position, expected_heliostat[0].to(device)
    )
    assert (
        len(extracted_list[0].heliostat_list[0].actuators.actuator_list)
        == expected_heliostat[1]
    )
    assert (
        extracted_list[0].heliostat_list[0].actuators.actuator_list[0].type
        == expected_heliostat[2]
    )
    torch.testing.assert_close(
        extracted_list[0]
        .heliostat_list[0]
        .actuators.actuator_list[0]
        .parameters.increment,
        expected_heliostat[3].to(device),
    )
    torch.testing.assert_close(
        extracted_list[0].heliostat_list[0].kinematic.initial_orientation,
        expected_heliostat[4].to(device),
    )
    torch.testing.assert_close(
        extracted_list[0].heliostat_list[0].surface.facet_list[0].canting,
        expected_heliostat[5].to(device),
    )
    torch.testing.assert_close(
        extracted_list[0].heliostat_list[0].surface.facet_list[0].control_points[0, 3],
        expected_heliostat[6].to(device),
    )


@pytest.mark.parametrize(
    "heliostat_paths, power_plant_position, max_epochs_for_surface_training, expected_types, expected_heliostat_ideal, expected_heliostat_fitted",
    [
        (
            [
                # Ideal heliostat
                (
                    "ideal_heliostat",
                    pathlib.Path(ARTIST_ROOT)
                    / "tests/data/field_data/AA39-heliostat-properties.json",
                ),
                # Fitted heliostat
                (
                    "fitted_heliostat",
                    pathlib.Path(ARTIST_ROOT)
                    / "tests/data/field_data/AA39-heliostat-properties.json",
                    pathlib.Path(ARTIST_ROOT)
                    / "tests/data/field_data/AA39-deflectometry.h5",
                ),
            ],
            torch.tensor([50.91342112259258, 6.387824755874856, 87.0]),
            2,
            [HeliostatListConfig, PrototypeConfig],
            [
                torch.tensor(
                    [11.664672851562, 24.570718765259, 1.688941955566, 1.000000000000]
                ),
                2,
                "linear",
                torch.tensor(154166.671875000000),
                torch.tensor([0, 0, 1, 0]),
                torch.tensor(
                    [
                        [
                            8.024845719337e-01,
                            0.000000000000e00,
                            -4.984567873180e-03,
                            0.000000000000e00,
                        ],
                        [
                            1.956921187229e-05,
                            6.374921798706e-01,
                            3.150522708893e-03,
                            0.000000000000e00,
                        ],
                    ]
                ),
                torch.tensor([-0.802500069141, -0.436184167862,  0.000000000000])
            ],
            [
                torch.tensor(
                    [11.664672851562, 24.570718765259, 1.688941955566, 1.000000000000]
                ),
                2,
                "linear",
                torch.tensor(154166.671875000000),
                torch.tensor([0, 0, 1, 0]),
                torch.tensor(
                    [
                        [
                            8.024845719337e-01,
                            0.000000000000e00,
                            -4.984567873180e-03,
                            0.000000000000e00,
                        ],
                        [
                            1.956921187229e-05,
                            6.374921798706e-01,
                            3.150522708893e-03,
                            0.000000000000e00,
                        ],
                    ]
                ),
                torch.tensor([-1.606200933456,  0.212735980749,  0.040915220976])
            ],
        ),
    ],
)
def test_extract_paint_heliostats_mixed_surface(
    heliostat_paths: (
        list[tuple[str, pathlib.Path]] | list[tuple[str, pathlib.Path, pathlib.Path]]
    ),
    power_plant_position: torch.Tensor,
    max_epochs_for_surface_training: int,
    expected_types: list[Any],
    expected_heliostat_ideal: list[Any],
    expected_heliostat_fitted: list[Any],
    device: torch.device,
) -> None:
    """
    Test the heliostat extraction for a mixed scenario of ideal and fitted surfaces.

    Parameters
    ----------
    heliostat_paths : list[tuple[str, pathlib.Path]] | list[tuple[str, pathlib.Path, pathlib.Path]]
        A list of tuples, each containing the heliostat name, properties file path,
        and optionally a deflectometry file path (None for ideal surfaces).
    power_plant_position : torch.Tensor
        The position of the power plant in latitude, longitude and elevation.
    max_epochs_for_surface_training : int
        The maximum number of epochs for fitting the NURBS.
    expected_types : list[Any]
        The expected extracted data types.
    expected_heliostat_ideal : list[Union[torch.Tensor, int, str]]
        The expected data for the ideal heliostat.
    expected_heliostat_fitted : list[Union[torch.Tensor, int, str]]
        The expected data for the fitted heliostat.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If the test does not complete as expected.
    """
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    optimizer = torch.optim.Adam([torch.empty(1, requires_grad=True)], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.2,
        patience=50,
        threshold=1e-7,
        threshold_mode="abs",
    )

    extracted_list = list(
        paint_scenario_parser.extract_paint_heliostats_mixed_surface(
            paths=heliostat_paths,
            power_plant_position=power_plant_position.to(device),
            number_of_nurbs_control_points=torch.tensor([20, 20], device=device),
            nurbs_fit_max_epoch=max_epochs_for_surface_training,
            nurbs_fit_optimizer=optimizer,
            nurbs_fit_scheduler=scheduler,
            device=device,
        )
    )

    # Assert overall return types
    assert isinstance(extracted_list[0], expected_types[0])
    assert isinstance(extracted_list[1], expected_types[1])

    # Find the ideal and fitted heliostat in the returned list
    ideal_heliostat = next(
        h for h in extracted_list[0].heliostat_list if h.name == "ideal_heliostat"
    )
    fitted_heliostat = next(
        h for h in extracted_list[0].heliostat_list if h.name == "fitted_heliostat"
    )

    # Assert ideal heliostat properties.
    assert ideal_heliostat.name == heliostat_paths[0][0]
    torch.testing.assert_close(
        ideal_heliostat.position, expected_heliostat_ideal[0].to(device)
    )
    assert len(ideal_heliostat.actuators.actuator_list) == expected_heliostat_ideal[1]
    assert (
        ideal_heliostat.actuators.actuator_list[0].type == expected_heliostat_ideal[2]
    )
    torch.testing.assert_close(
        ideal_heliostat.actuators.actuator_list[0].parameters.increment,
        expected_heliostat_ideal[3].to(device),
    )
    torch.testing.assert_close(
        ideal_heliostat.kinematic.initial_orientation,
        expected_heliostat_ideal[4].to(device),
    )
    torch.testing.assert_close(
        ideal_heliostat.surface.facet_list[0].canting,
        expected_heliostat_ideal[5].to(device),
    )
    torch.testing.assert_close(
        ideal_heliostat.surface.facet_list[0].control_points[0, 3],
        expected_heliostat_ideal[6].to(device),
    )

    # Assert fitted heliostat properties.
    assert fitted_heliostat.name == heliostat_paths[1][0]
    torch.testing.assert_close(
        fitted_heliostat.position, expected_heliostat_fitted[0].to(device)
    )
    assert len(fitted_heliostat.actuators.actuator_list) == expected_heliostat_fitted[1]
    assert (
        fitted_heliostat.actuators.actuator_list[0].type == expected_heliostat_fitted[2]
    )
    torch.testing.assert_close(
        fitted_heliostat.actuators.actuator_list[0].parameters.increment,
        expected_heliostat_fitted[3].to(device),
    )
    torch.testing.assert_close(
        fitted_heliostat.kinematic.initial_orientation,
        expected_heliostat_fitted[4].to(device),
    )
    torch.testing.assert_close(
        fitted_heliostat.surface.facet_list[0].canting,
        expected_heliostat_fitted[5].to(device),
    )
    torch.testing.assert_close(
        fitted_heliostat.surface.facet_list[0].control_points[0, 3],
        expected_heliostat_fitted[6].to(device),
    )


def _make_fake_calibration_data(
    base_directory_path: pathlib.Path,
    heliostat_name_list: list[str],
    image_variant_name: str,
    count_per_heliostat: int,
) -> str:
    """
    Create a deterministic fake folder tree with property/image pairs.

    Parameters
    ----------
    base_directory_path : pathlib.Path
        The base directory where the fake folder tree will be created.
    heliostat_name_list : list[str]
        List of heliostat names for which the calibration data will be created.
    image_variant_name : str
        Identifier for the variant of image data to use (e.g., ''raw'', 'processed'').
    count_per_heliostat : int
        Number of property/image pairs to create per heliostat.

    Returns
    -------
    str
        The name of the folder containing the calibration data.
    """
    paint_calibration_folder_name = paint_mappings.SAVE_CALIBRATION
    for heliostat_name in heliostat_name_list:
        calibration_directory_path = (
            base_directory_path / heliostat_name / paint_mappings.SAVE_CALIBRATION
        )
        calibration_directory_path.mkdir(parents=True, exist_ok=True)
        for index in range(count_per_heliostat):
            (
                calibration_directory_path
                / f"{index}{paint_mappings.CALIBRATION_PROPERTIES_IDENTIFIER}"
            ).write_text("{}")
            # Write a tiny, obviously-fake PNG header so opening as binary won't crash.
            (
                calibration_directory_path / f"{index}-{image_variant_name}.png"
            ).write_bytes(b"\x89PNG\r\nfake")
    return paint_calibration_folder_name


@pytest.mark.parametrize(
    "randomize_selection_flag, random_seed_value, number_of_measurements, image_variant_name",
    [
        (False, 0, 2, "flux"),
        (True, 123, 2, "flux"),
    ],
)
def test_build_heliostat_data_mapping_shape_parametrized(
    tmp_path: pathlib.Path,
    monkeypatch: MonkeyPatch,
    randomize_selection_flag: bool,
    random_seed_value: int,
    number_of_measurements: int,
    image_variant_name: str,
) -> None:
    """
    Test shape, type, and correspondence checks for heliostat data mapping.

    This parametrized test verifies that `utils.build_heliostat_data_mapping`
    returns a correctly structured list of mappings for both
    `randomize_selection=False` and `randomize_selection=True`.

    The test:
    1. Creates fake calibration data for multiple heliostats.
    2. Monkeypatches relevant module variables to point to the fake data.
    3. Invokes the mapping function with the given parameters.
    4. Verifies:
    - The return value is a list of the same length as the heliostat list.
    - Each element is a tuple of `(heliostat_name, property_file_paths, image_file_paths)`.
    - Types of elements and paths are correct.
    - The number of measurements matches the expected value.
    - Property/image file paths correspond by ID and reside in the same directory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest for creating fake calibration data.
    monkeypatch : MonkeyPatch
        Pytest fixture to dynamically replace module attributes for testing.
    randomize_selection_flag : bool
        Flag to randomize selection of measurement files when building the mapping.
    random_seed_value : int
        Random seed to use when `randomize_selection_flag` is `True` for reproducibility.
    number_of_measurements : int
        Number of measurement files to select per heliostat.
    image_variant_name : str
        Identifier for the variant of image data to use (e.g., ``raw``, ``processed``).

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    heliostat_name_list = ["heliostat_1", "heliostat_2"]
    # Create 5 samples per heliostat.
    _ = _make_fake_calibration_data(
        tmp_path,
        heliostat_name_list,
        image_variant_name,
        count_per_heliostat=5,
    )

    result_mapping_list = paint_scenario_parser.build_heliostat_data_mapping(
        base_path=str(tmp_path),
        heliostat_names=heliostat_name_list,
        number_of_measurements=number_of_measurements,
        image_variant=image_variant_name,
        randomize=randomize_selection_flag,
        seed=random_seed_value,
    )

    assert isinstance(result_mapping_list, list)
    assert len(result_mapping_list) == len(heliostat_name_list)

    for heliostat_entry in result_mapping_list:
        assert isinstance(heliostat_entry, tuple) and len(heliostat_entry) == 3
        heliostat_name, property_file_paths, image_file_paths = heliostat_entry

        assert isinstance(heliostat_name, str) and heliostat_name in heliostat_name_list

        assert isinstance(property_file_paths, list)
        assert isinstance(image_file_paths, list)
        assert all(
            isinstance(property_path, pathlib.Path)
            for property_path in property_file_paths
        )
        assert all(
            isinstance(image_path, pathlib.Path) for image_path in image_file_paths
        )

        assert len(property_file_paths) == number_of_measurements
        assert len(image_file_paths) == number_of_measurements

        # Correspondence by ID and directory.
        for property_file_path, image_file_path in zip(
            property_file_paths, image_file_paths
        ):
            assert property_file_path.parent == image_file_path.parent
            assert (
                property_file_path.stem.split("-")[0]
                == image_file_path.stem.split("-")[0]
            )


@pytest.mark.parametrize("random_seed_value", [7, 11, 123, 2024])
def test_build_heliostat_data_mapping_randomization_changes_order(
    tmp_path: pathlib.Path,
    monkeypatch: MonkeyPatch,
    random_seed_value: int,
) -> None:
    """
    Test that randomized selection order or subset differs from the non-randomized version.

    This test verifies that when `randomize=True`, the file selection order (or subset)
    returned by `utils.build_heliostat_data_mapping` differs from the deterministic
    sorted selection for at least one heliostat, given enough available samples.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest for creating fake calibration data.
    monkeypatch : MonkeyPatch
        Pytest fixture to dynamically replace module attributes for testing.
    random_seed_value : int
        Random seed to use for reproducibility in randomized selection.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    image_variant_name = "flux"
    heliostat_name_list = ["heliostat_1", "heliostat_2"]
    number_of_measurements = 4

    # Create 10 samples per heliostat so a different subset/order is very likely.
    _ = _make_fake_calibration_data(
        tmp_path,
        heliostat_name_list,
        image_variant_name,
        count_per_heliostat=10,
    )

    result_sorted_list = paint_scenario_parser.build_heliostat_data_mapping(
        base_path=str(tmp_path),
        heliostat_names=heliostat_name_list,
        number_of_measurements=number_of_measurements,
        image_variant=image_variant_name,
        randomize=False,
        seed=random_seed_value,
    )
    result_randomized_list = paint_scenario_parser.build_heliostat_data_mapping(
        base_path=str(tmp_path),
        heliostat_names=heliostat_name_list,
        number_of_measurements=number_of_measurements,
        image_variant=image_variant_name,
        randomize=True,
        seed=random_seed_value,
    )

    # Compare per heliostat.
    different_for_any_heliostat = False
    for (sorted_name, sorted_property_paths, _), (
        random_name,
        randomized_property_paths,
        _,
    ) in zip(result_sorted_list, result_randomized_list):
        assert sorted_name == random_name

        sorted_identifiers = [p.stem.split("-")[0] for p in sorted_property_paths]
        randomized_identifiers = [
            p.stem.split("-")[0] for p in randomized_property_paths
        ]

        assert len(sorted_identifiers) == number_of_measurements
        assert len(randomized_identifiers) == number_of_measurements

        universe = {str(i) for i in range(10)}
        assert set(sorted_identifiers).issubset(universe)
        assert set(randomized_identifiers).issubset(universe)

        if randomized_identifiers != sorted_identifiers:
            different_for_any_heliostat = True
        assert different_for_any_heliostat, (
            "Randomized selection did not differ from sorted order."
        )
