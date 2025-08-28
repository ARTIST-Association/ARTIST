import pathlib
from typing import Any

import pytest
import torch
from _pytest.monkeypatch import MonkeyPatch

from artist import ARTIST_ROOT
from artist.data_loader import paint_loader
from artist.scenario.configuration_classes import (
    HeliostatListConfig,
    PowerPlantConfig,
    PrototypeConfig,
    TargetAreaListConfig,
)
from artist.util import config_dictionary

torch.manual_seed(7)
torch.cuda.manual_seed(7)


@pytest.mark.parametrize(
    "heliostat_calibration_mapping, power_plant_position, expected_list",
    [
        (
            [
                (
                    "AA39",
                    [
                        pathlib.Path(ARTIST_ROOT)
                        / "tests/data/field_data/AA39-calibration-properties_1.json",
                        pathlib.Path(ARTIST_ROOT)
                        / "tests/data/field_data/AA39-calibration-properties_2.json",
                    ],
                )
            ],
            torch.tensor([50.91342112259258, 6.387824755874856, 87.0]),
            [
                torch.tensor(
                    [
                        [
                            0.180133327842,
                            -3.419259548187,
                            35.798927307129,
                            1.000000000000,
                        ],
                        [
                            -17.412885665894,
                            -3.039341926575,
                            51.611984252930,
                            1.000000000000,
                        ],
                    ]
                ),
                torch.tensor(
                    [
                        [
                            -0.094675041735,
                            0.492933481932,
                            -0.864900708199,
                            0.000000000000,
                        ],
                        [
                            -0.274074256420,
                            0.439921498299,
                            -0.855191409588,
                            0.000000000000,
                        ],
                    ]
                ),
                torch.tensor([[28061.0, 47874.0], [22585.0, 48224.0]]),
                torch.tensor([0, 2]),
                torch.tensor([3, 0]),
            ],
        )
    ],
)
def test_extract_paint_calibration_data(
    heliostat_calibration_mapping: list[tuple[str, list[pathlib.Path]]],
    power_plant_position: torch.Tensor,
    expected_list: list[torch.Tensor],
    device: torch.device,
) -> None:
    """
    Test the function to extract calibration data from ``PAINT`` calibration data.

    Parameters
    ----------
    heliostat_calibration_mapping : list[tuple[str, list[pathlib.Path]]]
        The mapping of heliostats and their calibration data files.
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
        paint_loader.extract_paint_calibration_properties_data(
            heliostat_calibration_mapping=heliostat_calibration_mapping,
            power_plant_position=power_plant_position.to(device),
            heliostat_names=["AA31", "AA39"],
            target_area_names=[
                "multi_focus_tower",
                "receiver",
                "solar_tower_juelich_upper",
                "solar_tower_juelich_lower",
            ],
            device=device,
        )
    )

    for actual, expected in zip(extracted_list, expected_list):
        torch.testing.assert_close(actual, expected.to(device), atol=5e-4, rtol=5e-4)


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
        paint_loader.extract_paint_tower_measurements(
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
        (
            [
                (
                    "heliostat_1",
                    pathlib.Path(ARTIST_ROOT)
                    / "tests/data/field_data/AA39-heliostat-properties.json",
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
                torch.tensor([-1.609997987747, 0.206321120262, 0.043028946966]),
            ],
        ),
    ],
)
def test_extract_paint_heliostats(
    heliostat_and_deflectometry_paths: list[tuple[str, pathlib.Path, pathlib.Path]],
    power_plant_position: torch.Tensor,
    max_epochs_for_surface_training: int,
    expected_types: list[Any],
    expected_heliostat: list[Any],
    device: torch.device,
) -> None:
    """
    Test the heliostat extraction for ``PAINT`` data.

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
        paint_loader.extract_paint_heliostats(
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


def test_extract_paint_heliostat_no_optimizer(device: torch.device) -> None:
    """
    Test the heliostat extraction errors for ``PAINT`` data.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    heliostat_and_deflectometry_paths = [
        (
            "heliostat_1",
            pathlib.Path(ARTIST_ROOT)
            / "tests/data/field_data/AA39-heliostat-properties.json",
            pathlib.Path(ARTIST_ROOT) / "tests/data/field_data/AA39-deflectometry.h5",
        )
    ]
    with pytest.raises(ValueError) as exc_info:
        paint_loader.extract_paint_heliostats(
            paths=heliostat_and_deflectometry_paths,
            power_plant_position=torch.tensor(
                [50.91342112259258, 6.387824755874856, 87.0], device=device
            ),
            number_of_nurbs_control_points=torch.tensor([20, 20], device=device),
            nurbs_fit_max_epoch=2,
            device=device,
        )

        assert (
            "When providing deflectometry data to generate surfaces with a NURBS fit, an optimizer needs to be provided!"
            in str(exc_info.value)
        )


@pytest.mark.parametrize(
    "wgs84_coordinates, reference_point, expected_enu_coordinates",
    [
        # Coordinates of Juelich power plant and multi-focus tower.
        (
            (
                torch.tensor(
                    [[50.91339645088695, 6.387574436728054, 138.97975]],
                    dtype=torch.float64,
                ),
                torch.tensor(
                    [50.913421630859, 6.387824755874856, 87.000000000000],
                    dtype=torch.float64,
                ),
                torch.tensor([[-17.6045, -2.8012, 51.9798]]),
            )
        ),
    ],
)
def test_wgs84_to_enu_converter(
    wgs84_coordinates: torch.Tensor,
    reference_point: torch.Tensor,
    expected_enu_coordinates: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test the WGS84 to ENU conversion.

    Parameters
    ----------
    wgs84_coordinates : torch.Tensor
        The coordinates in latitude, longitude, altitude that are to be transformed.
    reference_point : torch.Tensor
        The center of origin of the ENU coordinate system in WGS84 coordinates.
    expected_enu_coordinates : torch.Tensor
        The expected enu coordinates.
    device : torch.device| str
        The device on which to initialize tensors (default is cuda).

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    calculated_enu_coordinates = paint_loader.convert_wgs84_coordinates_to_local_enu(
        wgs84_coordinates.to(device), reference_point.to(device), device
    )

    torch.testing.assert_close(
        calculated_enu_coordinates, expected_enu_coordinates.to(device)
    )


@pytest.mark.parametrize(
    "azimuth, elevation, degree, expected",
    [
        (
            torch.tensor([-45.0, -45.0, 45.0, 135.0, 225.0, 315.0]),
            torch.tensor([0.0, 45.0, 45.0, 45.0, 45.0, 45.0]),
            True,
            torch.tensor(
                [
                    [
                        -1 / torch.sqrt(torch.tensor([2.0])),
                        -1 / torch.sqrt(torch.tensor([2.0])),
                        0.0,
                    ],
                    [
                        -0.5,
                        -0.5,
                        1 / torch.sqrt(torch.tensor([2.0])),
                    ],
                    [0.5, -0.5, 1 / torch.sqrt(torch.tensor([2.0]))],
                    [0.5, 0.5, 1 / torch.sqrt(torch.tensor([2.0]))],
                    [-0.5, 0.5, 1 / torch.sqrt(torch.tensor([2.0]))],
                    [-0.5, -0.5, 1 / torch.sqrt(torch.tensor([2.0]))],
                ]
            ),
        ),
        (
            torch.tensor([-torch.pi / 4, torch.pi / 4]),
            torch.tensor([torch.pi / 4, torch.pi / 4]),
            False,
            torch.tensor(
                [
                    [-0.5, -0.5, 1 / torch.sqrt(torch.tensor([2.0]))],
                    [0.5, -0.5, 1 / torch.sqrt(torch.tensor([2.0]))],
                ]
            ),
        ),
    ],
)
def test_azimuth_elevation_to_enu(
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    degree: bool,
    expected: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test the azimuth, elevation to east, north, up converter.

    Parameters
    ----------
    azimuth : torch.Tensor
        The azimuth angle.
    elevation : torch.Tensor
        The elevation angle.
    degree : bool
        Angles in degree.
    expected : torch.Tensor
        The expected coordinates in the ENU (east, north, up) coordinate system.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    enu_coordinates = paint_loader.azimuth_elevation_to_enu(
        azimuth=azimuth, elevation=elevation, degree=degree, device=device
    )
    torch.testing.assert_close(
        enu_coordinates, expected.to(device), rtol=1e-4, atol=1e-4
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
    paint_calibration_folder_name = config_dictionary.paint_calibration_folder_name
    for heliostat_name in heliostat_name_list:
        calibration_directory_path = (
            base_directory_path / heliostat_name / paint_calibration_folder_name
        )
        calibration_directory_path.mkdir(parents=True, exist_ok=True)
        for index in range(count_per_heliostat):
            (
                calibration_directory_path
                / f"{index}{config_dictionary.paint_calibration_properties_file_name_ending.split('*')[-1]}"
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
    paint_calibration_folder_name = _make_fake_calibration_data(
        tmp_path,
        heliostat_name_list,
        image_variant_name,
        count_per_heliostat=5,
    )

    monkeypatch.setattr(
        config_dictionary,
        "paint_calibration_folder_name",
        paint_calibration_folder_name,
        raising=True,
    )
    result_mapping_list = paint_loader.build_heliostat_data_mapping(
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
    paint_calibration_folder_name = _make_fake_calibration_data(
        tmp_path,
        heliostat_name_list,
        image_variant_name,
        count_per_heliostat=10,
    )

    monkeypatch.setattr(
        config_dictionary,
        "paint_calibration_folder_name",
        paint_calibration_folder_name,
        raising=True,
    )

    result_sorted_list = paint_loader.build_heliostat_data_mapping(
        base_path=str(tmp_path),
        heliostat_names=heliostat_name_list,
        number_of_measurements=number_of_measurements,
        image_variant=image_variant_name,
        randomize=False,
        seed=random_seed_value,
    )
    result_randomized_list = paint_loader.build_heliostat_data_mapping(
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
