import logging
import pathlib
import types

import h5py
import pytest
import torch
from pytest_mock import MockerFixture

from artist.io import h5_columnar_heliostats
from artist.scenario import H5ScenarioGenerator
from artist.util import constants
from artist.util.config import (
    HeliostatListConfig,
    LightSourceListConfig,
    PowerPlantConfig,
    PrototypeConfig,
    TargetAreaCylindricalListConfig,
    TargetAreaPlanarListConfig,
)


@pytest.fixture
def scenario_generator(mocker: MockerFixture) -> H5ScenarioGenerator:
    """Create a patched H5ScenarioGenerator with mocked config objects."""
    mock_power_plant_config = mocker.MagicMock(spec=PowerPlantConfig)
    mock_target_area_list_planar_config = mocker.MagicMock(
        spec=TargetAreaPlanarListConfig
    )
    mock_target_area_list_cylindrical_config = mocker.MagicMock(
        spec=TargetAreaCylindricalListConfig
    )
    mock_light_source_list_config = mocker.MagicMock(spec=LightSourceListConfig)
    mock_prototype_config = mocker.MagicMock(spec=PrototypeConfig)
    mock_heliostat_list_config = mocker.MagicMock(spec=HeliostatListConfig)

    mock_power_plant_config.create_power_plant_dict.return_value = {"param1": 123}
    mock_target_area_list_planar_config.create_target_area_list_dict.return_value = {
        "param2": 456
    }
    mock_target_area_list_cylindrical_config.create_target_area_list_dict.return_value = {
        "param3": 4567
    }
    mock_light_source_list_config.create_light_source_list_dict.return_value = {
        "param4": 789
    }
    mock_prototype_config.create_prototype_dict.return_value = {"param5": "abc"}
    mock_heliostat_list_config.create_heliostat_list_dict.return_value = {
        "heliostat_1": {
            constants.heliostat_id: 1,
            constants.heliostat_position: torch.tensor([0.0, 1.0, 0.0, 1.0]),
        }
    }

    mocker.patch.object(
        H5ScenarioGenerator, "_check_equal_facet_numbers", return_value=None
    )

    generator = H5ScenarioGenerator(
        file_path=pathlib.Path("scenario"),
        version=1.0,
        power_plant_config=mock_power_plant_config,
        target_area_list_planar_config=mock_target_area_list_planar_config,
        target_area_list_cylindrical_config=mock_target_area_list_cylindrical_config,
        light_source_list_config=mock_light_source_list_config,
        prototype_config=mock_prototype_config,
        heliostat_list_config=mock_heliostat_list_config,
    )

    mocker.patch.object(generator, "_get_number_of_heliostat_groups", return_value=3)
    mocker.patch.object(generator, "_flatten_dict", side_effect=lambda d, *_: d)
    mocker.patch.object(
        generator,
        "_include_parameters",
        side_effect=lambda file, prefix, parameters: [
            file.create_dataset(f"{prefix}/{k}", data=v) for k, v in parameters.items()
        ],
    )
    return generator


@pytest.mark.parametrize(
    "filename",
    [
        pathlib.Path("scenario.h5"),
        pathlib.Path("scenario"),
        pathlib.Path("scenario.txt"),
    ],
)
def test_generate_scenario(
    scenario_generator: H5ScenarioGenerator,
    tmp_path: pathlib.Path,
    filename: pathlib.Path,
) -> None:
    """Test scenario generation and saved HDF5 structure for supported filename variants."""
    scenario_generator.file_path = tmp_path / filename
    scenario_generator.generate_scenario()

    save_name = (
        scenario_generator.file_path
        if scenario_generator.file_path.suffix == ".h5"
        else scenario_generator.file_path.with_suffix(".h5")
    )
    assert save_name.exists()

    with h5py.File(save_name, "r") as f:
        assert f.attrs["version"] == 1.0
        assert constants.number_of_heliostat_groups in f
        assert f[constants.number_of_heliostat_groups][()] == 3

        expected_datasets = {
            constants.power_plant_key: ["param1"],
            constants.target_area_planar_key: ["param2"],
            constants.target_area_cylindrical_key: ["param3"],
            constants.light_source_key: ["param4"],
            constants.prototype_key: ["param5"],
        }
        for prefix, keys in expected_datasets.items():
            for key in keys:
                assert f"{prefix}/{key}" in f

        heliostats = f[constants.heliostat_key]
        assert h5_columnar_heliostats.is_columnar(heliostats)
        assert heliostats[constants.names].asstr()[()].tolist() == ["heliostat_1"]
        assert heliostats[constants.heliostat_position][()].tolist() == [
            [0.0, 1.0, 0.0, 1.0]
        ]


def test_generate_scenario_falls_back_to_legacy_layout(
    scenario_generator: H5ScenarioGenerator,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that heliostats the columnar layout cannot store are saved in the legacy layout with a warning."""
    scenario_generator.file_path = tmp_path / "scenario.h5"
    # Positions of different shapes cannot be stacked into one dataset.
    scenario_generator.heliostat_list_config.create_heliostat_list_dict.return_value = {  # type: ignore[attr-defined]
        "heliostat_1": {
            constants.heliostat_id: 1,
            constants.heliostat_position: torch.tensor([0.0, 1.0, 0.0, 1.0]),
        },
        "heliostat_2": {
            constants.heliostat_id: 2,
            constants.heliostat_position: torch.tensor([0.0, 1.0, 0.0]),
        },
    }
    scenario_generator._flatten_dict = types.MethodType(  # type: ignore[method-assign]
        H5ScenarioGenerator._flatten_dict, scenario_generator
    )
    scenario_generator._include_parameters = H5ScenarioGenerator._include_parameters  # type: ignore[method-assign]

    with caplog.at_level(logging.WARNING):
        scenario_generator.generate_scenario()

    assert "Storing the heliostats in the legacy layout" in caplog.text
    with h5py.File(scenario_generator.file_path, "r") as f:
        heliostats = f[constants.heliostat_key]
        assert not h5_columnar_heliostats.is_columnar(heliostats)
        assert heliostats[f"heliostat_2/{constants.heliostat_position}"].shape == (3,)


def test_generate_scenario_invalid_parent_path_raises(mocker: MockerFixture) -> None:
    """Test that generator init raises if output parent directory does not exist."""
    invalid_path = pathlib.Path("/definitely/nonexistent/path/scenario.h5")

    mock_power_plant_config = mocker.MagicMock(spec=PowerPlantConfig)
    mock_target_area_list_planar_config = mocker.MagicMock(
        spec=TargetAreaPlanarListConfig
    )
    mock_target_area_list_cylindrical_config = mocker.MagicMock(
        spec=TargetAreaCylindricalListConfig
    )
    mock_light_source_list_config = mocker.MagicMock(spec=LightSourceListConfig)
    mock_prototype_config = mocker.MagicMock(spec=PrototypeConfig)
    mock_heliostat_list_config = mocker.MagicMock(spec=HeliostatListConfig)

    with pytest.raises(FileNotFoundError):
        H5ScenarioGenerator(
            file_path=invalid_path,
            version=1.0,
            power_plant_config=mock_power_plant_config,
            target_area_list_planar_config=mock_target_area_list_planar_config,
            target_area_list_cylindrical_config=mock_target_area_list_cylindrical_config,
            light_source_list_config=mock_light_source_list_config,
            prototype_config=mock_prototype_config,
            heliostat_list_config=mock_heliostat_list_config,
        )
