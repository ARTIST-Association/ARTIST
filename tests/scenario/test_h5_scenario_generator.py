import pathlib

import h5py
import pytest
from pytest_mock import MockerFixture

from artist.scenario.configuration_classes import (
    HeliostatListConfig,
    LightSourceListConfig,
    PowerPlantConfig,
    PrototypeConfig,
    TargetAreaCylindricalListConfig,
    TargetAreaPlanarListConfig,
)
from artist.scenario.h5_scenario_generator import H5ScenarioGenerator
from artist.util import constants


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
        "param6": "xyz"
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
            constants.heliostat_key: ["param6"],
        }
        for prefix, keys in expected_datasets.items():
            for key in keys:
                assert f"{prefix}/{key}" in f


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
