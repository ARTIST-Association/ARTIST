import pathlib

import h5py
import pytest
from pytest_mock import MockerFixture

from artist.scenario.configuration_classes import (
    HeliostatListConfig,
    LightSourceListConfig,
    PowerPlantConfig,
    PrototypeConfig,
    TargetAreaListConfig,
)
from artist.scenario.h5_scenario_generator import H5ScenarioGenerator
from artist.util import config_dictionary


@pytest.fixture
def scenario_generator(mocker: MockerFixture) -> H5ScenarioGenerator:
    """
    Create the h5 scenario generator.

    Parameters
    ----------
    mocker : MockerFixture
        A pytest-mocker fixture used to create mock objects.

    Returns
    -------
    H5ScenarioGenerator
        The h5 scenario generator.
    """
    mock_power_plant_config = mocker.MagicMock(spec=PowerPlantConfig)
    mock_target_area_list_config = mocker.MagicMock(spec=TargetAreaListConfig)
    mock_light_source_list_config = mocker.MagicMock(spec=LightSourceListConfig)
    mock_prototype_config = mocker.MagicMock(spec=PrototypeConfig)
    mock_heliostat_list_config = mocker.MagicMock(spec=HeliostatListConfig)

    mock_power_plant_config.create_power_plant_dict.return_value = {"param1": 123}
    mock_target_area_list_config.create_target_area_list_dict.return_value = {
        "param2": 456
    }
    mock_light_source_list_config.create_light_source_list_dict.return_value = {
        "param3": 789
    }
    mock_prototype_config.create_prototype_dict.return_value = {"param4": "abc"}
    mock_heliostat_list_config.create_heliostat_list_dict.return_value = {
        "param5": "xyz"
    }

    mocker.patch.object(
        H5ScenarioGenerator, "_check_equal_facet_numbers", return_value=None
    )

    scenario_generator = H5ScenarioGenerator(
        file_path=pathlib.Path("scenario"),
        version=1.0,
        power_plant_config=mock_power_plant_config,
        target_area_list_config=mock_target_area_list_config,
        light_source_list_config=mock_light_source_list_config,
        prototype_config=mock_prototype_config,
        heliostat_list_config=mock_heliostat_list_config,
    )

    mocker.patch.object(
        scenario_generator, "_get_number_of_heliostat_groups", return_value=3
    )
    mocker.patch.object(
        scenario_generator, "_flatten_dict", side_effect=lambda d, *_: d
    )
    mocker.patch.object(
        scenario_generator,
        "_include_parameters",
        side_effect=lambda file, prefix, parameters: [
            file.create_dataset(f"{prefix}/{k}", data=v) for k, v in parameters.items()
        ],
    )
    return scenario_generator


@pytest.mark.parametrize(
    "filename",
    [
        (pathlib.Path("scenario.h5")),
        (pathlib.Path("scenario")),
        (pathlib.Path("scenario.txt")),
    ],
)
def test_generate_scenario(
    scenario_generator: H5ScenarioGenerator,
    tmp_path: pathlib.Path,
    filename: pathlib.Path,
) -> None:
    """
    Test the h5 scenario generator.

    Parameters
    ----------
    scenario_generator : H5ScenarioGenerator
        The h5 scenario generator.
    tmp_path : pathlib.Path
        Pytest temporary directory fixture.
    filename : pathlib.Path
        File name to test.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
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
        assert config_dictionary.number_of_heliostat_groups in f
        assert f[config_dictionary.number_of_heliostat_groups][()] == 3

        expected_datasets = {
            config_dictionary.power_plant_key: ["param1"],
            config_dictionary.target_area_key: ["param2"],
            config_dictionary.light_source_key: ["param3"],
            config_dictionary.prototype_key: ["param4"],
            config_dictionary.heliostat_key: ["param5"],
        }

        for prefix, keys in expected_datasets.items():
            for key in keys:
                dataset_path = f"{prefix}/{key}"
                assert dataset_path in f
