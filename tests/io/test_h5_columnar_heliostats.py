import logging
import pathlib
import shutil
from typing import Any

import h5py
import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture

from artist import ARTIST_ROOT
from artist.io import h5_columnar_heliostats, paint_scenario_parser
from artist.io.h5_columnar_heliostats import ColumnarLayoutError
from artist.scenario import H5ScenarioGenerator, Scenario
from artist.util import constants
from artist.util.config import LightSourceConfig, LightSourceListConfig
from tests.field.heliostat_field_state import assert_states_equal, heliostat_group_state

scenarios_path = pathlib.Path(ARTIST_ROOT) / "tests/data/scenarios"


def load_states(path: pathlib.Path) -> list[dict[str, Any]]:
    """
    Load a scenario on the CPU and collect the state of its heliostat groups.

    Parameters
    ----------
    path : pathlib.Path
        The path of the scenario.

    Returns
    -------
    list[dict[str, Any]]
        The state of each heliostat group.
    """
    with h5py.File(path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file,
            number_of_surface_points_per_facet=torch.tensor([10, 10]),
            device=torch.device("cpu"),
        )
    return [
        heliostat_group_state(heliostat_group)
        for heliostat_group in scenario.heliostat_field.heliostat_groups
    ]


@pytest.mark.parametrize(
    "scenario_name",
    [
        "test_blocking",
        "test_scenario_paint_four_heliostats",
        "test_scenario_paint_four_heliostats_ideal",
        "test_scenario_paint_mix_ideal_prototype_deflectometry",
        "test_scenario_paint_single_heliostat",
        "test_scenario_stral_single_heliostat",
    ],
)
def test_converted_scenario_loads_identically(
    scenario_name: str, tmp_path: pathlib.Path
) -> None:
    """
    Test that a scenario converted to the columnar layout loads exactly as the legacy scenario.

    Parameters
    ----------
    scenario_name : str
        The name of the test scenario.
    tmp_path : pathlib.Path
        A temporary directory.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    legacy_path = scenarios_path / f"{scenario_name}.h5"
    columnar_path = tmp_path / "columnar.h5"

    h5_columnar_heliostats.convert_scenario_to_columnar(legacy_path, columnar_path)

    with (
        h5py.File(legacy_path, "r") as legacy_file,
        h5py.File(columnar_path, "r") as columnar_file,
    ):
        assert dict(columnar_file.attrs) == dict(legacy_file.attrs)
        assert set(columnar_file.keys()) == set(legacy_file.keys())
        assert h5_columnar_heliostats.is_columnar(
            columnar_file[constants.heliostat_key]
        )
    assert_states_equal(load_states(legacy_path), load_states(columnar_path))


def test_missing_parameters_load_identically(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """
    Test that missing kinematics deviations and actuator parameters default to zero and are reported in both layouts.

    Parameters
    ----------
    tmp_path : pathlib.Path
        A temporary directory.
    caplog : pytest.LogCaptureFixture
        The captured log.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    legacy_path = tmp_path / "legacy.h5"
    columnar_path = tmp_path / "columnar.h5"
    shutil.copy(scenarios_path / "test_scenario_paint_four_heliostats.h5", legacy_path)
    with h5py.File(legacy_path, "a") as scenario_file:
        heliostats = scenario_file[constants.heliostat_key]
        names = list(heliostats.keys())
        del heliostats[f"{names[1]}/kinematics/deviations/first_joint_tilt_n"]
        del heliostats[f"{names[2]}/kinematics/deviations"]
        del heliostats[f"{names[1]}/actuator/actuator_1/parameters/offset"]
    h5_columnar_heliostats.convert_scenario_to_columnar(legacy_path, columnar_path)

    warnings = []
    for path in (legacy_path, columnar_path):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="artist.field.heliostat_field"):
            states = load_states(path)
        warnings.append(
            [
                record.getMessage()
                for record in caplog.records
                if "legacy layout" not in record.getMessage()
            ]
        )
        if path == legacy_path:
            legacy_states = states
        else:
            assert_states_equal(legacy_states, states)

    assert warnings[0] == warnings[1]
    assert any(
        f"first_joint_tilt_n set for 2 heliostat(s) ({names[1]}, {names[2]})" in message
        for message in warnings[0]
    )
    assert any(
        f"offset set for actuator_1 on 1 heliostat(s) ({names[1]})" in message
        for message in warnings[0]
    )


def test_legacy_layout_warning(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """
    Test that loading the legacy layout warns once, and loading the columnar layout does not warn.

    Parameters
    ----------
    tmp_path : pathlib.Path
        A temporary directory.
    caplog : pytest.LogCaptureFixture
        The captured log.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    legacy_path = scenarios_path / "test_scenario_paint_four_heliostats.h5"
    columnar_path = tmp_path / "columnar.h5"
    h5_columnar_heliostats.convert_scenario_to_columnar(legacy_path, columnar_path)

    with caplog.at_level(logging.WARNING):
        load_states(legacy_path)
    assert caplog.text.count("legacy layout") == 1
    assert "convert_scenario_to_columnar" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        load_states(columnar_path)
    assert "legacy layout" not in caplog.text


def heliostat(
    name: str, number_of_actuators: int = 2, **parameters: Any
) -> tuple[str, dict[str, Any]]:
    """
    Create the flattened parameters of a minimal heliostat.

    Parameters
    ----------
    name : str
        The heliostat name.
    number_of_actuators : int
        The number of actuators (default is 2).
    **parameters : Any
        Parameters to add or, if None, to remove.

    Returns
    -------
    tuple[str, dict[str, Any]]
        The heliostat name and its flattened parameters.
    """
    flat_parameters: dict[str, Any] = {
        "id": 1,
        "position": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "kinematics/type": "rigid_body",
        "kinematics/initial_orientation": torch.tensor([0.0, -1.0, 0.0, 0.0]),
        "kinematics/deviations/first_joint_tilt_n": torch.tensor(0.1),
    }
    for actuator in range(number_of_actuators):
        flat_parameters.update(
            {
                f"actuator/actuator_{actuator}/type": "linear",
                f"actuator/actuator_{actuator}/clockwise_axis_movement": True,
                f"actuator/actuator_{actuator}/min_max_motor_positions": torch.tensor(
                    [0, 100]
                ),
                f"actuator/actuator_{actuator}/parameters/offset": torch.tensor(0.2),
            }
        )
    for key, value in parameters.items():
        key = key.replace("__", "/")
        if value is None:
            del flat_parameters[key]
        else:
            flat_parameters[key] = value
    return name, flat_parameters


@pytest.mark.parametrize(
    "heliostats, message",
    [
        ([], "no heliostats"),
        ([heliostat("a"), heliostat("a")], "must be unique"),
        ([heliostat("a", id=None)], "has no id"),
        ([heliostat("a", kinematics__type=None)], "has no kinematics/type"),
        ([heliostat("a", unknown=1.0)], "contains the parameter unknown"),
        (
            [heliostat("a"), heliostat("b", position=torch.tensor([0.0, 1.0, 0.0]))],
            "position",
        ),
        (
            [heliostat("a", kinematics__deviations__first_joint_tilt_n=torch.ones(2))],
            "non-scalar kinematics/deviations/first_joint_tilt_n",
        ),
        (
            [heliostat("a"), heliostat("b", number_of_actuators=3)],
            "numbers of actuators",
        ),
    ],
)
def test_heliostat_columns_errors(
    heliostats: list[tuple[str, dict[str, Any]]], message: str
) -> None:
    """
    Test that heliostats the columnar layout cannot store raise an error instead of being stored incompletely.

    Parameters
    ----------
    heliostats : list[tuple[str, dict[str, Any]]]
        The heliostats.
    message : str
        A part of the expected error message.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    with pytest.raises(ColumnarLayoutError) as exc_info:
        h5_columnar_heliostats.heliostat_columns(heliostats)
    assert message in str(exc_info.value)


def test_heliostat_columns_missing_parameters_are_nan() -> None:
    """
    Test that parameters missing for some heliostats are stored as NaN and heliostats are sorted by name.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    columns = h5_columnar_heliostats.heliostat_columns(
        [
            heliostat("b", kinematics__deviations__first_joint_tilt_n=None),
            heliostat("a", actuator__actuator_1__parameters__offset=None),
        ]
    )

    assert columns[constants.names].tolist() == ["a", "b"]
    np.testing.assert_array_equal(
        columns["kinematics/deviations/first_joint_tilt_n"],
        np.array([np.float32(0.1), np.nan]),
    )
    np.testing.assert_array_equal(
        columns["actuator/parameters/offset"],
        np.array([[np.float32(0.2), np.nan], [np.float32(0.2), np.float32(0.2)]]),
    )


@pytest.mark.parametrize(
    "dataset, value, message",
    [
        (
            "kinematics/type",
            "invalid",
            "The kinematics type: invalid is not yet implemented!",
        ),
        (
            "actuator/type",
            "invalid",
            "The actuator type: invalid is not yet implemented!",
        ),
        (
            "actuator/type",
            ["linear", "ideal"],
            "all actuators for a given heliostat must have the same type",
        ),
    ],
)
def test_invalid_columnar_heliostats(
    dataset: str, value: Any, message: str, tmp_path: pathlib.Path
) -> None:
    """
    Test that invalid heliostat parameters in the columnar layout raise the same errors as in the legacy layout.

    Parameters
    ----------
    dataset : str
        The dataset to overwrite.
    value : Any
        The value to write into each row of the dataset.
    message : str
        A part of the expected error message.
    tmp_path : pathlib.Path
        A temporary directory.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    columnar_path = tmp_path / "columnar.h5"
    h5_columnar_heliostats.convert_scenario_to_columnar(
        scenarios_path / "test_scenario_paint_four_heliostats.h5", columnar_path
    )
    with h5py.File(columnar_path, "a") as scenario_file:
        heliostats = scenario_file[constants.heliostat_key]
        heliostats[dataset][...] = np.broadcast_to(
            np.array(value, dtype=h5py.string_dtype()), heliostats[dataset].shape
        )

    with pytest.raises(ValueError) as exc_info:
        load_states(columnar_path)
    assert message in str(exc_info.value)


def test_wrong_number_of_actuators(tmp_path: pathlib.Path) -> None:
    """
    Test that heliostats with a wrong number of actuators for their kinematics raise an error.

    Parameters
    ----------
    tmp_path : pathlib.Path
        A temporary directory.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    columnar_path = tmp_path / "columnar.h5"
    h5_columnar_heliostats.convert_scenario_to_columnar(
        scenarios_path / "test_scenario_paint_four_heliostats.h5", columnar_path
    )
    with h5py.File(columnar_path, "a") as scenario_file:
        del scenario_file[constants.heliostat_key]
        h5_columnar_heliostats.write_columnar_heliostats(
            scenario_file.create_group(constants.heliostat_key),
            h5_columnar_heliostats.heliostat_columns(
                [heliostat("a", number_of_actuators=3)]
            ),
        )

    with pytest.raises(ValueError) as exc_info:
        load_states(columnar_path)
    assert "Expected 2 actuators, found 3 actuator(s)." in str(exc_info.value)


def test_legacy_surfaces_of_different_shapes(tmp_path: pathlib.Path) -> None:
    """
    Test that a legacy scenario whose heliostat groups have surfaces of different shapes still loads.

    Such heliostats cannot be arranged in one set of columns, so they are read one by one.

    Parameters
    ----------
    tmp_path : pathlib.Path
        A temporary directory.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    legacy_path = tmp_path / "legacy.h5"
    shutil.copy(scenarios_path / "test_scenario_paint_four_heliostats.h5", legacy_path)
    with h5py.File(legacy_path, "a") as scenario_file:
        for heliostat_config in scenario_file[constants.heliostat_key].values():
            if heliostat_config["actuator/actuator_0/type"].asstr()[()] == "linear":
                for facet in heliostat_config["surface/facets"].values():
                    control_points = facet["control_points"][::2, ::2]
                    del facet["control_points"]
                    facet["control_points"] = control_points

    states = load_states(legacy_path)
    assert [
        (state["names"], tuple(state["nurbs_control_points"].shape)) for state in states
    ] == [
        (["AA28", "AC43"], (2, 4, 20, 20, 3)),
        (["AA31", "AA39"], (2, 4, 10, 10, 3)),
    ]


def test_convert_scenario_to_columnar_errors(tmp_path: pathlib.Path) -> None:
    """
    Test that converting into the input file or converting a columnar scenario again raises an error.

    Parameters
    ----------
    tmp_path : pathlib.Path
        A temporary directory.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    legacy_path = scenarios_path / "test_scenario_paint_single_heliostat.h5"
    columnar_path = tmp_path / "columnar.h5"

    with pytest.raises(ValueError, match="different file"):
        h5_columnar_heliostats.convert_scenario_to_columnar(legacy_path, legacy_path)

    h5_columnar_heliostats.convert_scenario_to_columnar(legacy_path, columnar_path)
    with pytest.raises(ValueError, match="already stored in the columnar layout"):
        h5_columnar_heliostats.convert_scenario_to_columnar(
            columnar_path, tmp_path / "again.h5"
        )


def test_generated_columnar_scenario_loads_like_legacy_scenario(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """
    Test that a scenario generated from PAINT data loads identically in the columnar and in the legacy layout.

    Some heliostats use the prototypes, so that the scenario mixes individual and prototype parameters.

    Parameters
    ----------
    tmp_path : pathlib.Path
        A temporary directory.
    mocker : MockerFixture
        A pytest-mock fixture used to force the legacy layout.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    device = torch.device("cpu")
    field_data = pathlib.Path(ARTIST_ROOT) / "tests/data/field_data"
    (
        power_plant_config,
        target_area_list_planar_config,
        target_area_list_cylindrical_config,
    ) = paint_scenario_parser.extract_paint_tower_measurements(
        tower_measurements_path=field_data / "tower-measurements.json", device=device
    )
    heliostat_list_config, prototype_config = (
        paint_scenario_parser.extract_paint_heliostats_ideal_surface(
            paths=[
                (name, field_data / "AA39-heliostat-properties.json")
                for name in ("AA39", "AB01", "AC02", "AD03", "AE04")
            ],
            power_plant_position=power_plant_config.power_plant_position,
            device=device,
        )
    )
    heliostat_list_config.heliostat_list[1].surface = None
    heliostat_list_config.heliostat_list[2].kinematics = None
    heliostat_list_config.heliostat_list[3].actuators = None
    heliostat_list_config.heliostat_list[4].position = torch.tensor(
        [10.0, 20.0, 0.0, 1.0]
    )
    light_source_list_config = LightSourceListConfig(
        light_source_list=[
            LightSourceConfig(
                light_source_key="sun",
                light_source_type=constants.sun_key,
                number_of_rays=10,
                distribution_type=constants.light_source_distribution_is_normal,
                mean=0.0,
                covariance=4.3681e-06,
            )
        ]
    )

    def generate(path: pathlib.Path) -> None:
        H5ScenarioGenerator(
            file_path=path,
            power_plant_config=power_plant_config,
            target_area_list_planar_config=target_area_list_planar_config,
            target_area_list_cylindrical_config=target_area_list_cylindrical_config,
            light_source_list_config=light_source_list_config,
            heliostat_list_config=heliostat_list_config,
            prototype_config=prototype_config,
        ).generate_scenario()

    generate(tmp_path / "columnar.h5")
    mocker.patch.object(
        h5_columnar_heliostats,
        "heliostat_columns",
        side_effect=ColumnarLayoutError("forced legacy layout"),
    )
    generate(tmp_path / "legacy.h5")
    mocker.stopall()

    with (
        h5py.File(tmp_path / "columnar.h5", "r") as columnar_file,
        h5py.File(tmp_path / "legacy.h5", "r") as legacy_file,
    ):
        assert h5_columnar_heliostats.is_columnar(
            columnar_file[constants.heliostat_key]
        )
        assert not h5_columnar_heliostats.is_columnar(
            legacy_file[constants.heliostat_key]
        )
    assert_states_equal(
        load_states(tmp_path / "legacy.h5"), load_states(tmp_path / "columnar.h5")
    )
