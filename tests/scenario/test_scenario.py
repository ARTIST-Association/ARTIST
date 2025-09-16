import pathlib
from unittest.mock import MagicMock

import h5py
import pytest
import torch

from artist import ARTIST_ROOT
from artist.field.heliostat_field import HeliostatField
from artist.field.heliostat_group_rigid_body import HeliostatGroupRigidBody
from artist.field.tower_target_areas import TowerTargetAreas
from artist.scenario.scenario import Scenario


def test_get_number_of_heliostat_groups_from_hdf5() -> None:
    """
    Test the get number of heliostat groups method.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    scenario_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/data/scenarios/test_scenario_paint_single_heliostat.h5"
    )
    number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
        scenario_path=scenario_path
    )

    assert number_of_heliostat_groups == 1


def test_value_errors_load_scenario_from_hdf5(device: torch.device) -> None:
    """
    Test the get number of heliostat groups method.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    broken_actuator_prototype_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/data/scenarios/test_broken_scenario_actuator_prototype.h5"
    )
    with h5py.File(broken_actuator_prototype_path) as scenario_file:
        with pytest.raises(ValueError) as exc_info:
            Scenario.load_scenario_from_hdf5(
                scenario_file=scenario_file,
                number_of_surface_points_per_facet=torch.tensor(
                    [50, 50], device=device
                ),
                device=device,
            )
    assert (
        "There is an error in the prototype. When using the Rigid Body Kinematic, all actuators for this prototype must have the same type."
        in str(exc_info.value)
    )

    broken_actuator_individual_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/data/scenarios/test_broken_scenario_actuator_individual.h5"
    )
    with h5py.File(broken_actuator_individual_path) as scenario_file:
        with pytest.raises(ValueError) as exc_info:
            Scenario.load_scenario_from_hdf5(
                scenario_file=scenario_file,
                number_of_surface_points_per_facet=torch.tensor(
                    [50, 50], device=device
                ),
                device=device,
            )
    assert (
        "When using the Rigid Body Kinematic, all actuators for a given heliostat must have the same type."
        in str(exc_info.value)
    )


@pytest.mark.parametrize(
    "mapping, single_incident_ray_direction, single_target_area_index, expected",
    [
        (
            [
                ("AB38", "receiver", torch.tensor([0.0, 0.0, 1.0, 0.0])),
                (
                    "AA31",
                    "solar_tower_juelich_upper",
                    torch.tensor([0.0, 1.0, 0.0, 0.0]),
                ),
            ],
            None,
            None,
            (
                torch.tensor([1, 0, 0, 1], dtype=torch.int32),
                torch.tensor([3, 1], dtype=torch.int32),
                torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
            ),
        ),
        (
            [
                ("AA39", "receiver", torch.tensor([0.0, 0.0, 1.0, 0.0])),
                (
                    "AA39",
                    "solar_tower_juelich_upper",
                    torch.tensor([0.0, 1.0, 0.0, 0.0]),
                ),
                (
                    "AA39",
                    "solar_tower_juelich_upper",
                    torch.tensor([1.0, 0.0, 0.0, 0.0]),
                ),
                (
                    "AA31",
                    "multi_focus_tower",
                    torch.tensor([1.0, 0.0, 0.0, 0.0]),
                ),
            ],
            None,
            None,
            (
                torch.tensor([1, 0, 3, 0], dtype=torch.int32),
                torch.tensor([0, 1, 3, 3], dtype=torch.int32),
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                    ]
                ),
            ),
        ),
        (
            None,
            None,
            None,
            (
                torch.tensor([1, 1, 1, 1], dtype=torch.int32),
                torch.tensor([0, 0, 0, 0], dtype=torch.int32),
                torch.tensor(
                    [
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                    ]
                ),
            ),
        ),
        (
            None,
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            1,
            (
                torch.tensor([1, 1, 1, 1], dtype=torch.int32),
                torch.tensor([1, 1, 1, 1], dtype=torch.int32),
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                    ]
                ),
            ),
        ),
        (
            [
                (
                    "AA39",
                    "receiver",
                    torch.tensor([0.0, -1.0, 0.0, 0.0]),
                ),
                (
                    "AA39",
                    "invalid_target_name_1",
                    torch.tensor([0.0, 1.0, 0.0, 0.0]),
                ),
                ("AA31", "multi_focus_tower", torch.tensor([1.0, 0.0, 1.0])),
                (
                    "AA31",
                    "invalid_target_name_2",
                    torch.tensor([1.0, 0.0, 1.0, 0.0]),
                ),
            ],
            None,
            None,
            "Invalid target 'invalid_target_name_1' (Found at index 1 of provided mapping) not found in this scenario. Invalid incident ray direction (Found at index 2 of provided mapping). This must be a normalized 4D tensor with last element 0.0. Invalid target 'invalid_target_name_2' (Found at index 3 of provided mapping) not found in this scenario. Invalid incident ray direction (Found at index 3 of provided mapping). This must be a normalized 4D tensor with last element 0.0.",
        ),
        (
            None,
            torch.tensor([5.0, 1.0]),
            7,
            "The specified single incident ray direction is invalid. Please provide a normalized 4D tensor with last element 0.0.",
        ),
        (
            None,
            torch.tensor([0.0, 1.0, 0.0, 0.0]),
            5,
            "The specified single target area index is invalid. Only 4 target areas exist in this scenario.",
        ),
    ],
)
def test_index_mapping(
    mapping: list[tuple[str, str, torch.Tensor]] | None,
    single_incident_ray_direction: torch.Tensor,
    single_target_area_index: int,
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | str,
    device: torch.device,
) -> None:
    """
    Test the index mapping of the scenario class.

    Parameters
    ----------
    mapping : list[tuple[str, str, torch.Tensor]] | None
        The mapping of heliostats, target areas and incident ray directions.
    single_incident_ray_direction : torch.Tensor
        The default incident ray direction.
    single_target_area_index : int
        The default target area index.
    expected : tuple[torch.Tensor, torch.Tensor, torch.Tensor] | str
        The expected values.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    # Mock the Scenario.
    mock_scenario = MagicMock(spec=Scenario)
    mock_target_areas = MagicMock(spec=TowerTargetAreas)
    mock_target_areas.names = [
        "multi_focus_tower",
        "receiver",
        "solar_tower_juelich_lower",
        "solar_tower_juelich_upper",
    ]
    mock_target_areas.number_of_target_areas = 4
    mock_scenario.target_areas = mock_target_areas

    mock_heliostat_field = MagicMock(spec=HeliostatField)
    mock_heliostat_group = MagicMock(spec=HeliostatGroupRigidBody)
    mock_heliostat_group.names = ["AA31", "AA35", "AA39", "AB38"]
    mock_heliostat_group.number_of_heliostats = 4
    mock_heliostat_field.heliostat_groups = [mock_heliostat_group]
    mock_scenario.heliostat_field = mock_heliostat_field

    mock_scenario.index_mapping = MagicMock(wraps=Scenario.index_mapping)
    if isinstance(expected, str):
        with pytest.raises(ValueError) as exc_info:
            if (
                single_incident_ray_direction is None
                and single_target_area_index is None
            ):
                (active_heliostats_mask, target_area_mask, incident_ray_directions) = (
                    mock_scenario.index_mapping(
                        self=mock_scenario,
                        heliostat_group=mock_heliostat_group,
                        string_mapping=mapping,
                        device=device,
                    )
                )
            else:
                (active_heliostats_mask, target_area_mask, incident_ray_directions) = (
                    mock_scenario.index_mapping(
                        self=mock_scenario,
                        heliostat_group=mock_heliostat_group,
                        single_incident_ray_direction=single_incident_ray_direction,
                        single_target_area_index=single_target_area_index,
                        device=device,
                    )
                )
        assert expected in str(exc_info.value)
    else:
        if single_incident_ray_direction is None and single_target_area_index is None:
            (active_heliostats_mask, target_area_mask, incident_ray_directions) = (
                mock_scenario.index_mapping(
                    self=mock_scenario,
                    heliostat_group=mock_heliostat_group,
                    string_mapping=mapping,
                    device=device,
                )
            )
        else:
            (active_heliostats_mask, target_area_mask, incident_ray_directions) = (
                mock_scenario.index_mapping(
                    self=mock_scenario,
                    heliostat_group=mock_heliostat_group,
                    single_incident_ray_direction=single_incident_ray_direction,
                    single_target_area_index=single_target_area_index,
                    device=device,
                )
            )
        torch.testing.assert_close(
            active_heliostats_mask.to(device),
            expected[0].to(device),
            atol=5e-4,
            rtol=5e-4,
        )
        torch.testing.assert_close(
            target_area_mask.to(device), expected[1].to(device), atol=5e-4, rtol=5e-4
        )
        torch.testing.assert_close(
            incident_ray_directions.to(device),
            expected[2].to(device),
            atol=5e-4,
            rtol=5e-4,
        )


def test_load_scenario_and_change_control_points(
    device: torch.device,
) -> None:
    """
    Test the change of number of control points while loading a scenario.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    scenario_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/data/scenarios/test_scenario_paint_single_heliostat.h5"
    )
    # Load the scenario.
    with h5py.File(
        scenario_path,
        "r",
    ) as config_h5:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=config_h5,
            number_of_surface_points_per_facet=torch.tensor([50, 50], device=device),
            change_number_of_control_points_per_facet=torch.tensor(
                [13, 13], device=device
            ),
            device=device,
        )

        torch.testing.assert_close(
            scenario.heliostat_field.heliostat_groups[0].nurbs_control_points.shape,
            torch.Size([1, 4, 13, 13, 3]),
        )


def test_set_number_of_rays(
    device: torch.device,
) -> None:
    """
    Test the set number of rays method.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    scenario_path = (
        pathlib.Path(ARTIST_ROOT)
        / "tests/data/scenarios/test_scenario_paint_single_heliostat.h5"
    )

    with h5py.File(
        scenario_path,
        "r",
    ) as config_h5:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=config_h5,
            device=device,
        )

        scenario.set_number_of_rays(200)

        torch.testing.assert_close(
            scenario.light_sources.light_source_list[0].number_of_rays,
            200,
        )
