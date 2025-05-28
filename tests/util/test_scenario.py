from typing import Optional
from unittest.mock import MagicMock

import pytest
import torch

from artist.field.heliostat_field import HeliostatField
from artist.field.heliostat_group_rigid_body import HeliostatGroupRigidBody
from artist.field.tower_target_areas import TowerTargetAreas
from artist.util.scenario import Scenario


@pytest.mark.parametrize(
    "mapping, default_incident_ray_direction, default_target_area_index, expected",
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
    ],
)
def test_index_mapping(
    mapping: Optional[list[tuple[str, str, torch.Tensor]]],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    default_incident_ray_direction,
    default_target_area_index,
    device: torch.device,
) -> None:
    """
    Test the index mapping of the scenario class.

    Parameters
    ----------
    mapping : Optional[list[tuple[str, str, torch.Tensor]]]
        The mapping of heliostats, target areas and incident ray directions.
    expected : tuple[torch.Tensor, torch.Tensor, torch.Tensor]
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
    mock_scenario.target_areas = mock_target_areas

    mock_heliostat_field = MagicMock(spec=HeliostatField)
    mock_heliostat_group = MagicMock(spec=HeliostatGroupRigidBody)
    mock_heliostat_group.names = ["AA31", "AA35", "AA39", "AB38"]
    mock_heliostat_group.number_of_heliostats = 4
    mock_heliostat_field.heliostat_groups = [mock_heliostat_group]
    mock_scenario.heliostat_field = mock_heliostat_field

    mock_scenario.index_mapping = MagicMock(wraps=Scenario.index_mapping)
    if default_incident_ray_direction is None and default_target_area_index is None:
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
                default_incident_ray_direction=default_incident_ray_direction,
                default_target_area_index=default_target_area_index,
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
        incident_ray_directions.to(device), expected[2].to(device), atol=5e-4, rtol=5e-4
    )
