from typing import Optional
from unittest.mock import MagicMock

import pytest
import torch

from artist.field.heliostat_field import HeliostatField
from artist.field.heliostat_group_rigid_body import HeliostatGroupRigidBody
from artist.field.tower_target_areas import TowerTargetAreas
from artist.util.scenario import Scenario


@pytest.mark.parametrize(
    "mapping, expected",
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
            (
                torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
                torch.tensor([3, 0]),
                torch.tensor([1, 3]),
            ),
        ),
        (
            None,
            (
                torch.tensor(
                    [
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                    ]
                ),
                torch.tensor([0, 1, 2, 3]),
                torch.tensor([1, 1, 1, 1]),
            ),
        ),
    ],
)
def test_index_mapping(
    mapping: Optional[list[tuple[str, str, torch.Tensor]]],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
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

    (incident_ray_directions, active_heliostats_indices, target_area_indices) = (
        Scenario.index_mapping(
            mock_scenario,
            string_mapping=mapping,
            heliostat_group_index=0,
            device=device,
        )
    )

    torch.testing.assert_close(
        incident_ray_directions.to(device), expected[0].to(device), atol=5e-4, rtol=5e-4
    )
    torch.testing.assert_close(
        active_heliostats_indices.to(device),
        expected[1].to(device),
        atol=5e-4,
        rtol=5e-4,
    )
    torch.testing.assert_close(
        target_area_indices.to(device), expected[2].to(device), atol=5e-4, rtol=5e-4
    )
