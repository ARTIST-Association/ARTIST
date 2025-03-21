from typing import Union
from unittest.mock import MagicMock

import pytest

from artist.field.tower_target_area_array import TargetAreaArray
from artist.util.scenario import Scenario, TargetArea


@pytest.mark.parametrize(
    "target_area_name, expected",
    [("reciever", TargetArea), ("invalid_target_area_name", None)],
)
def test_get_target_area(
    target_area_name: str, expected: Union[TargetArea, None]
) -> None:
    """
    Test the get target area method from the scenario class.

    Parameters
    ----------
    target_area_name : str
        The name of the target area.
    expected : Union[TargetArea, None]
        The expected target area or None if a value error is expected.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    # Mock a target area.
    mock_target_area = MagicMock(spec=TargetArea)
    mock_target_area.name = "receiver"
    mock_target_area_array = MagicMock(spec=TargetAreaArray)

    # Mock the Scenario containing target_areas.
    mock_scenario = MagicMock(spec=Scenario)
    mock_scenario.target_areas = mock_target_area_array
    mock_scenario.target_areas.target_area_list = [mock_target_area]

    # Check if the ValueError is thrown as expected.
    if expected is None:
        with pytest.raises(ValueError) as exc_info:
            Scenario.get_target_area(mock_scenario, target_area_name)
        assert (
            f"No target area with the name {target_area_name} found in the sceanrio!"
            in str(exc_info.value)
        )
    else:
        # Check if a target area instance is returned.
        target_area = Scenario.get_target_area(mock_scenario, target_area_name)
        assert target_area is mock_target_area
        assert target_area.name == target_area_name
