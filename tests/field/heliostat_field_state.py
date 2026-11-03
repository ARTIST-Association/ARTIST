from typing import Any

import torch

from artist.field.heliostat_group import HeliostatGroup


def heliostat_group_state(heliostat_group: HeliostatGroup) -> dict[str, Any]:
    """
    Collect the loaded state of a heliostat group.

    Buffers that are only filled once heliostats are activated are uninitialized after loading and are left out.

    Parameters
    ----------
    heliostat_group : HeliostatGroup
        The heliostat group.

    Returns
    -------
    dict[str, Any]
        The heliostat names and all loaded tensors of the group, its kinematics, and its actuators.
    """
    state: dict[str, Any] = {
        "type": type(heliostat_group).__name__,
        "names": list(heliostat_group.names),
    }
    for prefix, component in (
        ("", heliostat_group),
        ("kinematics.", heliostat_group.kinematics),
        ("actuators.", heliostat_group.kinematics.actuators),
    ):
        for key, value in vars(component).items():
            if (
                torch.is_tensor(value)
                and not key.startswith("active_")
                and key != "preferred_reflection_directions"
            ):
                state[f"{prefix}{key}"] = value.detach().cpu()
    return state


def assert_states_equal(
    expected_states: list[dict[str, Any]], states: list[dict[str, Any]]
) -> None:
    """
    Assert that two lists of heliostat group states are identical.

    Parameters
    ----------
    expected_states : list[dict[str, Any]]
        The expected heliostat group states.
    states : list[dict[str, Any]]
        The heliostat group states to check.

    Raises
    ------
    AssertionError
        If the states differ.
    """
    assert len(states) == len(expected_states)
    for expected_state, state in zip(expected_states, states):
        assert state.keys() == expected_state.keys()
        for key, expected_value in expected_state.items():
            if torch.is_tensor(expected_value):
                torch.testing.assert_close(
                    state[key], expected_value, rtol=0, atol=0, msg=key
                )
            else:
                assert state[key] == expected_value, key
