import pytest
import torch

from artist.field.actuators import Actuators


def test_abstract_actuators(
    device: torch.device,
) -> None:
    """
    Test the abstract methods of actuators.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    abstract_actuator = Actuators()

    with pytest.raises(NotImplementedError) as exc_info:
        abstract_actuator.motor_positions_to_angles(
            torch.tensor([0.0, 0.0], device=device)
        )
    assert "Must be overridden!" in str(exc_info.value)

    with pytest.raises(NotImplementedError) as exc_info:
        abstract_actuator.angles_to_motor_positions(
            torch.tensor([0.0, 0.0], device=device)
        )
    assert "Must be overridden!" in str(exc_info.value)
