import pytest
import torch

from artist.field.actuators import Actuators
from artist.field.heliostat_group import HeliostatGroup
from artist.field.kinematic import Kinematic


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
    actuator_parameters = torch.rand((3, 7, 2))
    abstract_actuator = Actuators(
        actuator_parameters=actuator_parameters, device=device
    )

    with pytest.raises(NotImplementedError) as exc_info:
        abstract_actuator.motor_positions_to_angles(
            motor_positions=torch.tensor([0.0, 0.0], device=device),
            device=device,
        )
    assert "Must be overridden!" in str(exc_info.value)

    with pytest.raises(NotImplementedError) as exc_info:
        abstract_actuator.angles_to_motor_positions(
            angles=torch.tensor([0.0, 0.0], device=device),
            device=device,
        )
    assert "Must be overridden!" in str(exc_info.value)


def test_abstract_kinematics(
    device: torch.device,
) -> None:
    """
    Test the abstract methods of the kinematic.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    abstract_kinematic = Kinematic()

    with pytest.raises(NotImplementedError) as exc_info:
        abstract_kinematic.incident_ray_directions_to_orientations(
            incident_ray_directions=torch.tensor([0.0, 0.0, 1.0, 0.0], device=device),
            device=device,
        )
    assert "Must be overridden!" in str(exc_info.value)

    with pytest.raises(NotImplementedError) as exc_info:
        abstract_kinematic.motor_positions_to_orientations(
            motor_positions=torch.tensor([1.0, 1.0], device=device)
        )
    assert "Must be overridden!" in str(exc_info.value)


def test_abstract_heliostat_group(
    device: torch.device,
) -> None:
    """
    Test the abstract methods of a heliostat group.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    abstract_heliostat_group = HeliostatGroup(
        names=["helisotat_1"],
        positions=torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device),
        aim_points=torch.tensor([[0.0, -5.0, 0.0, 1.0]], device=device),
        surface_points=torch.rand((1, 100, 4), device=device),
        surface_normals=torch.rand((1, 100, 4), device=device),
        initial_orientations=torch.tensor([0.0, -1.0, 0.0, 0.0], device=device),
        kinematic_deviation_parameters=torch.rand((1, 18), device=device),
        actuator_parameters=torch.rand((1, 6, 2), device=device),
        device=device,
    )

    with pytest.raises(NotImplementedError) as exc_info:
        abstract_heliostat_group.align_surfaces_with_incident_ray_directions(
            aim_points=torch.tensor([0.0, -10.0, 0.0, 0.0], device=device),
            incident_ray_directions=torch.tensor([0.0, 1.0, 0.0, 0.0], device=device),
            device=device,
        )
    assert "Must be overridden!" in str(exc_info.value)

    with pytest.raises(NotImplementedError) as exc_info:
        abstract_heliostat_group.get_orientations_from_motor_positions(
            motor_positions=torch.tensor([0.0, 0.0], device=device),
            device=device,
        )
    assert "Must be overridden!" in str(exc_info.value)
