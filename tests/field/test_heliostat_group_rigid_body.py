import pytest
import torch

from artist.field.heliostat_group_rigid_body import HeliostatGroupRigidBody


@pytest.fixture
def heliostat_group(device: torch.device) -> HeliostatGroupRigidBody:
    """
    Generate a heliostat group of type rigid body linear.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    HeliostatGroupRigidBody
        A heliostat group.
    """
    actuator_parameters = torch.rand((3, 7, 2), device=device)
    actuator_parameters[:, 0, :] = 0
    actuator_parameters[:, 1, :] = 1

    return HeliostatGroupRigidBody(
        names=["heliostat_1", "heliostat_2", "heliostat_3"],
        positions=torch.tensor(
            [[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0], [3.0, 0.0, 0.0, 1.0]],
            device=device,
        ),
        aim_points=torch.tensor(
            [[0.0, -1.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]],
            device=device,
        ),
        surface_points=torch.rand((3, 100, 4), device=device),
        surface_normals=torch.rand((3, 100, 4), device=device),
        initial_orientations=torch.tensor(
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            device=device,
        ),
        kinematic_deviation_parameters=torch.rand((3, 18), device=device),
        actuator_parameters=actuator_parameters,
        device=device,
    )


@pytest.mark.parametrize(
    "active_heliostats_mask, expected_size",
    [
        (torch.tensor([1, 1, 1]), 3),
        (torch.tensor([1, 0, 1]), 2),
        (torch.tensor([6, 0, 1]), 7),
        (None, 3),
    ],
)
def test_activate_heliostats(
    heliostat_group: HeliostatGroupRigidBody,
    active_heliostats_mask: torch.Tensor,
    expected_size: int,
    device: torch.device,
) -> None:
    """
    Test the activate heliostats method of the heliostat groups.

    Parameters
    ----------
    heliostat_group : HeliostatGroupRigidBody
        The heliostat group.
    active_heliostats_mask : torch.Tensor
        The mask to activate certain heliostats.
    expected_size : int
        The expected size of the first dimensions of the heliostat parameters.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    if active_heliostats_mask is None:
        heliostat_group.activate_heliostats(device=device)

    else:
        heliostat_group.activate_heliostats(
            active_heliostats_mask=active_heliostats_mask.to(device),
            device=device,
        )
    assert heliostat_group.number_of_active_heliostats == expected_size

    group_attributes = ["active_surface_points", "active_surface_normals"]
    kinematic_attributes = [
        "active_heliostat_positions",
        "active_initial_orientations",
        "active_deviation_parameters",
    ]

    for attribute in group_attributes:
        parameter = getattr(heliostat_group, attribute)
        assert parameter.shape[0] == expected_size

    for attribute in kinematic_attributes:
        parameter = getattr(heliostat_group.kinematic, attribute)
        assert parameter.shape[0] == expected_size

    assert (
        heliostat_group.kinematic.actuators.active_actuator_parameters.shape[0]
        == expected_size
    )
