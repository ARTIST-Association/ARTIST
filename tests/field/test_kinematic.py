import pytest
import torch

from artist.field.kinematic import Kinematic
from artist.field.kinematic_rigid_body import RigidBody


@pytest.mark.parametrize(
    "expected",
    [
        torch.tensor(
            [
                [
                    [
                        9.999455809593e-01,
                        -4.559969624118e-10,
                        -1.043199468404e-02,
                        -1.852722256444e-03,
                    ],
                    [
                        -7.413947489113e-03,
                        7.035020589828e-01,
                        -7.106545567513e-01,
                        -1.891756802797e-01,
                    ],
                    [
                        7.338930387050e-03,
                        7.106932401657e-01,
                        7.034637928009e-01,
                        6.132812798023e-02,
                    ],
                    [
                        0.000000000000e00,
                        0.000000000000e00,
                        0.000000000000e00,
                        1.000000000000e00,
                    ],
                ]
            ]
        )
    ],
)
def test_kinematic_forward(expected: torch.Tensor, device: torch.device) -> None:
    """
    Test the forward method of the kinematic.

    Parameters
    ----------
    expected : torch.Tensor
        The expected test result.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    incident_ray_directions = torch.tensor([[0.0, 0.0, -1.0, 0.0]], device=device)
    aim_points = torch.tensor([[0.0, -10.0, 0.0, 1.0]], device=device)

    translation_deviation_parameters = torch.tensor(
        [
            [
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.3150,
                0.0000,
                -0.1776,
                -0.4045,
            ]
        ],
        device=device,
    )
    rotation_deviation_parameters = torch.tensor(
        [
            [
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ]
        ],
        device=device,
    )
    actuator_parameters_non_optimizable = torch.tensor(
        [
            [
                [0.0000e00, 0.0000e00],
                [0.0000e00, 1.0000e00],
                [0.0000e00, 0.0000e00],
                [60000, 80000],
                [1.5417e05, 1.5417e05],
                [3.4061e-01, 3.4790e-01],
                [3.2040e-01, 3.0900e-01],
            ]
        ],
        device=device,
    )
    actuator_parameters_optimizable = torch.tensor(
        [
            [
                [-1.5708e00, 9.5993e-01],
                [7.5000e-02, 7.5000e-02],
            ]
        ],
        device=device,
    )

    kinematic = RigidBody(
        number_of_heliostats=1,
        heliostat_positions=torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device),
        initial_orientations=torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=device),
        translation_deviation_parameters=translation_deviation_parameters,
        rotation_deviation_parameters=rotation_deviation_parameters,
        actuator_parameters_non_optimizable=actuator_parameters_non_optimizable,
        actuator_parameters_optimizable=actuator_parameters_optimizable,
        device=device,
    )

    active_heliostats_mask = torch.ones(
        kinematic.number_of_heliostats, dtype=torch.int32, device=device
    )

    kinematic.number_of_active_heliostats = active_heliostats_mask.sum().item()
    kinematic.active_heliostat_positions = (
        kinematic.heliostat_positions.repeat_interleave(active_heliostats_mask, dim=0)
    )
    kinematic.active_initial_orientations = (
        kinematic.initial_orientations.repeat_interleave(active_heliostats_mask, dim=0)
    )
    kinematic.active_translation_deviation_parameters = (
        kinematic.translation_deviation_parameters.repeat_interleave(
            active_heliostats_mask, dim=0
        )
    )
    kinematic.active_rotation_deviation_parameters = (
        kinematic.rotation_deviation_parameters.repeat_interleave(
            active_heliostats_mask, dim=0
        )
    )
    kinematic.actuators.active_non_optimizable_parameters = (
        kinematic.actuators.non_optimizable_parameters.repeat_interleave(
            active_heliostats_mask, dim=0
        )
    )
    kinematic.actuators.active_optimizable_parameters = (
        kinematic.actuators.optimizable_parameters.repeat_interleave(
            active_heliostats_mask, dim=0
        )
    )

    orientation_matrix = kinematic(
        incident_ray_directions=incident_ray_directions.to(device),
        aim_points=aim_points.to(device),
        device=device,
    )
    torch.testing.assert_close(
        orientation_matrix, expected.to(device), atol=5e-4, rtol=5e-4
    )


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
            aim_points=torch.tensor([0.0, 0.0, 1.0, 1.0], device=device),
            device=device,
        )
    assert "Must be overridden!" in str(exc_info.value)

    with pytest.raises(NotImplementedError) as exc_info:
        abstract_kinematic.motor_positions_to_orientations(
            motor_positions=torch.tensor([1.0, 1.0], device=device)
        )
    assert "Must be overridden!" in str(exc_info.value)
