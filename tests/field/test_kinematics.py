import pytest
import torch

from artist.field import Kinematics, RigidBody


@pytest.mark.parametrize(
    "expected",
    [
        torch.tensor(
            [
                [
                    [
                        1.000000000000e00,
                        7.034584910024e-14,
                        1.609325408936e-06,
                        -2.858161849417e-07,
                    ],
                    [
                        1.130841724262e-06,
                        7.115054130554e-01,
                        -7.026806473732e-01,
                        1.247960701585e-01,
                    ],
                    [
                        -1.145043825090e-06,
                        7.026806473732e-01,
                        7.115054130554e-01,
                        -1.263633668423e-01,
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
def test_kinematics_forward(expected: torch.Tensor, device: torch.device) -> None:
    """
    Test the forward method of the kinematics.

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
                0.0000,
                0.0000,
                0.1776,
                0.0000,
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

    kinematics = RigidBody(
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
        kinematics.number_of_heliostats, dtype=torch.int32, device=device
    )

    kinematics.number_of_active_heliostats = active_heliostats_mask.sum().item()
    kinematics.active_heliostat_positions = (
        kinematics.heliostat_positions.repeat_interleave(active_heliostats_mask, dim=0)
    )
    kinematics.active_initial_orientations = (
        kinematics.initial_orientations.repeat_interleave(active_heliostats_mask, dim=0)
    )
    kinematics.active_translation_deviation_parameters = (
        kinematics.translation_deviation_parameters.repeat_interleave(
            active_heliostats_mask, dim=0
        )
    )
    kinematics.active_rotation_deviation_parameters = (
        kinematics.rotation_deviation_parameters.repeat_interleave(
            active_heliostats_mask, dim=0
        )
    )
    kinematics.actuators.active_non_optimizable_parameters = (
        kinematics.actuators.non_optimizable_parameters.repeat_interleave(
            active_heliostats_mask, dim=0
        )
    )
    kinematics.actuators.active_optimizable_parameters = (
        kinematics.actuators.optimizable_parameters.repeat_interleave(
            active_heliostats_mask, dim=0
        )
    )

    orientation_matrix = kinematics(
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
    Test the abstract methods of the kinematics.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    abstract_kinematics = Kinematics()

    with pytest.raises(NotImplementedError) as exc_info:
        abstract_kinematics.incident_ray_directions_to_orientations(
            incident_ray_directions=torch.tensor([0.0, 0.0, 1.0, 0.0], device=device),
            aim_points=torch.tensor([0.0, 0.0, 1.0, 1.0], device=device),
            device=device,
        )
    assert "Must be overridden!" in str(exc_info.value)

    with pytest.raises(NotImplementedError) as exc_info:
        abstract_kinematics.motor_positions_to_orientations(
            motor_positions=torch.tensor([1.0, 1.0], device=device)
        )
    assert "Must be overridden!" in str(exc_info.value)
