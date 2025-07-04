import pytest
import torch

from artist.field.actuators import Actuators
from artist.util import type_mappings


@pytest.mark.parametrize(
    "actuator_parameters, expected",
    [
        (
            torch.tensor(
                [
                    [
                        [0.0000e00, 0.0000e00],
                        [0.0000e00, 1.0000e00],
                        [1.5417e05, 1.5417e05],
                        [7.7413e-02, 7.7522e-02],
                        [3.3531e-01, 3.4077e-01],
                        [3.3810e-01, 3.1910e-01],
                        [-1.5318e00, 9.4392e-01],
                    ]
                ]
            ),
            torch.tensor([[-0.971173584461, -0.085735797882]]),
        ),
        (
            torch.tensor([[[1.0000e00, 1.0000e00], [0.0000e00, 1.0000e00]]]),
            torch.tensor([[28061.0, 47874.0]]),
        ),
    ],
)
def test_actuators_forward(
    actuator_parameters: torch.Tensor, expected: torch.Tensor, device: torch.device
) -> None:
    """
    Test the forward method of actuators.

    Parameters
    ----------
    actuator_parameters : torch.Tensor
        The actuator parameters.
    expected : torch.Tensor
        The expected test result.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    actuators = type_mappings.actuator_type_mapping[
        actuator_parameters[0, 0, 0].item()
    ](actuator_parameters=actuator_parameters.to(device), device=device)
    actuators.active_actuator_parameters = actuators.actuator_parameters
    motor_positions = torch.tensor([[28061.0, 47874.0]], device=device)
    angles = actuators(motor_positions, device)

    torch.testing.assert_close(angles, expected.to(device))


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
