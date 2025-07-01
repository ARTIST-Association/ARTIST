import pytest
import torch

from artist.util import type_mappings


@pytest.mark.parametrize(
        "actuator_parameters, expected", 
        [
            (
                torch.tensor([[[ 0.0000e+00,  0.0000e+00],
                [ 0.0000e+00,  1.0000e+00],
                [ 1.5417e+05,  1.5417e+05],
                [ 7.7413e-02,  7.7522e-02],
                [ 3.3531e-01,  3.4077e-01],
                [ 3.3810e-01,  3.1910e-01],
                [-1.5318e+00,  9.4392e-01]]]),
                torch.tensor([0.0])
            ),
            (
                torch.tensor([[[ 1.0000e+00,  1.0000e+00],
                [ 0.0000e+00,  1.0000e+00]]]),
                torch.tensor([[28061.0], [47874.0]])
            )
    ])
def test_actuators_forward_adds_one(
    actuator_parameters: torch.Tensor,
    expected: torch.Tensor,
    device: torch.device
) -> None:
    actuators = type_mappings.actuator_type_mapping[
            actuator_parameters[0, 0, 0].item()
        ](actuator_parameters=actuator_parameters.to(device), device=device)
    actuators.active_actuator_parameters = actuators.actuator_parameters
    motor_positions = torch.tensor([[28061.0], [47874.0]], device=device)
    angles = actuators(motor_positions, device)

    assert torch.allclose(angles, expected.to(device))