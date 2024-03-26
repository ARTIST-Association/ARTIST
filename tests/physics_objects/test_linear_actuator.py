import pytest
import torch

from artist.field.actuator_linear import LinearActuator
from artist.field.kinematic_rigid_body import (
    RigidBody,
)
from artist.util import config_dictionary


@pytest.fixture
def deviation_parameters():
    """Define deviation parameters used in tests."""
    deviation_parameters = {
        config_dictionary.first_joint_translation_e: torch.tensor(0.0),
        config_dictionary.first_joint_translation_n: torch.tensor(0.0),
        config_dictionary.first_joint_translation_u: torch.tensor(0.0),
        config_dictionary.first_joint_tilt_e: torch.tensor(0.0),
        config_dictionary.first_joint_tilt_n: torch.tensor(0.0),
        config_dictionary.first_joint_tilt_u: torch.tensor(0.0),
        config_dictionary.second_joint_translation_e: torch.tensor(0.0),
        config_dictionary.second_joint_translation_n: torch.tensor(0.0),
        config_dictionary.second_joint_translation_u: torch.tensor(0.315),
        config_dictionary.second_joint_tilt_e: torch.tensor(0.0),
        config_dictionary.second_joint_tilt_n: torch.tensor(0.0),
        config_dictionary.second_joint_tilt_u: torch.tensor(0.0),
        config_dictionary.concentrator_translation_e: torch.tensor(0.0),
        config_dictionary.concentrator_translation_n: torch.tensor(-0.17755),
        config_dictionary.concentrator_translation_u: torch.tensor(-0.4045),
        config_dictionary.concentrator_tilt_e: torch.tensor(0.0),
        config_dictionary.concentrator_tilt_n: torch.tensor(0.0),
        config_dictionary.concentrator_tilt_u: torch.tensor(0.0),
    }
    return deviation_parameters


@pytest.fixture
def actuator_parameters():
    """Define actuator parameters used in tests as measured experimentally."""
    actuator_parameters = {
        config_dictionary.first_joint_increment: torch.tensor(154166.666),
        config_dictionary.first_joint_initial_stroke_length: torch.tensor(0.075),
        config_dictionary.first_joint_actuator_offset: torch.tensor(0.34061),
        config_dictionary.first_joint_radius: torch.tensor(0.3204),
        config_dictionary.first_joint_phi_0: torch.tensor(-1.570796),
        config_dictionary.second_joint_increment: torch.tensor(154166.666),
        config_dictionary.second_joint_initial_stroke_length: torch.tensor(0.075),
        config_dictionary.second_joint_actuator_offset: torch.tensor(0.3479),
        config_dictionary.second_joint_radius: torch.tensor(0.309),
        config_dictionary.second_joint_phi_0: torch.tensor(0.959931),
    }
    return actuator_parameters


@pytest.fixture
def kinematic_model_1(deviation_parameters, actuator_parameters):
    """Create a kinematic model to use in the test."""
    position = torch.tensor([0.0, 0.0, 0.0, 1.0])
    aim_point = torch.tensor([0.0, -10.0, 0.0, 1.0])
    return RigidBody(
        actuator_type=LinearActuator,
        position=position,
        aim_point=aim_point,
        deviation_parameters=deviation_parameters,
        actuator_parameters=actuator_parameters,
    )


@pytest.fixture
def kinematic_model_2(deviation_parameters, actuator_parameters):
    """Create a kinematic model to use in the test."""
    position = torch.tensor([0.0, 1.0, 0.0, 1.0])
    aim_point = torch.tensor([0.0, -9.0, 0.0, 1.0])
    return RigidBody(
        actuator_type=LinearActuator,
        position=position,
        aim_point=aim_point,
        deviation_parameters=deviation_parameters,
        actuator_parameters=actuator_parameters,
    )


@pytest.mark.parametrize(
    "kinematic_model_fixture, incident_ray_direction, expected",
    [
        (
            "kinematic_model_1",
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
            torch.tensor(
                [
                    [0.9999, 0.0104, 0.0000, -0.0019],
                    [-0.0074, 0.7107, 0.7035, -0.1891],
                    [0.0073, -0.7035, 0.7107, 0.0613],
                    [0.0000, 0.0000, 0.0000, 1.0000],
                ]
            ),
        ),
        (
            "kinematic_model_1",
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor(
                [
                    [0.7123, -0.7019, 0.0000, 0.1246],
                    [0.7019, 0.7122, -0.0103, -0.1255],
                    [0.0072, 0.0073, 0.9999, -0.0908],
                    [0.0000, 0.0000, 0.0000, 1.0000],
                ]
            ),
        ),
        (
            "kinematic_model_1",
            torch.tensor([0.0, -1.0, 0.0, 0.0]),
            torch.tensor(
                [
                    [9.9997e-01, 7.3368e-03, 0.0000e00, -1.3026e-03],
                    [-7.3367e-03, 9.9996e-01, -5.1375e-03, -1.7708e-01],
                    [-3.7693e-05, 5.1374e-03, 9.9999e-01, -9.0411e-02],
                    [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                ]
            ),
        ),
        (
            "kinematic_model_1",
            torch.tensor([-1.0, 0.0, 0.0, 0.0]),
            torch.tensor(
                [
                    [0.7019, 0.7123, 0.0000, -0.1265],
                    [-0.7122, 0.7019, -0.0103, -0.1237],
                    [-0.0073, 0.0072, 0.9999, -0.0908],
                    [0.0000, 0.0000, 0.0000, 1.0000],
                ]
            ),
        ),
        (
            "kinematic_model_1",
            torch.tensor([0.0, -1.0, 1.0, 0.0]),
            torch.tensor(
                [
                    [1.0000, 0.0080, 0.0000, -0.0014],
                    [-0.0074, 0.9258, 0.3780, -0.1982],
                    [0.0030, -0.3779, 0.9258, -0.0158],
                    [0.0000, 0.0000, 0.0000, 1.0000],
                ]
            ),
        ),
        (
            "kinematic_model_2",
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
            torch.tensor(
                [
                    [0.9999, 0.0104, 0.0000, -0.0019],
                    [-0.0074, 0.7107, 0.7035, 0.8109],
                    [0.0073, -0.7035, 0.7107, 0.0613],
                    [0.0000, 0.0000, 0.0000, 1.0000],
                ]
            ),
        ),
    ],
)
def test_orientation_matrix(
    request,
    kinematic_model_fixture,
    incident_ray_direction,
    expected,
):
    """Test that the alignment works as desired."""
    orientation_matrix = request.getfixturevalue(kinematic_model_fixture).align(
        incident_ray_direction
    )
    torch.testing.assert_close(orientation_matrix[0], expected, atol=5e-4, rtol=5e-4)
