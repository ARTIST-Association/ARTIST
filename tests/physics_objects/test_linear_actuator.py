import math

import pytest
import torch

from artist.physics_objects.actuator_linear import LinearActuator
from artist.physics_objects.kinematic_rigid_body import (
    RigidBodyModule,
)
from artist.util import config_dictionary

@pytest.fixture
def deviation_parameters():
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
    return RigidBodyModule(
        actuator_type=LinearActuator, position=position, aim_point=aim_point, deviation_parameters=deviation_parameters, actuator_parameters=actuator_parameters
    )


@pytest.fixture
def kinematic_model_2(deviation_parameters, actuator_parameters):
    """Create a kinematic model to use in the test."""
    position = torch.tensor([0.0, 1.0, 0.0, 1.0])
    aim_point = torch.tensor([0.0, -9.0, 0.0, 1.0])
    return RigidBodyModule(
        actuator_type=LinearActuator, position=position, aim_point=aim_point, deviation_parameters=deviation_parameters, actuator_parameters=actuator_parameters
    )


@pytest.mark.parametrize(
    "kinematic_model_fixture, incident_ray_direction, expected",
    [
        (
            "kinematic_model_1",
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
            torch.tensor(
                [
                    [1, 0, 0, 0],
                    [0, math.cos(-math.pi / 4), -math.sin(-math.pi / 4), 0],
                    [0, math.sin(-math.pi / 4), math.cos(-math.pi / 4), 0],
                    [0, 0, 0, 1],
                ]
            ),
        ),
        (
            "kinematic_model_1",
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor(
                [
                    [math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0],
                    [math.sin(math.pi / 4), math.cos(math.pi / 4), 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        (
            "kinematic_model_1",
            torch.tensor([0.0, -1.0, 0.0, 0.0]),
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        (
            "kinematic_model_1",
            torch.tensor([-1.0, 0.0, 0.0, 0.0]),
            torch.tensor(
                [
                    [math.cos(-math.pi / 4), -math.sin(-math.pi / 4), 0.0, 0.0],
                    [math.sin(-math.pi / 4), math.cos(-math.pi / 4), 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        (
            "kinematic_model_1",
            torch.tensor([0.0, -1.0, 1.0, 0.0]),
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, math.cos(-math.pi / 8), -math.sin(-math.pi / 8), 0.0],
                    [0.0, math.sin(-math.pi / 8), math.cos(-math.pi / 8), 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        (
            "kinematic_model_2",
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, math.cos(-math.pi / 4), -math.sin(-math.pi / 4), 1.0],
                    [0.0, math.sin(-math.pi / 4), math.cos(-math.pi / 4), 0.0],
                    [0.0, 0.0, 0.0, 1.0],
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
    """Tests that the alignment is working as desired."""
    orientation_matrix = request.getfixturevalue(kinematic_model_fixture).align(
        incident_ray_direction
    )
    torch.testing.assert_close(orientation_matrix[0], expected)
