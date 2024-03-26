import math

import pytest
import torch

from artist.field.actuator_ideal import IdealActuator
from artist.field.kinematic_rigid_body import (
    RigidBody,
)
from artist.util import config_dictionary


@pytest.fixture
def initial_offsets_south():
    """Define initial offsets for a south orientated heliostat."""
    initial_offsets = {
        config_dictionary.kinematic_initial_orientation_offset_e: torch.tensor(0.0),
        config_dictionary.kinematic_initial_orientation_offset_n: torch.tensor(0.0),
        config_dictionary.kinematic_initial_orientation_offset_u: torch.tensor(0.0),
    }
    return initial_offsets


@pytest.fixture
def initial_offsets_above():
    """Define initial offsets for a up orientated heliostat."""
    initial_offsets = {
        config_dictionary.kinematic_initial_orientation_offset_e: torch.tensor(
            math.pi / 2
        ),
        config_dictionary.kinematic_initial_orientation_offset_n: torch.tensor(0.0),
        config_dictionary.kinematic_initial_orientation_offset_u: torch.tensor(0.0),
    }
    return initial_offsets


@pytest.fixture
def kinematic_model_1(initial_offsets_south):
    """Create a kinematic model to use in the test."""
    position = torch.tensor([0.0, 0.0, 0.0, 1.0])
    aim_point = torch.tensor([0.0, -10.0, 0.0, 1.0])
    return RigidBody(
        actuator_type=IdealActuator,
        position=position,
        aim_point=aim_point,
        initial_orientation_offsets=initial_offsets_south,
    )


@pytest.fixture
def kinematic_model_2():
    """Create a kinematic model to use in the test."""
    position = torch.tensor([0.0, 1.0, 0.0, 1.0])
    aim_point = torch.tensor([0.0, -9.0, 0.0, 1.0])
    return RigidBody(
        actuator_type=IdealActuator, position=position, aim_point=aim_point
    )


@pytest.fixture
def kinematic_model_3(initial_offsets_above):
    """Create a kinematic model to use in the test."""
    position = torch.tensor([0.0, 0.0, 0.0, 1.0])
    aim_point = torch.tensor([0.0, -10.0, 0.0, 1.0])
    return RigidBody(
        actuator_type=IdealActuator,
        position=position,
        aim_point=aim_point,
        initial_orientation_offsets=initial_offsets_above,
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
        (
            "kinematic_model_3",
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0],
                    [0.0, math.sin(math.pi / 4), math.cos(math.pi / 4), 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        (
            "kinematic_model_3",
            torch.tensor([0.0, -1.0, 0.0, 0.0]),
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, math.cos(math.pi / 2), -math.sin(math.pi / 2), 0.0],
                    [0.0, math.sin(math.pi / 2), math.cos(math.pi / 2), 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        (
            "kinematic_model_3",
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor(
                [
                    [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0],
                    [math.sin(math.pi / 4), 0.0, -math.cos(math.pi / 4), 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        (
            "kinematic_model_3",
            torch.tensor([-1.0, 0.0, 0.0, 0.0]),
            torch.tensor(
                [
                    [math.cos(-math.pi / 4), 0.0, math.sin(-math.pi / 4), 0.0],
                    [math.sin(-math.pi / 4), 0.0, -math.cos(-math.pi / 4), 0.0],
                    [0.0, 1.0, 0.0, 0.0],
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
