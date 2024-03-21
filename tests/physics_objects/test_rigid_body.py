import math

import pytest
import torch

from artist.physics_objects.kinematic_rigid_body import (
    RigidBodyModule,
)


@pytest.fixture
def kinematic_model_1():
    """Create a kinematic model to use in the test."""
    position = torch.tensor([0.0, 0.0, 0.0, 1.0])
    aim_point = torch.tensor([0.0, -10.0, 0.0, 1.0])
    return RigidBodyModule(
        actuator_type="ideal_actuator", position=position, aim_point=aim_point
    )


@pytest.fixture
def kinematic_model_2():
    """Create a kinematic model to use in the test."""
    position = torch.tensor([0.0, 1.0, 0.0, 1.0])
    aim_point = torch.tensor([0.0, -9.0, 0.0, 1.0])
    return RigidBodyModule(
        actuator_type="ideal_actuator", position=position, aim_point=aim_point
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
