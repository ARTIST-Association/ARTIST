import math
import pytest
import torch

from artist.physics_objects.heliostats.alignment.kinematic.rigid_body import (
    RigidBodyModule,
)


@pytest.fixture
def kinematic_model_1():
    position = torch.tensor([0.0, 0.0, 0.0, 1.0])
    aim_point = torch.tensor([0.0, -10.0, 0.0, 1.0])
    return RigidBodyModule("ideal_actuator", position, aim_point)


@pytest.fixture
def kinematic_model_2():
    position = torch.tensor([0.0, 1.0, 0.0, 1.0])
    aim_point = torch.tensor([0.0, -9.0, 0.0, 1.0])
    return RigidBodyModule("ideal_actuator", position, aim_point)


@pytest.mark.parametrize(
    "kinematic_model_fixture, incident_ray_direction, expected",
    [
        (
            "kinematic_model_1",
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
            torch.tensor(
                [
                    [1, 0, 0, 0],
                    [0, 1 / math.sqrt(2), -1 / math.sqrt(2), 0],
                    [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
                    [0, 0, 0, 1],
                ]
            ),
        ),
        (
            "kinematic_model_1",
            torch.tensor([1.0, 0.0, 0.0, 1.0]),
            torch.tensor(
                [
                    [1 / math.sqrt(2), 0, 1 / math.sqrt(2), 0],
                    [1 / math.sqrt(2), 0, -1 / math.sqrt(2), 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                ]
            ),
        ),
        (
            "kinematic_model_1",
            torch.tensor([0.0, -1.0, 0.0, 1.0]),
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        (
            "kinematic_model_1",
            torch.tensor([-1.0, 0.0, 0.0, 1.0]),
            torch.tensor(
                [
                    [1 / math.sqrt(2), 0.0, -1 / math.sqrt(2), 0.0],
                    [-1 / math.sqrt(2), 0.0, -1 / math.sqrt(2), 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        (
            "kinematic_model_1",
            torch.tensor([0.0, -1.0, 1.0, 1.0]),
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, math.sin(math.pi / 8), -math.cos(math.pi / 8), 0.0],
                    [0.0, math.cos(math.pi / 8), math.sin(math.pi / 8), 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        (
            "kinematic_model_2",
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
            torch.tensor(
                [
                    [1, 0, 0, 0],
                    [0, 1 / math.sqrt(2), -1 / math.sqrt(2), 1.0],
                    [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
                    [0, 0, 0, 1],
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
    orientation_matrix = request.getfixturevalue(
        kinematic_model_fixture
    ).compute_rotation_matrix_from_aimpoint(incident_ray_direction)
    torch.testing.assert_close(orientation_matrix, expected)
