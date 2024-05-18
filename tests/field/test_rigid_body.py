import math

import pytest
import torch

from artist.field.kinematic_rigid_body import (
    RigidBody,
)
from artist.util import config_dictionary
from artist.util.configuration_classes import (
    ActuatorConfig,
    ActuatorListConfig,
    KinematicOffsets,
)


@pytest.fixture
def actuator_configuration() -> ActuatorListConfig:
    """Define actuator parameters used in tests."""
    actuator1_config = ActuatorConfig(
        actuator_key="",
        actuator_type=config_dictionary.ideal_actuator_key,
        actuator_clockwise=False,
    )
    actuator2_config = ActuatorConfig(
        actuator_key="",
        actuator_type=config_dictionary.ideal_actuator_key,
        actuator_clockwise=True,
    )
    return ActuatorListConfig(actuator_list=[actuator1_config, actuator2_config])


@pytest.fixture
def initial_offsets_south() -> KinematicOffsets:
    """Define initial offsets for a south-orientated heliostat."""
    initial_offsets = KinematicOffsets(
        kinematic_initial_orientation_offset_e=torch.tensor(0.0),
        kinematic_initial_orientation_offset_n=torch.tensor(0.0),
        kinematic_initial_orientation_offset_u=torch.tensor(0.0),
    )
    return initial_offsets


@pytest.fixture
def initial_offsets_above() -> KinematicOffsets:
    """Define initial offsets for an up-orientated heliostat."""
    initial_offsets = KinematicOffsets(
        kinematic_initial_orientation_offset_e=torch.tensor(math.pi / 2),
        kinematic_initial_orientation_offset_n=torch.tensor(0.0),
        kinematic_initial_orientation_offset_u=torch.tensor(0.0),
    )
    return initial_offsets


@pytest.fixture
def kinematic_model_1(actuator_configuration, initial_offsets_south) -> RigidBody:
    """Create a kinematic model to use in the test."""
    position = torch.tensor([0.0, 0.0, 0.0, 1.0])
    aim_point = torch.tensor([0.0, -10.0, 0.0, 1.0])
    return RigidBody(
        position=position,
        aim_point=aim_point,
        actuator_config=actuator_configuration,
        initial_orientation_offsets=initial_offsets_south,
    )


@pytest.fixture
def kinematic_model_2(actuator_configuration, initial_offsets_south) -> RigidBody:
    """Create a kinematic model to use in the test."""
    position = torch.tensor([0.0, 1.0, 0.0, 1.0])
    aim_point = torch.tensor([0.0, -9.0, 0.0, 1.0])
    return RigidBody(
        position=position,
        aim_point=aim_point,
        actuator_config=actuator_configuration,
        initial_orientation_offsets=initial_offsets_south,
    )


@pytest.fixture
def kinematic_model_3(actuator_configuration, initial_offsets_above) -> RigidBody:
    """Create a kinematic model to use in the test."""
    position = torch.tensor([0.0, 0.0, 0.0, 1.0])
    aim_point = torch.tensor([0.0, -10.0, 0.0, 1.0])
    return RigidBody(
        position=position,
        aim_point=aim_point,
        actuator_config=actuator_configuration,
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
) -> None:
    """Test that the alignment is working as desired."""
    orientation_matrix = request.getfixturevalue(kinematic_model_fixture).align(
        incident_ray_direction
    )
    torch.testing.assert_close(orientation_matrix[0], expected)
