"""
Test kinematic module.
"""
import math
import torch
import pytest

from artist.io.datapoint import HeliostatDataPoint, HeliostatDataPointLabel
from artist.physics_objects.heliostats.alignment.rigid_body import (
    RigidBodyModule,
)


@pytest.fixture
def kinematic_model():
    """Declare a kinematic model at the origin"""
    position = torch.tensor([0, 0, 0])
    return RigidBodyModule(position=position)


@pytest.fixture
def kinematic_model2():
    """Declare a kinematic model placed one unit north"""
    position2 = torch.tensor([0.0, 1.0, 0.0])
    return RigidBodyModule(position=position2)


@pytest.fixture
def datapoints():
    """Declare multiple data points with different light directions to be tested"""
    datapoint1 = HeliostatDataPoint(
        point_id=1,
        light_directions=torch.tensor([0.0, 0.0, 1.0]),
        desired_aimpoint=torch.tensor([0, -10, 0]),
        label=HeliostatDataPointLabel(),
    )
    datapoint2 = HeliostatDataPoint(
        point_id=2,
        light_directions=torch.tensor([1.0, 0.0, 0.0]),
        desired_aimpoint=torch.tensor([0, -10, 0]),
        label=HeliostatDataPointLabel(),
    )
    datapoint3 = HeliostatDataPoint(
        point_id=3,
        light_directions=torch.tensor([0.0, -1.0, 0.0]),
        desired_aimpoint=torch.tensor([0, -10, 0]),
        label=HeliostatDataPointLabel(),
    )
    datapoint4 = HeliostatDataPoint(
        point_id=4,
        light_directions=torch.tensor([-1.0, 0.0, 0.0]),
        desired_aimpoint=torch.tensor([0, -10, 0]),
        label=HeliostatDataPointLabel(),
    )
    datapoint5 = HeliostatDataPoint(
        point_id=5,
        light_directions=torch.tensor([0.0, -1.0, 1.0]),
        desired_aimpoint=torch.tensor([0, -10, 0]),
        label=HeliostatDataPointLabel(),
    )
    datapoint6 = HeliostatDataPoint(
        point_id=6,
        light_directions=torch.tensor([0.0, 0.0, 1.0]),
        desired_aimpoint=torch.tensor([0, -9, 0]),
        label=HeliostatDataPointLabel(),
    )
    return [datapoint1, datapoint2, datapoint3, datapoint4, datapoint5, datapoint6]


# Create the test cases to be checked in the test function
@pytest.mark.parametrize(
    "datapoint_index, kinematic_model_fixture, expected",
    [
        (
            0,
            "kinematic_model",
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
            1,
            "kinematic_model",
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
            2,
            "kinematic_model",
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
            3,
            "kinematic_model",
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
            4,
            "kinematic_model",
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
            5,
            "kinematic_model2",
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
def test_compute_orientation_from_aimpoint(
    request,
    datapoints,
    datapoint_index,
    kinematic_model_fixture,
    expected,
):
    """Run the kinematic test for multiple datapoints and different kinematic models"""
    orientation_matrix = request.getfixturevalue(
        # selects which kinematic model to used based on the
        # kinematic_model_fixture parameter
        kinematic_model_fixture
    ).compute_orientation_from_aimpoint(datapoints[datapoint_index])
    torch.testing.assert_close(orientation_matrix[0], expected)
