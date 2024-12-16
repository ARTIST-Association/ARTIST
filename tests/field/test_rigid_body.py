import pytest
import torch

from artist.field.kinematic_rigid_body import (
    RigidBody,
)
from artist.util import config_dictionary
from artist.util.configuration_classes import (
    ActuatorConfig,
    ActuatorListConfig,
)


@pytest.fixture
def actuator_configuration_1() -> ActuatorListConfig:
    """
    Define actuator parameters used in tests.

    Returns
    -------
    ActuatorListConfig
        A List containing parameters for each actuator.
    """
    actuator1_config = ActuatorConfig(
        key="",
        type=config_dictionary.ideal_actuator_key,
        clockwise_axis_movement=False,
    )
    return ActuatorListConfig(actuator_list=[actuator1_config])


@pytest.fixture
def actuator_configuration_2() -> ActuatorListConfig:
    """
    Define actuator parameters used in tests.

    Returns
    -------
    ActuatorListConfig
        A List containing parameters for each actuator.
    """
    actuator1_config = ActuatorConfig(
        key="",
        type=config_dictionary.ideal_actuator_key,
        clockwise_axis_movement=False,
    )
    actuator2_config = ActuatorConfig(
        key="",
        type=config_dictionary.ideal_actuator_key,
        clockwise_axis_movement=True,
    )
    return ActuatorListConfig(actuator_list=[actuator1_config, actuator2_config])


@pytest.fixture
def actuator_configuration_3() -> ActuatorListConfig:
    """
    Define actuator parameters used in tests.

    Returns
    -------
    ActuatorListConfig
        A List containing parameters for each actuator.
    """
    actuator1_config = ActuatorConfig(
        key="",
        type=config_dictionary.ideal_actuator_key,
        clockwise_axis_movement=False,
    )
    actuator2_config = ActuatorConfig(
        key="",
        type=config_dictionary.ideal_actuator_key,
        clockwise_axis_movement=True,
    )
    actuator3_config = ActuatorConfig(
        key="",
        type=config_dictionary.ideal_actuator_key,
        clockwise_axis_movement=True,
    )
    return ActuatorListConfig(
        actuator_list=[actuator1_config, actuator2_config, actuator3_config]
    )


@pytest.fixture
def initial_orientation_south(device: torch.device) -> torch.Tensor:
    """
    Define initial orientation vector for a south-orientated heliostat.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    torch.Tensor
        Initial kinematic orientation vector for a south-orientated heliostat.
    """
    initial_orientation_south = torch.tensor([0.0, -1.0, 0.0, 0.0], device=device)
    return initial_orientation_south


@pytest.fixture
def initial_orientation_up(device: torch.device) -> torch.Tensor:
    """
    Define initial orientation vector for an up-orientated heliostat.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    torch.Tensor
        Initial kinematic orientation vector for an up-orientated heliostat.
    """
    initial_orientation_up = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device)
    return initial_orientation_up


@pytest.fixture
def kinematic_model_1(
    actuator_configuration_2: ActuatorListConfig,
    initial_orientation_south: torch.Tensor,
    device: torch.device,
) -> RigidBody:
    """
    Create a kinematic model to use in the test.

    Parameters
    ----------
    actuator_configuration_2 : ActuatorListConfig
        The configuration of the actuators.
    initial_orientation_south : torch.Tensor
        The kinematic initial orientation.
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    RigidBody
        The kinematic model.
    """
    position = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    aim_point = torch.tensor([0.0, -10.0, 0.0, 1.0], device=device)
    return RigidBody(
        position=position,
        aim_point=aim_point,
        actuator_config=actuator_configuration_2,
        initial_orientation=initial_orientation_south,
        device=device,
    )


@pytest.fixture
def kinematic_model_2(
    actuator_configuration_2: ActuatorListConfig,
    initial_orientation_south: torch.Tensor,
    device: torch.device,
) -> RigidBody:
    """
    Create a kinematic model to use in the test.

    Parameters
    ----------
    actuator_configuration_2 : ActuatorListConfig
        The configuration of the actuators.
    initial_orientation_south : torch.Tensor
        The kinematic initial orientation.
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    RigidBody
        The kinematic model.
    """
    position = torch.tensor([0.0, 1.0, 0.0, 1.0], device=device)
    aim_point = torch.tensor([0.0, -9.0, 0.0, 1.0], device=device)
    return RigidBody(
        position=position,
        aim_point=aim_point,
        actuator_config=actuator_configuration_2,
        initial_orientation=initial_orientation_south,
        device=device,
    )


@pytest.fixture
def kinematic_model_3(
    actuator_configuration_2: ActuatorListConfig,
    initial_orientation_up: torch.Tensor,
    device: torch.device,
) -> RigidBody:
    """
    Create a kinematic model to use in the test.

    Parameters
    ----------
    actuator_configuration_2 : ActuatorListConfig
        The configuration of the actuators.
    initial_orientation_up : torch.Tensor
        The kinematic initial orientation offsets.
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    RigidBody
        The kinematic model.
    """
    position = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    aim_point = torch.tensor([0.0, -10.0, 0.0, 1.0], device=device)
    return RigidBody(
        position=position,
        aim_point=aim_point,
        actuator_config=actuator_configuration_2,
        initial_orientation=initial_orientation_up,
        device=device,
    )


@pytest.fixture
def kinematic_model_4(
    actuator_configuration_1: ActuatorListConfig,
    initial_orientation_up: torch.Tensor,
    device: torch.device,
) -> RigidBody:
    """
    Create a kinematic model to use in the test.

    Parameters
    ----------
    actuator_configuration_1 : ActuatorListConfig
        The configuration of the actuators.
    initial_orientation_up : torch.Tensor
        The kinematic initial orientation offsets.
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    RigidBody
        The kinematic model.
    """
    position = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    aim_point = torch.tensor([0.0, -10.0, 0.0, 1.0], device=device)
    return RigidBody(
        position=position,
        aim_point=aim_point,
        actuator_config=actuator_configuration_1,
        initial_orientation=initial_orientation_up,
        device=device,
    )


@pytest.fixture
def kinematic_model_5(
    actuator_configuration_3: ActuatorListConfig,
    initial_orientation_up: torch.Tensor,
    device: torch.device,
) -> RigidBody:
    """
    Create a kinematic model to use in the test.

    Parameters
    ----------
    actuator_configuration_3 : ActuatorListConfig
        The configuration of the actuators.
    initial_orientation_up : torch.Tensor
        The kinematic initial orientation offsets.
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    RigidBody
        The kinematic model.
    """
    position = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    aim_point = torch.tensor([0.0, -10.0, 0.0, 1.0], device=device)
    return RigidBody(
        position=position,
        aim_point=aim_point,
        actuator_config=actuator_configuration_3,
        initial_orientation=initial_orientation_up,
        device=device,
    )


@pytest.mark.parametrize(
    "kinematic_model_fixture, input, method, expected",
    [
        (
            "kinematic_model_1",
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
            "incident_ray_direction_to_orientation",
            torch.tensor(
                [
                    [1, 0, 0, 0],
                    [
                        0,
                        torch.cos(torch.tensor(-torch.pi / 4)),
                        -torch.sin(torch.tensor(-torch.pi / 4)),
                        0,
                    ],
                    [
                        0,
                        torch.sin(torch.tensor(-torch.pi / 4)),
                        torch.cos(torch.tensor(-torch.pi / 4)),
                        0,
                    ],
                    [0, 0, 0, 1],
                ]
            ),
        ),
        (
            "kinematic_model_1",
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "incident_ray_direction_to_orientation",
            torch.tensor(
                [
                    [
                        torch.cos(torch.tensor(torch.pi / 4)),
                        -torch.sin(torch.tensor(torch.pi / 4)),
                        0.0,
                        0.0,
                    ],
                    [
                        torch.sin(torch.tensor(torch.pi / 4)),
                        torch.cos(torch.tensor(torch.pi / 4)),
                        0.0,
                        0.0,
                    ],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        (
            "kinematic_model_1",
            torch.tensor([0.0, -1.0, 0.0, 0.0]),
            "incident_ray_direction_to_orientation",
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
            "incident_ray_direction_to_orientation",
            torch.tensor(
                [
                    [
                        torch.cos(torch.tensor(-torch.pi / 4)),
                        -torch.sin(torch.tensor(-torch.pi / 4)),
                        0.0,
                        0.0,
                    ],
                    [
                        torch.sin(torch.tensor(-torch.pi / 4)),
                        torch.cos(torch.tensor(-torch.pi / 4)),
                        0.0,
                        0.0,
                    ],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        (
            "kinematic_model_1",
            torch.tensor([0.0, -1.0, 1.0, 0.0]),
            "incident_ray_direction_to_orientation",
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [
                        0.0,
                        torch.cos(torch.tensor(-torch.pi / 8)),
                        -torch.sin(torch.tensor(-torch.pi / 8)),
                        0.0,
                    ],
                    [
                        0.0,
                        torch.sin(torch.tensor(-torch.pi / 8)),
                        torch.cos(torch.tensor(-torch.pi / 8)),
                        0.0,
                    ],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        (
            "kinematic_model_2",
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
            "incident_ray_direction_to_orientation",
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [
                        0.0,
                        torch.cos(torch.tensor(-torch.pi / 4)),
                        -torch.sin(torch.tensor(-torch.pi / 4)),
                        1.0,
                    ],
                    [
                        0.0,
                        torch.sin(torch.tensor(-torch.pi / 4)),
                        torch.cos(torch.tensor(-torch.pi / 4)),
                        0.0,
                    ],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        (
            "kinematic_model_3",
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
            "incident_ray_direction_to_orientation",
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [
                        0.0,
                        torch.cos(torch.tensor(torch.pi / 4)),
                        -torch.sin(torch.tensor(torch.pi / 4)),
                        0.0,
                    ],
                    [
                        0.0,
                        torch.sin(torch.tensor(torch.pi / 4)),
                        torch.cos(torch.tensor(torch.pi / 4)),
                        0.0,
                    ],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        (
            "kinematic_model_3",
            torch.tensor([0.0, -1.0, 0.0, 0.0]),
            "incident_ray_direction_to_orientation",
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [
                        0.0,
                        torch.cos(torch.tensor(torch.pi / 2)),
                        -torch.sin(torch.tensor(torch.pi / 2)),
                        0.0,
                    ],
                    [
                        0.0,
                        torch.sin(torch.tensor(torch.pi / 2)),
                        torch.cos(torch.tensor(torch.pi / 2)),
                        0.0,
                    ],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        (
            "kinematic_model_3",
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "incident_ray_direction_to_orientation",
            torch.tensor(
                [
                    [
                        torch.cos(torch.tensor(torch.pi / 4)),
                        0.0,
                        torch.sin(torch.tensor(torch.pi / 4)),
                        0.0,
                    ],
                    [
                        torch.sin(torch.tensor(torch.pi / 4)),
                        0.0,
                        -torch.cos(torch.tensor(torch.pi / 4)),
                        0.0,
                    ],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        (
            "kinematic_model_3",
            torch.tensor([-1.0, 0.0, 0.0, 0.0]),
            "incident_ray_direction_to_orientation",
            torch.tensor(
                [
                    [
                        torch.cos(torch.tensor(-torch.pi / 4)),
                        0.0,
                        torch.sin(torch.tensor(-torch.pi / 4)),
                        0.0,
                    ],
                    [
                        torch.sin(torch.tensor(-torch.pi / 4)),
                        0.0,
                        -torch.cos(torch.tensor(-torch.pi / 4)),
                        0.0,
                    ],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        (
            "kinematic_model_4",
            torch.tensor([-1.0, 0.0, 0.0, 0.0]),
            "incident_ray_direction_to_orientation",
            None,
        ),
        (
            "kinematic_model_5",
            torch.tensor([-1.0, 0.0, 0.0, 0.0]),
            "incident_ray_direction_to_orientation",
            None,
        ),
        (
            "kinematic_model_1",
            torch.tensor([0.0, 0.0]),
            "motor_positions_to_orientation",
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
            "kinematic_model_2",
            torch.tensor([5.0, 1000.0]),
            "motor_positions_to_orientation",
            torch.tensor(
                [
                    [0.562379062176, -0.826879560947, 0.000000000000, 0.000000000000],
                    [0.234554469585, 0.159525677562, 0.958924293518, 1.000000000000],
                    [-0.792914927006, -0.539278924465, 0.283662199974, 0.000000000000],
                    [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000],
                ]
            ),
        ),
        (
            "kinematic_model_3",
            torch.tensor([10.0, 40.0]),
            "motor_positions_to_orientation",
            torch.tensor(
                [
                    [-6.6694e-01, 3.2570e-08, 7.4511e-01, 0.0000e00],
                    [-6.2520e-01, 5.4402e-01, -5.5961e-01, 0.0000e00],
                    [-4.0536e-01, -8.3907e-01, -3.6283e-01, 0.0000e00],
                    [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                ]
            ),
        ),
        (
            "kinematic_model_4",
            torch.tensor([0.0, 0.0]),
            "motor_positions_to_orientation",
            None,
        ),
        (
            "kinematic_model_5",
            torch.tensor([0.0, 0.0]),
            "motor_positions_to_orientation",
            None,
        ),
    ],
)
def test_orientation_matrix(
    request: pytest.FixtureRequest,
    kinematic_model_fixture: str,
    input: torch.Tensor,
    method,
    expected: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test that the alignment is working as desired.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The pytest fixture used to consider different test cases.
    kinematic_model_fixture : str
        The kinematic model fixture used to select the kinematic model used in the test case.
    input : torch.Tensor

    method

    expected : torch.Tensor
        The expected orientation matrix or ``None`` if an error is expected.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    # Check if the ValueError is thrown as expected.
    get_orientation = getattr(request.getfixturevalue(kinematic_model_fixture), method)
    if expected is None:
        with pytest.raises(ValueError) as exc_info:
            get_orientation(input.to(device), device=device)
        assert (
            f"The rigid body kinematic requires exactly two actuators but {len(request.getfixturevalue(kinematic_model_fixture).actuators.actuator_list)} were specified, please check the configuration!"
            in str(exc_info.value)
        )
    else:
        # Check if the orientation matrix is correct.
        orientation_matrix = get_orientation(input.to(device), device=device)
        torch.testing.assert_close(orientation_matrix, expected.to(device))
