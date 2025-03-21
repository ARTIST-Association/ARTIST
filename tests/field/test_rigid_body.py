import pytest
import torch

from artist.field.kinematic_rigid_body import (
    RigidBody,
)


@pytest.fixture
def actuator_parameters(device: torch.device) -> torch.Tensor:
    """
    Define actuator parameters used in tests.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    torch.Tensor
        The actuator parameters for two ideal actuators.
    """
    parameters = torch.zeros((1, 7, 2), device=device)

    parameters[:, 0, :] = 1
    parameters[:, 1, 1] = 1

    return parameters


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
    initial_orientation_south = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=device)
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
    initial_orientation_up = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=device)
    return initial_orientation_up


@pytest.fixture
def kinematic_model_1(
    actuator_parameters: torch.Tensor,
    initial_orientation_south: torch.Tensor,
    device: torch.device,
) -> RigidBody:
    """
    Create a kinematic model to use in the test.

    Parameters
    ----------
    actuator_parameters : torch.Tensor
        The actuator parameters.
    initial_orientation_south : torch.Tensor
        The kinematic initial orientation.
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    RigidBody
        The kinematic model.
    """
    position = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
    aim_point = torch.tensor([[0.0, -10.0, 0.0, 1.0]], device=device)
    return RigidBody(
        number_of_heliostats=1,
        heliostat_positions=position,
        aim_points=aim_point,
        actuator_parameters=actuator_parameters,
        initial_orientations=initial_orientation_south,
        deviation_parameters=torch.zeros((1, 18), dtype=torch.float, device=device),
        device=device,
    )


@pytest.fixture
def kinematic_model_2(
    actuator_parameters: torch.Tensor,
    initial_orientation_south: torch.Tensor,
    device: torch.device,
) -> RigidBody:
    """
    Create a kinematic model to use in the test.

    Parameters
    ----------
    actuator_parameters : torch.Tensor
        The actuator parameters.
    initial_orientation_south : torch.Tensor
        The kinematic initial orientation.
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    RigidBody
        The kinematic model.
    """
    position = torch.tensor([[0.0, 1.0, 0.0, 1.0]], device=device)
    aim_point = torch.tensor([[0.0, -9.0, 0.0, 1.0]], device=device)
    return RigidBody(
        number_of_heliostats=1,
        heliostat_positions=position,
        aim_points=aim_point,
        actuator_parameters=actuator_parameters,
        initial_orientations=initial_orientation_south,
        deviation_parameters=torch.zeros((1, 18), dtype=torch.float, device=device),
        device=device,
    )


@pytest.fixture
def kinematic_model_3(
    actuator_parameters: torch.Tensor,
    initial_orientation_up: torch.Tensor,
    device: torch.device,
) -> RigidBody:
    """
    Create a kinematic model to use in the test.

    Parameters
    ----------
    actuator_parameters : torch.Tensor
        The actuator parameters.
    initial_orientation_up : torch.Tensor
        The kinematic initial orientation offsets.
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    RigidBody
        The kinematic model.
    """
    position = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
    aim_point = torch.tensor([[0.0, -10.0, 0.0, 1.0]], device=device)
    return RigidBody(
        number_of_heliostats=1,
        heliostat_positions=position,
        aim_points=aim_point,
        actuator_parameters=actuator_parameters,
        initial_orientations=initial_orientation_up,
        deviation_parameters=torch.zeros((1, 18), dtype=torch.float, device=device),
        device=device,
    )


@pytest.mark.parametrize(
    "kinematic_model_fixture, input, method, expected",
    [
        (
            "kinematic_model_1",
            torch.tensor([[0.0, 0.0, -1.0, 0.0]]),
            "incident_ray_direction_to_orientations",
            torch.tensor(
                [
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
                ]
            ),
        ),
        (
            "kinematic_model_1",
            torch.tensor([[-1.0, 0.0, 0.0, 0.0]]),
            "incident_ray_direction_to_orientations",
            torch.tensor(
                [
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
                ]
            ),
        ),
        (
            "kinematic_model_1",
            torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
            "incident_ray_direction_to_orientations",
            torch.tensor(
                [
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ]
            ),
        ),
        (
            "kinematic_model_1",
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "incident_ray_direction_to_orientations",
            torch.tensor(
                [
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
                ]
            ),
        ),
        (
            "kinematic_model_1",
            torch.tensor([[0.0000, 0.7071, -0.7071, 0.0000]]),
            "incident_ray_direction_to_orientations",
            torch.tensor(
                [
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
                ]
            ),
        ),
        (
            "kinematic_model_2",
            torch.tensor([[0.0, 0.0, -1.0, 0.0]]),
            "incident_ray_direction_to_orientations",
            torch.tensor(
                [
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
                ]
            ),
        ),
        (
            "kinematic_model_3",
            torch.tensor([[0.0, 0.0, -1.0, 0.0]]),
            "incident_ray_direction_to_orientations",
            torch.tensor(
                [
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
                ]
            ),
        ),
        (
            "kinematic_model_3",
            torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
            "incident_ray_direction_to_orientations",
            torch.tensor(
                [
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
                ]
            ),
        ),
        (
            "kinematic_model_3",
            torch.tensor([[-1.0, 0.0, 0.0, 0.0]]),
            "incident_ray_direction_to_orientations",
            torch.tensor(
                [
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
                ]
            ),
        ),
        (
            "kinematic_model_3",
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "incident_ray_direction_to_orientations",
            torch.tensor(
                [
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
                ]
            ),
        ),
        (
            "kinematic_model_1",
            torch.tensor([[0.0, 0.0]]),
            "motor_positions_to_orientations",
            torch.tensor(
                [
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ]
            ),
        ),
        (
            "kinematic_model_2",
            torch.tensor([[5.0, 1000.0]]),
            "motor_positions_to_orientations",
            torch.tensor(
                [
                    [
                        [
                            0.562379062176,
                            -0.826879560947,
                            0.000000000000,
                            0.000000000000,
                        ],
                        [
                            0.234554469585,
                            0.159525677562,
                            0.958924293518,
                            1.000000000000,
                        ],
                        [
                            -0.792914927006,
                            -0.539278924465,
                            0.283662199974,
                            0.000000000000,
                        ],
                        [
                            0.000000000000,
                            0.000000000000,
                            0.000000000000,
                            1.000000000000,
                        ],
                    ]
                ]
            ),
        ),
        (
            "kinematic_model_3",
            torch.tensor([[10.0, 40.0]]),
            "motor_positions_to_orientations",
            torch.tensor(
                [
                    [
                        [-6.6694e-01, 3.2570e-08, 7.4511e-01, 0.0000e00],
                        [-6.2520e-01, 5.4402e-01, -5.5961e-01, 0.0000e00],
                        [-4.0536e-01, -8.3907e-01, -3.6283e-01, 0.0000e00],
                        [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                    ]
                ]
            ),
        ),
    ],
)
def test_orientation_matrix(
    request: pytest.FixtureRequest,
    kinematic_model_fixture: str,
    input: torch.Tensor,
    method: str,
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
        The input to the kinematic orientation function (either an incident ray direction or motor positions).
    method : str
        Name of the kinematic orientation function.
    expected : torch.Tensor
        The expected orientation matrix.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    get_orientation = getattr(request.getfixturevalue(kinematic_model_fixture), method)

    # Check if the orientation matrix is correct.
    orientation_matrix = get_orientation(input.to(device), device=device)
    torch.testing.assert_close(orientation_matrix, expected.to(device))
