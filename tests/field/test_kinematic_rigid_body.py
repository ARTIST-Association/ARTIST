import pytest
import torch

from artist.field.kinematic_rigid_body import (
    RigidBody,
)


@pytest.fixture
def kinematic_parameters(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Define kinematic parameters used in tests.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    torch.Tensor
        The translation deviation parameters.
    torch.Tensor
        The rotation deviation parameters.
    """
    translation_deviation_parameters = torch.zeros((6, 9), device=device)

    translation_deviation_parameters[:, 5] = 0.315
    translation_deviation_parameters[:, 7] = -0.17755
    translation_deviation_parameters[:, 8] = -0.4045

    rotation_deviation_parameters = torch.zeros((6, 4), device=device)

    return translation_deviation_parameters, rotation_deviation_parameters


@pytest.fixture
def kinematic_model_linear(
    kinematic_parameters: torch.Tensor,
    device: torch.device,
) -> RigidBody:
    """
    Create a rigid body kinematic with linear actuators and deviation parameters.

    Parameters
    ----------
    kinematic_parameters : torch.Tensor
        The kinematic deviation parameters.
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    RigidBody
        The kinematic model.
    """
    heliostat_positions = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        device=device,
    )
    initial_orientation = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=device)

    linear_actuator_parameters = torch.zeros((6, 9, 2), device=device)
    linear_actuator_parameters[:, 1, 1] = 1
    linear_actuator_parameters[:, 4, 0] = 154166.666
    linear_actuator_parameters[:, 4, 1] = 154166.666
    linear_actuator_parameters[:, 5, 0] = 0.34061
    linear_actuator_parameters[:, 5, 1] = 0.3479
    linear_actuator_parameters[:, 6, 0] = 0.3204
    linear_actuator_parameters[:, 6, 1] = 0.309
    linear_actuator_parameters[:, 7, 0] = -1.570796
    linear_actuator_parameters[:, 7, 1] = 0.959931
    linear_actuator_parameters[:, 8, :] = 0.075

    return RigidBody(
        number_of_heliostats=6,
        heliostat_positions=heliostat_positions,
        initial_orientations=initial_orientation.expand(6, 4),
        translation_deviation_parameters=kinematic_parameters[0],
        rotation_deviation_parameters=kinematic_parameters[1],
        actuator_parameters=linear_actuator_parameters,
        device=device,
    )


@pytest.fixture
def kinematic_model_ideal_1(
    device: torch.device,
) -> RigidBody:
    """
    Create a rigid body kinematic with ideal actuators and no deviation parameters.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    RigidBody
        The kinematic model.
    """
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
    )
    initial_orientations = torch.tensor(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    )

    actuator_parameters = torch.zeros((10, 4, 2), device=device)
    actuator_parameters[:, 0, :] = 1
    actuator_parameters[:, 1, 1] = 1

    return RigidBody(
        number_of_heliostats=10,
        heliostat_positions=positions,
        actuator_parameters=actuator_parameters,
        initial_orientations=initial_orientations,
        translation_deviation_parameters=torch.zeros(
            (10, 9), dtype=torch.float, device=device
        ),
        rotation_deviation_parameters=torch.zeros(
            (10, 4), dtype=torch.float, device=device
        ),
        device=device,
    )


@pytest.fixture
def kinematic_model_ideal_2(
    device: torch.device,
) -> RigidBody:
    """
    Create rigid body kinematic with ideal actuators and deviation parameters.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    RigidBody
        The kinematic model.
    """
    positions = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        device=device,
    )
    initial_orientations = torch.tensor(
        [[0.0, -1.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        device=device,
    )

    actuator_parameters = torch.zeros((3, 4, 2), device=device)
    actuator_parameters[:, 0, :] = 1
    actuator_parameters[:, 1, 1] = 1

    return RigidBody(
        number_of_heliostats=3,
        heliostat_positions=positions,
        actuator_parameters=actuator_parameters,
        initial_orientations=initial_orientations,
        translation_deviation_parameters=torch.zeros(
            (3, 9), dtype=torch.float, device=device
        ),
        rotation_deviation_parameters=torch.zeros(
            (3, 4), dtype=torch.float, device=device
        ),
        device=device,
    )


@pytest.mark.parametrize(
    "kinematic_model_fixture, aim_points, incident_ray_directions, expected",
    [
        (
            "kinematic_model_linear",
            torch.tensor(
                [
                    [0.0, -10.0, 0.0, 1.0],
                    [0.0, -10.0, 0.0, 1.0],
                    [0.0, -10.0, 0.0, 1.0],
                    [0.0, -10.0, 0.0, 1.0],
                    [0.0, -10.0, 0.0, 1.0],
                    [0.0, -9.0, 0.0, 1.0],
                ]
            ),
            torch.tensor(
                [
                    [0.0, 0.0, -1.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0000, 0.7071, -0.7071, 0.0000],
                    [0.0, 0.0, -1.0, 0.0],
                ]
            ),
            torch.tensor(
                [
                    [
                        [0.9999, 0.0104, 0.0000, -0.0019],
                        [-0.0074, 0.7107, 0.7035, -0.1891],
                        [0.0073, -0.7035, 0.7107, 0.0613],
                        [0.0000, 0.0000, 0.0000, 1.0000],
                    ],
                    [
                        [0.7123, -0.7019, 0.0000, 0.1246],
                        [0.7019, 0.7122, -0.0103, -0.1255],
                        [0.0072, 0.0073, 0.9999, -0.0908],
                        [0.0000, 0.0000, 0.0000, 1.0000],
                    ],
                    [
                        [9.9997e-01, 7.3368e-03, 0.0000e00, -1.3026e-03],
                        [-7.3367e-03, 9.9996e-01, -5.1375e-03, -1.7708e-01],
                        [-3.7693e-05, 5.1374e-03, 9.9999e-01, -9.0411e-02],
                        [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                    ],
                    [
                        [0.7019, 0.7123, 0.0000, -0.1265],
                        [-0.7122, 0.7019, -0.0103, -0.1237],
                        [-0.0073, 0.0072, 0.9999, -0.0908],
                        [0.0000, 0.0000, 0.0000, 1.0000],
                    ],
                    [
                        [1.0000, 0.0080, 0.0000, -0.0014],
                        [-0.0074, 0.9258, 0.3780, -0.1982],
                        [0.0030, -0.3779, 0.9258, -0.0158],
                        [0.0000, 0.0000, 0.0000, 1.0000],
                    ],
                    [
                        [0.9999, 0.0104, 0.0000, -0.0019],
                        [-0.0074, 0.7107, 0.7035, 0.8109],
                        [0.0073, -0.7035, 0.7107, 0.0613],
                        [0.0000, 0.0000, 0.0000, 1.0000],
                    ],
                ]
            ),
        ),
        (
            "kinematic_model_ideal_1",
            torch.tensor(
                [
                    [0.0, -10.0, 0.0, 1.0],
                    [0.0, -10.0, 0.0, 1.0],
                    [0.0, -10.0, 0.0, 1.0],
                    [0.0, -10.0, 0.0, 1.0],
                    [0.0, -10.0, 0.0, 1.0],
                    [0.0, -9.0, 0.0, 1.0],
                    [0.0, -10.0, 0.0, 1.0],
                    [0.0, -10.0, 0.0, 1.0],
                    [0.0, -10.0, 0.0, 1.0],
                    [0.0, -10.0, 0.0, 1.0],
                ]
            ),
            torch.tensor(
                [
                    [0.0, 0.0, -1.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0000, 0.7071, -0.7071, 0.0000],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ]
            ),
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
                    ],
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
                    ],
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
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
                    ],
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
                    ],
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
                    ],
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
                    ],
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
                    ],
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
                    ],
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
                    ],
                ]
            ),
        ),
    ],
)
def test_incident_ray_direction_to_orientation(
    request: pytest.FixtureRequest,
    kinematic_model_fixture: str,
    aim_points: torch.Tensor,
    incident_ray_directions: torch.Tensor,
    expected: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test that the incident ray direction to orientation method works as desired.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The pytest fixture used to consider different test cases.
    kinematic_model_fixture : str
        The kinematic model fixture used to select the kinematic model used in the test case.
    aim_points : torch.Tensor
        The aim points for the heliostats.
    incident_ray_directions : torch.Tensor
        The incident ray direction considered.
    expected : torch.Tensor
        The expected orientation matrix.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    kinematic = request.getfixturevalue(kinematic_model_fixture)

    active_heliostats_mask = torch.ones(
        kinematic.number_of_heliostats, dtype=torch.int32, device=device
    )

    kinematic.number_of_active_heliostats = active_heliostats_mask.sum().item()
    kinematic.active_heliostat_positions = (
        kinematic.heliostat_positions.repeat_interleave(active_heliostats_mask, dim=0)
    )
    kinematic.active_initial_orientations = (
        kinematic.initial_orientations.repeat_interleave(active_heliostats_mask, dim=0)
    )
    kinematic.active_translation_deviation_parameters = (
        kinematic.translation_deviation_parameters.repeat_interleave(
            active_heliostats_mask, dim=0
        )
    )
    kinematic.active_rotation_deviation_parameters = (
        kinematic.rotation_deviation_parameters.repeat_interleave(
            active_heliostats_mask, dim=0
        )
    )
    kinematic.actuators.active_geometry_parameters = (
        kinematic.actuators.geometry_parameters.repeat_interleave(
            active_heliostats_mask, dim=0
        )
    )
    kinematic.actuators.active_initial_parameters = (
        kinematic.actuators.initial_parameters.repeat_interleave(
            active_heliostats_mask, dim=0
        )
    )

    orientation_matrix = kinematic.incident_ray_directions_to_orientations(
        incident_ray_directions=incident_ray_directions.to(device),
        aim_points=aim_points.to(device),
        device=device,
    )
    torch.testing.assert_close(
        orientation_matrix, expected.to(device), atol=5e-4, rtol=5e-4
    )


@pytest.mark.parametrize(
    "kinematic_model_fixture, motor_positions, expected",
    [
        (
            "kinematic_model_linear",
            torch.tensor(
                [
                    [0.0, 0.0],
                    [5.0, 1000.0],
                    [10.0, 40.0],
                    [0.0, 0.0],
                    [5.0, 1000.0],
                    [10.0, 40.0],
                ]
            ),
            torch.tensor(
                [
                    [
                        [5.7358e-01, -8.1915e-01, 0.0000e00, 1.4544e-01],
                        [2.5715e-07, 1.8006e-07, 1.0000e00, -8.9500e-02],
                        [-8.1915e-01, -5.7358e-01, 3.1392e-07, 1.0184e-01],
                        [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                    ],
                    [
                        [5.9221e-01, -8.0578e-01, 0.0000e00, 1.4307e-01],
                        [8.2862e-05, 6.0899e-05, 1.0000e00, -8.9511e-02],
                        [-8.0578e-01, -5.9221e-01, 1.0283e-04, 1.0514e-01],
                        [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                    ],
                    [
                        [5.7434e-01, -8.1862e-01, 0.0000e00, 1.4535e-01],
                        [1.6830e-04, 1.1808e-04, 1.0000e00, -8.9521e-02],
                        [-8.1862e-01, -5.7434e-01, 2.0559e-04, 1.0196e-01],
                        [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                    ],
                    [
                        [5.7358e-01, -8.1915e-01, 0.0000e00, 1.4544e-01],
                        [2.5715e-07, 1.8006e-07, 1.0000e00, -8.9500e-02],
                        [-8.1915e-01, -5.7358e-01, 3.1392e-07, 1.0184e-01],
                        [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                    ],
                    [
                        [5.9221e-01, -8.0578e-01, 0.0000e00, 1.4307e-01],
                        [8.2862e-05, 6.0899e-05, 1.0000e00, -8.9511e-02],
                        [-8.0578e-01, -5.9221e-01, 1.0283e-04, 1.0514e-01],
                        [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                    ],
                    [
                        [5.7434e-01, -8.1862e-01, 0.0000e00, 1.4535e-01],
                        [1.6830e-04, 1.1808e-04, 1.0000e00, 9.1048e-01],
                        [-8.1862e-01, -5.7434e-01, 2.0559e-04, 1.0196e-01],
                        [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                    ],
                ],
            ),
        ),
        (
            "kinematic_model_ideal_1",
            torch.tensor([[26651, 15875]]),
            torch.tensor(
                [
                    [
                        [-8.6163e-01, 5.0753e-01, 0.0000e00, 0.0000e00],
                        [3.2746e-01, 5.5592e-01, 7.6402e-01, 0.0000e00],
                        [3.8777e-01, 6.5830e-01, -6.4519e-01, 0.0000e00],
                        [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                    ],
                    [
                        [-8.6163e-01, 5.0753e-01, 0.0000e00, 0.0000e00],
                        [3.2746e-01, 5.5592e-01, 7.6402e-01, 0.0000e00],
                        [3.8777e-01, 6.5830e-01, -6.4519e-01, 0.0000e00],
                        [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                    ],
                    [
                        [-8.6163e-01, 5.0753e-01, 0.0000e00, 0.0000e00],
                        [3.2746e-01, 5.5592e-01, 7.6402e-01, 0.0000e00],
                        [3.8777e-01, 6.5830e-01, -6.4519e-01, 0.0000e00],
                        [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                    ],
                    [
                        [-8.6163e-01, 5.0753e-01, 0.0000e00, 0.0000e00],
                        [3.2746e-01, 5.5592e-01, 7.6402e-01, 0.0000e00],
                        [3.8777e-01, 6.5830e-01, -6.4519e-01, 0.0000e00],
                        [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                    ],
                    [
                        [-8.6163e-01, 5.0753e-01, 0.0000e00, 0.0000e00],
                        [3.2746e-01, 5.5592e-01, 7.6402e-01, 0.0000e00],
                        [3.8777e-01, 6.5830e-01, -6.4519e-01, 0.0000e00],
                        [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                    ],
                    [
                        [-8.6163e-01, 5.0753e-01, 0.0000e00, 0.0000e00],
                        [3.2746e-01, 5.5592e-01, 7.6402e-01, 1.0000e00],
                        [3.8777e-01, 6.5830e-01, -6.4519e-01, 0.0000e00],
                        [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                    ],
                    [
                        [-8.6163e-01, -2.2185e-08, -5.0753e-01, 0.0000e00],
                        [3.2746e-01, 7.6402e-01, -5.5592e-01, 0.0000e00],
                        [3.8777e-01, -6.4519e-01, -6.5830e-01, 0.0000e00],
                        [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                    ],
                    [
                        [-8.6163e-01, -2.2185e-08, -5.0753e-01, 0.0000e00],
                        [3.2746e-01, 7.6402e-01, -5.5592e-01, 0.0000e00],
                        [3.8777e-01, -6.4519e-01, -6.5830e-01, 0.0000e00],
                        [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                    ],
                    [
                        [-8.6163e-01, -2.2185e-08, -5.0753e-01, 0.0000e00],
                        [3.2746e-01, 7.6402e-01, -5.5592e-01, 0.0000e00],
                        [3.8777e-01, -6.4519e-01, -6.5830e-01, 0.0000e00],
                        [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                    ],
                    [
                        [-8.6163e-01, -2.2185e-08, -5.0753e-01, 0.0000e00],
                        [3.2746e-01, 7.6402e-01, -5.5592e-01, 0.0000e00],
                        [3.8777e-01, -6.4519e-01, -6.5830e-01, 0.0000e00],
                        [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                    ],
                ]
            ),
        ),
        (
            "kinematic_model_ideal_2",
            torch.tensor([[0.0, 0.0], [5.0, 1000.0], [10.0, 40.0]]),
            torch.tensor(
                [
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
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
                    ],
                    [
                        [-6.6694e-01, 3.2570e-08, 7.4511e-01, 0.0000e00],
                        [-6.2520e-01, 5.4402e-01, -5.5961e-01, 0.0000e00],
                        [-4.0536e-01, -8.3907e-01, -3.6283e-01, 0.0000e00],
                        [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                    ],
                ]
            ),
        ),
    ],
)
def test_motor_positions_to_orientations(
    request: pytest.FixtureRequest,
    kinematic_model_fixture: str,
    motor_positions: torch.Tensor,
    expected: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test that the motor positions to orientation method works as desired.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The pytest fixture used to consider different test cases.
    kinematic_model_fixture : str
        The kinematic model fixture used to select the kinematic model used in the test case.
    motor_positions : torch.Tensor
        The motor positions.
    expected : torch.Tensor
        The expected orientation matrix.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    kinematic = request.getfixturevalue(kinematic_model_fixture)

    active_heliostats_mask = torch.ones(
        kinematic.number_of_heliostats, dtype=torch.int32, device=device
    )

    kinematic.number_of_active_heliostats = active_heliostats_mask.sum().item()
    kinematic.active_heliostat_positions = (
        kinematic.heliostat_positions.repeat_interleave(active_heliostats_mask, dim=0)
    )
    kinematic.active_initial_orientations = (
        kinematic.initial_orientations.repeat_interleave(active_heliostats_mask, dim=0)
    )
    kinematic.active_translation_deviation_parameters = (
        kinematic.translation_deviation_parameters.repeat_interleave(
            active_heliostats_mask, dim=0
        )
    )
    kinematic.active_rotation_deviation_parameters = (
        kinematic.rotation_deviation_parameters.repeat_interleave(
            active_heliostats_mask, dim=0
        )
    )
    kinematic.actuators.active_geometry_parameters = (
        kinematic.actuators.geometry_parameters.repeat_interleave(
            active_heliostats_mask, dim=0
        )
    )
    kinematic.actuators.active_initial_parameters = (
        kinematic.actuators.initial_parameters.repeat_interleave(
            active_heliostats_mask, dim=0
        )
    )

    orientation_matrix = kinematic.motor_positions_to_orientations(
        motor_positions=motor_positions.to(device),
        device=device,
    )
    torch.testing.assert_close(
        orientation_matrix, expected.to(device), atol=5e-4, rtol=5e-4
    )
