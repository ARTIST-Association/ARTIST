import pytest
import torch

from artist.field.kinematics_rigid_body import (
    RigidBody,
)


@pytest.fixture
def kinematics_parameters(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Define kinematics parameters used in tests.

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
def kinematics_model_linear(
    kinematics_parameters: torch.Tensor,
    device: torch.device,
) -> RigidBody:
    """
    Create a rigid body kinematics with linear actuators and deviation parameters.

    Parameters
    ----------
    kinematics_parameters : torch.Tensor
        The kinematics deviation parameters.
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    RigidBody
        The kinematics model.
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

    actuator_parameters_non_optimizable = torch.zeros((6, 7, 2), device=device)
    actuator_parameters_non_optimizable[:, 1, 1] = 1
    actuator_parameters_non_optimizable[:, 4, 0] = 154166.666
    actuator_parameters_non_optimizable[:, 4, 1] = 154166.666
    actuator_parameters_non_optimizable[:, 5, 0] = 0.34061
    actuator_parameters_non_optimizable[:, 5, 1] = 0.3479
    actuator_parameters_non_optimizable[:, 6, 0] = 0.3204
    actuator_parameters_non_optimizable[:, 6, 1] = 0.309

    actuator_parameters_optimizable = torch.zeros((6, 2, 2), device=device)
    actuator_parameters_optimizable[:, 0, 0] = -1.570796
    actuator_parameters_optimizable[:, 0, 1] = 0.959931
    actuator_parameters_optimizable[:, 1, :] = 0.075

    return RigidBody(
        number_of_heliostats=6,
        heliostat_positions=heliostat_positions,
        initial_orientations=initial_orientation.expand(6, 4),
        translation_deviation_parameters=kinematics_parameters[0],
        rotation_deviation_parameters=kinematics_parameters[1],
        actuator_parameters_non_optimizable=actuator_parameters_non_optimizable,
        actuator_parameters_optimizable=actuator_parameters_optimizable,
        device=device,
    )


@pytest.fixture
def kinematics_model_ideal_1(
    device: torch.device,
) -> RigidBody:
    """
    Create a rigid body kinematics with ideal actuators and no deviation parameters.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    RigidBody
        The kinematics model.
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

    actuator_parameters_non_optimizable = torch.zeros((10, 4, 2), device=device)
    actuator_parameters_non_optimizable[:, 0, :] = 1
    actuator_parameters_non_optimizable[:, 1, 1] = 1

    return RigidBody(
        number_of_heliostats=10,
        heliostat_positions=positions,
        initial_orientations=initial_orientations,
        translation_deviation_parameters=torch.zeros(
            (10, 9), dtype=torch.float, device=device
        ),
        rotation_deviation_parameters=torch.zeros(
            (10, 4), dtype=torch.float, device=device
        ),
        actuator_parameters_non_optimizable=actuator_parameters_non_optimizable,
        device=device,
    )


@pytest.fixture
def kinematics_model_ideal_2(
    device: torch.device,
) -> RigidBody:
    """
    Create rigid body kinematics with ideal actuators and deviation parameters.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    RigidBody
        The kinematics model.
    """
    positions = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        device=device,
    )
    initial_orientations = torch.tensor(
        [[0.0, -1.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        device=device,
    )

    actuator_parameters_non_optimizable = torch.zeros((3, 4, 2), device=device)
    actuator_parameters_non_optimizable[:, 0, :] = 1
    actuator_parameters_non_optimizable[:, 1, 1] = 1

    return RigidBody(
        number_of_heliostats=3,
        heliostat_positions=positions,
        initial_orientations=initial_orientations,
        translation_deviation_parameters=torch.zeros(
            (3, 9), dtype=torch.float, device=device
        ),
        rotation_deviation_parameters=torch.zeros(
            (3, 4), dtype=torch.float, device=device
        ),
        actuator_parameters_non_optimizable=actuator_parameters_non_optimizable,
        device=device,
    )


@pytest.mark.parametrize(
    "kinematics_model_fixture, aim_points, incident_ray_directions, expected",
    [
        (
            "kinematics_model_linear",
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
                        [
                            9.999456405640e-01,
                            -4.558640964714e-10,
                            -1.042895484716e-02,
                            -1.851661014371e-03,
                        ],
                        [
                            -7.411775179207e-03,
                            7.035031914711e-01,
                            -7.106534838676e-01,
                            -1.891400665045e-01,
                        ],
                        [
                            7.336803711951e-03,
                            7.106921076775e-01,
                            7.034649252892e-01,
                            6.129327416420e-02,
                        ],
                        [
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            1.000000000000e00,
                        ],
                    ],
                    [
                        [
                            7.122755050659e-01,
                            3.068102216730e-08,
                            7.019000053406e-01,
                            1.246223449707e-01,
                        ],
                        [
                            7.018629312515e-01,
                            -1.027521397918e-02,
                            -7.122378945351e-01,
                            -1.255382150412e-01,
                        ],
                        [
                            7.212151307613e-03,
                            9.999471902847e-01,
                            -7.318804971874e-03,
                            -9.079471230507e-02,
                        ],
                        [
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            1.000000000000e00,
                        ],
                    ],
                    [
                        [
                            9.999730587006e-01,
                            -3.207012433393e-10,
                            -7.336789276451e-03,
                            -1.302646938711e-03,
                        ],
                        [
                            -7.336692418903e-03,
                            -5.136868450791e-03,
                            -9.999598860741e-01,
                            -1.770831346512e-01,
                        ],
                        [
                            -3.768780152313e-05,
                            9.999868273735e-01,
                            -5.136730149388e-03,
                            -9.041082859039e-02,
                        ],
                        [
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            1.000000000000e00,
                        ],
                    ],
                    [
                        [
                            7.018998265266e-01,
                            -3.113456159554e-08,
                            -7.122757434845e-01,
                            -1.264645606279e-01,
                        ],
                        [
                            -7.122381329536e-01,
                            -1.027521397918e-02,
                            -7.018627524376e-01,
                            -1.236961036921e-01,
                        ],
                        [
                            -7.318763993680e-03,
                            9.999471902847e-01,
                            -7.212193217129e-03,
                            -9.077578783035e-02,
                        ],
                        [
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            1.000000000000e00,
                        ],
                    ],
                    [
                        [
                            9.999683499336e-01,
                            -3.478643761934e-10,
                            -7.958209142089e-03,
                            -1.412980025634e-03,
                        ],
                        [
                            -7.367914076895e-03,
                            3.779509067535e-01,
                            -9.257963299751e-01,
                            -1.982017606497e-01,
                        ],
                        [
                            3.007812658325e-03,
                            9.258256554604e-01,
                            3.779389560223e-01,
                            -1.575836539268e-02,
                        ],
                        [
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            1.000000000000e00,
                        ],
                    ],
                    [
                        [
                            9.999456405640e-01,
                            -4.558640964714e-10,
                            -1.042895484716e-02,
                            -1.851661014371e-03,
                        ],
                        [
                            -7.411775179207e-03,
                            7.035031914711e-01,
                            -7.106534838676e-01,
                            8.108599185944e-01,
                        ],
                        [
                            7.336803711951e-03,
                            7.106921076775e-01,
                            7.034649252892e-01,
                            6.129327416420e-02,
                        ],
                        [
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            1.000000000000e00,
                        ],
                    ],
                ]
            ),
        ),
        (
            "kinematics_model_ideal_1",
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
                        [
                            1.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                        ],
                        [
                            0.000000000000e00,
                            7.071067690849e-01,
                            -7.071067690849e-01,
                            0.000000000000e00,
                        ],
                        [
                            0.000000000000e00,
                            7.071067690849e-01,
                            7.071067690849e-01,
                            0.000000000000e00,
                        ],
                        [
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            1.000000000000e00,
                        ],
                    ],
                    [
                        [
                            7.071067690849e-01,
                            3.090861966371e-08,
                            7.071068286896e-01,
                            0.000000000000e00,
                        ],
                        [
                            7.071068286896e-01,
                            -3.090861966371e-08,
                            -7.071067690849e-01,
                            0.000000000000e00,
                        ],
                        [
                            0.000000000000e00,
                            1.000000000000e00,
                            -4.371138828674e-08,
                            0.000000000000e00,
                        ],
                        [
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            1.000000000000e00,
                        ],
                    ],
                    [
                        [
                            1.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                        ],
                        [
                            0.000000000000e00,
                            -4.371138828674e-08,
                            -1.000000000000e00,
                            0.000000000000e00,
                        ],
                        [
                            0.000000000000e00,
                            1.000000000000e00,
                            -4.371138828674e-08,
                            0.000000000000e00,
                        ],
                        [
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            1.000000000000e00,
                        ],
                    ],
                    [
                        [
                            7.071067690849e-01,
                            -3.090861966371e-08,
                            -7.071068286896e-01,
                            0.000000000000e00,
                        ],
                        [
                            -7.071068286896e-01,
                            -3.090861966371e-08,
                            -7.071067690849e-01,
                            0.000000000000e00,
                        ],
                        [
                            0.000000000000e00,
                            1.000000000000e00,
                            -4.371138828674e-08,
                            0.000000000000e00,
                        ],
                        [
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            1.000000000000e00,
                        ],
                    ],
                    [
                        [
                            1.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                        ],
                        [
                            0.000000000000e00,
                            3.826815783978e-01,
                            -9.238802790642e-01,
                            0.000000000000e00,
                        ],
                        [
                            0.000000000000e00,
                            9.238802790642e-01,
                            3.826815783978e-01,
                            0.000000000000e00,
                        ],
                        [
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            1.000000000000e00,
                        ],
                    ],
                    [
                        [
                            1.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                        ],
                        [
                            0.000000000000e00,
                            7.071067690849e-01,
                            -7.071067690849e-01,
                            1.000000000000e00,
                        ],
                        [
                            0.000000000000e00,
                            7.071067690849e-01,
                            7.071067690849e-01,
                            0.000000000000e00,
                        ],
                        [
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            1.000000000000e00,
                        ],
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
    kinematics_model_fixture: str,
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
    kinematics_model_fixture : str
        The kinematics model fixture used to select the kinematics model used in the test case.
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
    kinematics = request.getfixturevalue(kinematics_model_fixture)

    active_heliostats_mask = torch.ones(
        kinematics.number_of_heliostats, dtype=torch.int32, device=device
    )

    kinematics.number_of_active_heliostats = active_heliostats_mask.sum().item()
    kinematics.active_heliostat_positions = (
        kinematics.heliostat_positions.repeat_interleave(active_heliostats_mask, dim=0)
    )
    kinematics.active_initial_orientations = (
        kinematics.initial_orientations.repeat_interleave(active_heliostats_mask, dim=0)
    )
    kinematics.active_translation_deviation_parameters = (
        kinematics.translation_deviation_parameters.repeat_interleave(
            active_heliostats_mask, dim=0
        )
    )
    kinematics.active_rotation_deviation_parameters = (
        kinematics.rotation_deviation_parameters.repeat_interleave(
            active_heliostats_mask, dim=0
        )
    )
    kinematics.actuators.active_non_optimizable_parameters = (
        kinematics.actuators.non_optimizable_parameters.repeat_interleave(
            active_heliostats_mask, dim=0
        )
    )
    if kinematics.actuators.active_optimizable_parameters.numel() > 0:
        kinematics.actuators.active_optimizable_parameters = (
            kinematics.actuators.optimizable_parameters.repeat_interleave(
                active_heliostats_mask, dim=0
            )
        )
    else:
        kinematics.actuators.active_optimizable_parameters = torch.tensor(
            [], requires_grad=True
        )

    orientation_matrix = kinematics.incident_ray_directions_to_orientations(
        incident_ray_directions=incident_ray_directions.to(device),
        aim_points=aim_points.to(device),
        device=device,
    )
    torch.testing.assert_close(
        orientation_matrix, expected.to(device), atol=5e-4, rtol=5e-4
    )


@pytest.mark.parametrize(
    "kinematics_model_fixture, motor_positions, expected",
    [
        (
            "kinematics_model_linear",
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
                        [
                            5.735765099525e-01,
                            3.580627350175e-08,
                            8.191520571709e-01,
                            1.454404443502e-01,
                        ],
                        [
                            2.571453308065e-07,
                            1.000000000000e00,
                            -2.237665057692e-07,
                            -8.950003981590e-02,
                        ],
                        [
                            -8.191520571709e-01,
                            3.389882863303e-07,
                            5.735765099525e-01,
                            1.018384844065e-01,
                        ],
                        [
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            1.000000000000e00,
                        ],
                    ],
                    [
                        [
                            5.922092795372e-01,
                            3.522194802486e-08,
                            8.057842254639e-01,
                            1.430669873953e-01,
                        ],
                        [
                            8.305405208375e-05,
                            1.000000000000e00,
                            -6.108410161687e-05,
                            -8.951085805893e-02,
                        ],
                        [
                            -8.057842254639e-01,
                            1.030982093653e-04,
                            5.922092795372e-01,
                            1.051375344396e-01,
                        ],
                        [
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            1.000000000000e00,
                        ],
                    ],
                    [
                        [
                            5.743377208710e-01,
                            3.578294993645e-08,
                            8.186184763908e-01,
                            1.453457176685e-01,
                        ],
                        [
                            1.681064895820e-04,
                            1.000000000000e00,
                            -1.179862010758e-04,
                            -8.952096104622e-02,
                        ],
                        [
                            -8.186184763908e-01,
                            2.053789939964e-04,
                            5.743377208710e-01,
                            1.019552871585e-01,
                        ],
                        [
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            1.000000000000e00,
                        ],
                    ],
                    [
                        [
                            5.735765099525e-01,
                            3.580627350175e-08,
                            8.191520571709e-01,
                            1.454404443502e-01,
                        ],
                        [
                            2.571453308065e-07,
                            1.000000000000e00,
                            -2.237665057692e-07,
                            -8.950003981590e-02,
                        ],
                        [
                            -8.191520571709e-01,
                            3.389882863303e-07,
                            5.735765099525e-01,
                            1.018384844065e-01,
                        ],
                        [
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            1.000000000000e00,
                        ],
                    ],
                    [
                        [
                            5.922092795372e-01,
                            3.522194802486e-08,
                            8.057842254639e-01,
                            1.430669873953e-01,
                        ],
                        [
                            8.305405208375e-05,
                            1.000000000000e00,
                            -6.108410161687e-05,
                            -8.951085805893e-02,
                        ],
                        [
                            -8.057842254639e-01,
                            1.030982093653e-04,
                            5.922092795372e-01,
                            1.051375344396e-01,
                        ],
                        [
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            1.000000000000e00,
                        ],
                    ],
                    [
                        [
                            5.743377208710e-01,
                            3.578294993645e-08,
                            8.186184763908e-01,
                            1.453457176685e-01,
                        ],
                        [
                            1.681064895820e-04,
                            1.000000000000e00,
                            -1.179862010758e-04,
                            9.104790687561e-01,
                        ],
                        [
                            -8.186184763908e-01,
                            2.053789939964e-04,
                            5.743377208710e-01,
                            1.019552871585e-01,
                        ],
                        [
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            1.000000000000e00,
                        ],
                    ],
                ]
            ),
        ),
        (
            "kinematics_model_ideal_1",
            torch.tensor([[26651, 15875]]),
            torch.tensor(
                [
                    [
                        [
                            -8.616312146187e-01,
                            -2.218505557039e-08,
                            -5.075349211693e-01,
                            0.000000000000e00,
                        ],
                        [
                            3.274583220482e-01,
                            7.640190720558e-01,
                            -5.559191107750e-01,
                            0.000000000000e00,
                        ],
                        [
                            3.877663612366e-01,
                            -6.451936960220e-01,
                            -6.583026647568e-01,
                            0.000000000000e00,
                        ],
                        [
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            1.000000000000e00,
                        ],
                    ],
                    [
                        [
                            -8.616312146187e-01,
                            -2.218505557039e-08,
                            -5.075349211693e-01,
                            0.000000000000e00,
                        ],
                        [
                            3.274583220482e-01,
                            7.640190720558e-01,
                            -5.559191107750e-01,
                            0.000000000000e00,
                        ],
                        [
                            3.877663612366e-01,
                            -6.451936960220e-01,
                            -6.583026647568e-01,
                            0.000000000000e00,
                        ],
                        [
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            1.000000000000e00,
                        ],
                    ],
                    [
                        [
                            -8.616312146187e-01,
                            -2.218505557039e-08,
                            -5.075349211693e-01,
                            0.000000000000e00,
                        ],
                        [
                            3.274583220482e-01,
                            7.640190720558e-01,
                            -5.559191107750e-01,
                            0.000000000000e00,
                        ],
                        [
                            3.877663612366e-01,
                            -6.451936960220e-01,
                            -6.583026647568e-01,
                            0.000000000000e00,
                        ],
                        [
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            1.000000000000e00,
                        ],
                    ],
                    [
                        [
                            -8.616312146187e-01,
                            -2.218505557039e-08,
                            -5.075349211693e-01,
                            0.000000000000e00,
                        ],
                        [
                            3.274583220482e-01,
                            7.640190720558e-01,
                            -5.559191107750e-01,
                            0.000000000000e00,
                        ],
                        [
                            3.877663612366e-01,
                            -6.451936960220e-01,
                            -6.583026647568e-01,
                            0.000000000000e00,
                        ],
                        [
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            1.000000000000e00,
                        ],
                    ],
                    [
                        [
                            -8.616312146187e-01,
                            -2.218505557039e-08,
                            -5.075349211693e-01,
                            0.000000000000e00,
                        ],
                        [
                            3.274583220482e-01,
                            7.640190720558e-01,
                            -5.559191107750e-01,
                            0.000000000000e00,
                        ],
                        [
                            3.877663612366e-01,
                            -6.451936960220e-01,
                            -6.583026647568e-01,
                            0.000000000000e00,
                        ],
                        [
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            1.000000000000e00,
                        ],
                    ],
                    [
                        [
                            -8.616312146187e-01,
                            -2.218505557039e-08,
                            -5.075349211693e-01,
                            0.000000000000e00,
                        ],
                        [
                            3.274583220482e-01,
                            7.640190720558e-01,
                            -5.559191107750e-01,
                            1.000000000000e00,
                        ],
                        [
                            3.877663612366e-01,
                            -6.451936960220e-01,
                            -6.583026647568e-01,
                            0.000000000000e00,
                        ],
                        [
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            1.000000000000e00,
                        ],
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
            "kinematics_model_ideal_2",
            torch.tensor([[0.0, 0.0], [5.0, 1000.0], [10.0, 40.0]]),
            torch.tensor(
                [
                    [
                        [
                            1.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                        ],
                        [
                            0.000000000000e00,
                            -4.371138828674e-08,
                            -1.000000000000e00,
                            0.000000000000e00,
                        ],
                        [
                            0.000000000000e00,
                            1.000000000000e00,
                            -4.371138828674e-08,
                            0.000000000000e00,
                        ],
                        [
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            1.000000000000e00,
                        ],
                    ],
                    [
                        [
                            5.623790621758e-01,
                            3.614405486019e-08,
                            8.268795609474e-01,
                            0.000000000000e00,
                        ],
                        [
                            2.345544546843e-01,
                            9.589242935181e-01,
                            -1.595257073641e-01,
                            1.000000000000e00,
                        ],
                        [
                            -7.929149270058e-01,
                            2.836621999741e-01,
                            5.392789244652e-01,
                            0.000000000000e00,
                        ],
                        [
                            0.000000000000e00,
                            0.000000000000e00,
                            0.000000000000e00,
                            1.000000000000e00,
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
    kinematics_model_fixture: str,
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
    kinematics_model_fixture : str
        The kinematics model fixture used to select the kinematics model used in the test case.
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
    kinematics = request.getfixturevalue(kinematics_model_fixture)

    active_heliostats_mask = torch.ones(
        kinematics.number_of_heliostats, dtype=torch.int32, device=device
    )

    kinematics.number_of_active_heliostats = active_heliostats_mask.sum().item()
    kinematics.active_heliostat_positions = (
        kinematics.heliostat_positions.repeat_interleave(active_heliostats_mask, dim=0)
    )
    kinematics.active_initial_orientations = (
        kinematics.initial_orientations.repeat_interleave(active_heliostats_mask, dim=0)
    )
    kinematics.active_translation_deviation_parameters = (
        kinematics.translation_deviation_parameters.repeat_interleave(
            active_heliostats_mask, dim=0
        )
    )
    kinematics.active_rotation_deviation_parameters = (
        kinematics.rotation_deviation_parameters.repeat_interleave(
            active_heliostats_mask, dim=0
        )
    )
    kinematics.actuators.active_non_optimizable_parameters = (
        kinematics.actuators.non_optimizable_parameters.repeat_interleave(
            active_heliostats_mask, dim=0
        )
    )
    if kinematics.actuators.active_optimizable_parameters.numel() > 0:
        kinematics.actuators.active_optimizable_parameters = (
            kinematics.actuators.optimizable_parameters.repeat_interleave(
                active_heliostats_mask, dim=0
            )
        )
    else:
        kinematics.actuators.active_optimizable_parameters = torch.tensor(
            [], requires_grad=True
        )

    orientation_matrix = kinematics.motor_positions_to_orientations(
        motor_positions=motor_positions.to(device),
        device=device,
    )
    torch.testing.assert_close(
        orientation_matrix, expected.to(device), atol=5e-4, rtol=5e-4
    )
