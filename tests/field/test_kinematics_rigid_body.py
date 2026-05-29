import pytest
import torch

from artist.field import RigidBody


def _activate_all_heliostats_for_test(
    kinematics: RigidBody, device: torch.device
) -> None:
    active_heliostats_mask = torch.ones(
        kinematics.number_of_heliostats, dtype=torch.int32, device=device
    )
    kinematics.number_of_active_heliostats = int(active_heliostats_mask.sum().item())
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
    if kinematics.actuators.optimizable_parameters.numel() > 0:
        kinematics.actuators.active_optimizable_parameters = (
            kinematics.actuators.optimizable_parameters.repeat_interleave(
                active_heliostats_mask, dim=0
            )
        )
    else:
        kinematics.actuators.active_optimizable_parameters = torch.tensor(
            [], requires_grad=True, device=device
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

    translation_deviation_parameters[:, 7] = 0.175

    rotation_deviation_parameters = torch.zeros((6, 4), device=device)

    return translation_deviation_parameters, rotation_deviation_parameters


@pytest.fixture
def kinematics_model_linear(
    kinematics_parameters: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> RigidBody:
    """
    Create a rigid body kinematics with linear actuators and deviation parameters.

    Parameters
    ----------
    kinematics_parameters : tuple[torch.Tensor, torch.Tensor]
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
    actuator_parameters_non_optimizable[:, 3, 0] = 68745
    actuator_parameters_non_optimizable[:, 3, 1] = 75308
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
    actuator_parameters_non_optimizable[:, 2, :] = -torch.pi
    actuator_parameters_non_optimizable[:, 3, :] = torch.pi

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
            torch.tensor([[[ 1.000000000000e+00,  6.774044351713e-14,  1.549720764160e-06,
          -2.712011166750e-07],
         [ 1.089058969228e-06,  7.114415168762e-01, -7.027453184128e-01,
           1.229804158211e-01],
         [-1.102535748032e-06,  7.027453184128e-01,  7.114415168762e-01,
          -1.245022714138e-01],
         [ 0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,
           1.000000000000e+00]],

        [[ 7.027451992035e-01,  3.109810009505e-08,  7.114416360855e-01,
          -1.245022863150e-01],
         [ 7.114416360855e-01,  1.399793518431e-06, -7.027451992035e-01,
           1.229804083705e-01],
         [-1.017725480779e-06,  1.000000000000e+00,  9.615737326385e-07,
          -1.759248817734e-07],
         [ 0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,
           1.000000000000e+00]],

        [[ 1.000000000000e+00,  2.605401856173e-14,  5.960464477539e-07,
          -1.043081283569e-07],
         [ 5.960464477539e-07,  5.523350523617e-07, -1.000000000000e+00,
           1.749999970198e-01],
         [-3.552713678801e-13,  1.000000000000e+00,  5.523350523617e-07,
          -1.043081283569e-07],
         [ 0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,
           1.000000000000e+00]],

        [[ 7.027456760406e-01, -3.109808233148e-08, -7.114411592484e-01,
           1.245022043586e-01],
         [-7.114411592484e-01,  1.757421387083e-06, -7.027456760406e-01,
           1.229804903269e-01],
         [ 1.272155941479e-06,  1.000000000000e+00,  1.212895881508e-06,
          -2.199062549835e-07],
         [ 0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,
           1.000000000000e+00]],

        [[ 1.000000000000e+00,  3.126481956358e-14,  7.152557373047e-07,
          -1.251697483440e-07],
         [ 6.598978075090e-07,  3.857483565807e-01, -9.226040244102e-01,
           1.614557057619e-01],
         [-2.759087465165e-07,  9.226040244102e-01,  3.857483565807e-01,
          -6.750596314669e-02],
         [ 0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,
           1.000000000000e+00]],

        [[ 1.000000000000e+00,  6.774044351713e-14,  1.549720764160e-06,
          -2.712011166750e-07],
         [ 1.089058969228e-06,  7.114415168762e-01, -7.027453184128e-01,
           1.122980356216e+00],
         [-1.102535748032e-06,  7.027453184128e-01,  7.114415168762e-01,
          -1.245022714138e-01],
         [ 0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,
           1.000000000000e+00]]])
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
    _activate_all_heliostats_for_test(kinematics, device)

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
            torch.tensor([[[ 5.735765099525e-01,  3.580627350175e-08,  8.191520571709e-01,
          -1.433516144753e-01],
         [ 2.571453308065e-07,  1.000000000000e+00, -2.237665057692e-07,
           3.150964289489e-08],
         [-8.191520571709e-01,  3.389882863303e-07,  5.735765099525e-01,
          -1.003758907318e-01],
         [ 0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,
           1.000000000000e+00]],

        [[ 5.922092795372e-01,  3.522194802486e-08,  8.057842254639e-01,
          -1.410122364759e-01],
         [ 8.305405208375e-05,  1.000000000000e+00, -6.108410161687e-05,
           1.068206802302e-05],
         [-8.057842254639e-01,  1.030982093653e-04,  5.922092795372e-01,
          -1.036366224289e-01],
         [ 0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,
           1.000000000000e+00]],

        [[ 5.743377208710e-01,  3.578294993645e-08,  8.186184763908e-01,
          -1.432582288980e-01],
         [ 1.681064895820e-04,  1.000000000000e+00, -1.179862010758e-04,
           2.063993451884e-05],
         [-8.186184763908e-01,  2.053789939964e-04,  5.743377208710e-01,
          -1.005090996623e-01],
         [ 0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,
           1.000000000000e+00]],

        [[ 5.735765099525e-01,  3.580627350175e-08,  8.191520571709e-01,
          -1.433516144753e-01],
         [ 2.571453308065e-07,  1.000000000000e+00, -2.237665057692e-07,
           3.150964289489e-08],
         [-8.191520571709e-01,  3.389882863303e-07,  5.735765099525e-01,
          -1.003758907318e-01],
         [ 0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,
           1.000000000000e+00]],

        [[ 5.922092795372e-01,  3.522194802486e-08,  8.057842254639e-01,
          -1.410122364759e-01],
         [ 8.305405208375e-05,  1.000000000000e+00, -6.108410161687e-05,
           1.068206802302e-05],
         [-8.057842254639e-01,  1.030982093653e-04,  5.922092795372e-01,
          -1.036366224289e-01],
         [ 0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,
           1.000000000000e+00]],

        [[ 5.743377208710e-01,  3.578294993645e-08,  8.186184763908e-01,
          -1.432582288980e-01],
         [ 1.681064895820e-04,  1.000000000000e+00, -1.179862010758e-04,
           1.000020623207e+00],
         [-8.186184763908e-01,  2.053789939964e-04,  5.743377208710e-01,
          -1.005090996623e-01],
         [ 0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,
           1.000000000000e+00]]])
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
    _activate_all_heliostats_for_test(kinematics, device)

    orientation_matrix = kinematics.motor_positions_to_orientations(
        motor_positions=motor_positions.to(device),
        device=device,
    )
    torch.testing.assert_close(
        orientation_matrix, expected.to(device), atol=5e-4, rtol=5e-4
    )
