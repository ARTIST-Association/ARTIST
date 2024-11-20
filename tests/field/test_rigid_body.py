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


@pytest.fixture(params=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def device(request: pytest.FixtureRequest) -> torch.device:
    """
    Return the device on which to initialize tensors.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The pytest fixture used to consider different test cases.

    Returns
    -------
    torch.device
        The device on which to initialize tensors.
    """
    return torch.device(request.param)


@pytest.fixture
def actuator_configuration() -> ActuatorListConfig:
    """
    Define actuator parameters used in tests.

    Returns
    -------
    ActuatorListConfig
        A List containing parameters for each actuator.
    """
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
def initial_offsets_south(device: torch.device) -> KinematicOffsets:
    """
    Define initial offsets for a south-orientated heliostat.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    KinematicOffsets
        Initial kinematic offsets for a south-orientated heliostat.
    """
    initial_offsets = KinematicOffsets(
        kinematic_initial_orientation_offset_e=torch.tensor(0.0, device=device),
        kinematic_initial_orientation_offset_n=torch.tensor(0.0, device=device),
        kinematic_initial_orientation_offset_u=torch.tensor(0.0, device=device),
    )
    return initial_offsets


@pytest.fixture
def initial_offsets_above(device: torch.device) -> KinematicOffsets:
    """
    Define initial offsets for an up-orientated heliostat.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    KinematicOffsets
        Initial kinematic offsets for an up-orientated heliostat.
    """
    initial_offsets = KinematicOffsets(
        kinematic_initial_orientation_offset_e=torch.tensor(
            torch.pi / 2, device=device
        ),
        kinematic_initial_orientation_offset_n=torch.tensor(0.0, device=device),
        kinematic_initial_orientation_offset_u=torch.tensor(0.0, device=device),
    )
    return initial_offsets


@pytest.fixture
def kinematic_model_1(
    actuator_configuration: ActuatorListConfig,
    initial_offsets_south: KinematicOffsets,
    device: torch.device,
) -> RigidBody:
    """
    Create a kinematic model to use in the test.

    Parameters
    ----------
    actuator_configuration : ActuatorListConfig
        The configuration of the actuators.
    initial_offsets_south : KinematicOffsets
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
        actuator_config=actuator_configuration,
        initial_orientation_offsets=initial_offsets_south,
        device=device,
    )


@pytest.fixture
def kinematic_model_2(
    actuator_configuration: ActuatorListConfig,
    initial_offsets_south: KinematicOffsets,
    device: torch.device,
) -> RigidBody:
    """
    Create a kinematic model to use in the test.

    Parameters
    ----------
    actuator_configuration : ActuatorListConfig
        The configuration of the actuators.
    initial_offsets_south : KinematicOffsets
        The kinematic initial orientation offsets.
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
        actuator_config=actuator_configuration,
        initial_orientation_offsets=initial_offsets_south,
        device=device,
    )


@pytest.fixture
def kinematic_model_3(
    actuator_configuration: ActuatorListConfig,
    initial_offsets_above: KinematicOffsets,
    device: torch.device,
) -> RigidBody:
    """
    Create a kinematic model to use in the test.

    Parameters
    ----------
    actuator_configuration : ActuatorListConfig
        The configuration of the actuators.
    initial_offsets_above : KinematicOffsets
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
        actuator_config=actuator_configuration,
        initial_orientation_offsets=initial_offsets_above,
        device=device,
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
    ],
)
def test_orientation_matrix(
    request: pytest.FixtureRequest,
    kinematic_model_fixture: str,
    incident_ray_direction: torch.Tensor,
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
    incident_ray_direction : torch.Tensor
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
    orientation_matrix = request.getfixturevalue(
        kinematic_model_fixture
    ).incident_ray_direction_to_orientation(
        incident_ray_direction.to(device), device=device
    )
    torch.testing.assert_close(orientation_matrix, expected.to(device))
