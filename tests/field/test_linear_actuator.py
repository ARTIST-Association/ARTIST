import pytest
import torch

from artist.field.kinematic_rigid_body import (
    RigidBody,
)
from artist.util import config_dictionary
from artist.util.configuration_classes import (
    ActuatorConfig,
    ActuatorListConfig,
    ActuatorParameters,
    KinematicDeviations,
)


@pytest.fixture
def deviation_parameters(device: torch.device) -> KinematicDeviations:
    """
    Define deviation parameters used in tests.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    KinematicDeviations
        The deviation parameters.
    """
    deviation_parameters = KinematicDeviations(
        first_joint_translation_e=torch.tensor(0.0, device=device),
        first_joint_translation_n=torch.tensor(0.0, device=device),
        first_joint_translation_u=torch.tensor(0.0, device=device),
        first_joint_tilt_e=torch.tensor(0.0, device=device),
        first_joint_tilt_n=torch.tensor(0.0, device=device),
        first_joint_tilt_u=torch.tensor(0.0, device=device),
        second_joint_translation_e=torch.tensor(0.0, device=device),
        second_joint_translation_n=torch.tensor(0.0, device=device),
        second_joint_translation_u=torch.tensor(0.315, device=device),
        second_joint_tilt_e=torch.tensor(0.0, device=device),
        second_joint_tilt_n=torch.tensor(0.0, device=device),
        second_joint_tilt_u=torch.tensor(0.0, device=device),
        concentrator_translation_e=torch.tensor(0.0, device=device),
        concentrator_translation_n=torch.tensor(-0.17755, device=device),
        concentrator_translation_u=torch.tensor(-0.4045, device=device),
        concentrator_tilt_e=torch.tensor(0.0, device=device),
        concentrator_tilt_n=torch.tensor(0.0, device=device),
        concentrator_tilt_u=torch.tensor(0.0, device=device),
    )
    return deviation_parameters


@pytest.fixture
def actuator_configuration(device: torch.device) -> ActuatorListConfig:
    """
    Define actuator parameters used in tests as measured experimentally.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    ActuatorListConfig
        A List containing parameters for each actuator.
    """
    actuator1_parameters = ActuatorParameters(
        increment=torch.tensor(154166.666, device=device),
        initial_stroke_length=torch.tensor(0.075, device=device),
        offset=torch.tensor(0.34061, device=device),
        pivot_radius=torch.tensor(0.3204, device=device),
        initial_angle=torch.tensor(-1.570796, device=device),
    )

    actuator2_parameters = ActuatorParameters(
        increment=torch.tensor(154166.666, device=device),
        initial_stroke_length=torch.tensor(0.075, device=device),
        offset=torch.tensor(0.3479, device=device),
        pivot_radius=torch.tensor(0.309, device=device),
        initial_angle=torch.tensor(0.959931, device=device),
    )
    actuator1_config = ActuatorConfig(
        key="",
        type=config_dictionary.linear_actuator_key,
        clockwise_axis_movement=False,
        parameters=actuator1_parameters,
    )
    actuator2_config = ActuatorConfig(
        key="",
        type=config_dictionary.linear_actuator_key,
        clockwise_axis_movement=True,
        parameters=actuator2_parameters,
    )

    return ActuatorListConfig(actuator_list=[actuator1_config, actuator2_config])


@pytest.fixture
def kinematic_model_1(
    actuator_configuration: ActuatorListConfig,
    deviation_parameters: KinematicDeviations,
    device: torch.device,
) -> RigidBody:
    """
    Create a kinematic model to use in the test.

    Parameters
    ----------
    actuator_configuration : ActuatorListConfig
        The configuration of the actuators.
    deviation_parameters : KinematicDeviations
        The kinematic deviations.
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
        initial_orientation=torch.tensor([0.0, -1.0, 0.0, 0.0], device=device),
        deviation_parameters=deviation_parameters,
        device=device,
    )


@pytest.fixture
def kinematic_model_2(
    actuator_configuration: ActuatorListConfig,
    deviation_parameters: KinematicDeviations,
    device: torch.device,
) -> RigidBody:
    """
    Create a kinematic model to use in the test.

    Parameters
    ----------
    actuator_configuration : ActuatorListConfig
        The configuration of the actuators.
    deviation_parameters : KinematicDeviations
        The kinematic deviations.
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
        initial_orientation=torch.tensor([0.0, -1.0, 0.0, 0.0], device=device),
        deviation_parameters=deviation_parameters,
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
                    [0.9999, 0.0104, 0.0000, -0.0019],
                    [-0.0074, 0.7107, 0.7035, -0.1891],
                    [0.0073, -0.7035, 0.7107, 0.0613],
                    [0.0000, 0.0000, 0.0000, 1.0000],
                ]
            ),
        ),
        (
            "kinematic_model_1",
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor(
                [
                    [0.7123, -0.7019, 0.0000, 0.1246],
                    [0.7019, 0.7122, -0.0103, -0.1255],
                    [0.0072, 0.0073, 0.9999, -0.0908],
                    [0.0000, 0.0000, 0.0000, 1.0000],
                ]
            ),
        ),
        (
            "kinematic_model_1",
            torch.tensor([0.0, -1.0, 0.0, 0.0]),
            torch.tensor(
                [
                    [9.9997e-01, 7.3368e-03, 0.0000e00, -1.3026e-03],
                    [-7.3367e-03, 9.9996e-01, -5.1375e-03, -1.7708e-01],
                    [-3.7693e-05, 5.1374e-03, 9.9999e-01, -9.0411e-02],
                    [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                ]
            ),
        ),
        (
            "kinematic_model_1",
            torch.tensor([-1.0, 0.0, 0.0, 0.0]),
            torch.tensor(
                [
                    [0.7019, 0.7123, 0.0000, -0.1265],
                    [-0.7122, 0.7019, -0.0103, -0.1237],
                    [-0.0073, 0.0072, 0.9999, -0.0908],
                    [0.0000, 0.0000, 0.0000, 1.0000],
                ]
            ),
        ),
        (
            "kinematic_model_1",
            torch.tensor([0.0, -1.0, 1.0, 0.0]),
            torch.tensor(
                [
                    [1.0000, 0.0080, 0.0000, -0.0014],
                    [-0.0074, 0.9258, 0.3780, -0.1982],
                    [0.0030, -0.3779, 0.9258, -0.0158],
                    [0.0000, 0.0000, 0.0000, 1.0000],
                ]
            ),
        ),
        (
            "kinematic_model_2",
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
            torch.tensor(
                [
                    [0.9999, 0.0104, 0.0000, -0.0019],
                    [-0.0074, 0.7107, 0.7035, 0.8109],
                    [0.0073, -0.7035, 0.7107, 0.0613],
                    [0.0000, 0.0000, 0.0000, 1.0000],
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
    Test that the alignment works as desired.

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
    torch.testing.assert_close(
        orientation_matrix, expected.to(device), atol=5e-4, rtol=5e-4
    )
