import pytest
import torch

from artist.field.kinematic_rigid_body import (
    RigidBody,
)


@pytest.fixture
def deviation_parameters(device: torch.device) -> torch.Tensor:
    """
    Define deviation parameters used in tests.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    torch.Tensor
        The deviation parameters.
    """
    deviation_parameters = torch.zeros((6, 18), device=device)
    deviation_parameters[:, 8] = 0.315
    deviation_parameters[:, 13] = -0.17755
    deviation_parameters[:, 14] = -0.4045

    return deviation_parameters


@pytest.fixture
def actuator_parameters(device: torch.device) -> torch.Tensor:
    """
    Define actuator parameters used in tests as measured experimentally.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    torch.Tensor
        A List containing parameters for each actuator.
    """
    parameters = torch.zeros((6, 6, 2), device=device)
    parameters[:, 0, 1] = 1
    parameters[:, 1, 0] = 154166.666
    parameters[:, 1, 1] = 154166.666
    parameters[:, 2, :] = 0.075
    parameters[:, 3, 0] = 0.34061
    parameters[:, 3, 1] = 0.3479
    parameters[:, 4, 0] = 0.3204
    parameters[:, 4, 1] = 0.309
    parameters[:, 5, 0] = -1.570796
    parameters[:, 5, 1] = 0.959931

    return parameters


@pytest.fixture
def kinematic_model(
    actuator_parameters: torch.Tensor,
    deviation_parameters: torch.Tensor,
    device: torch.device,
) -> RigidBody:
    """
    Create a kinematic model to use in the test.

    Parameters
    ----------
    actuator_parameters : torch.Tensor
        The parameters of the actuators.
    deviation_parameters : torch.Tensor
        The kinematic deviations.
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    RigidBody
        The kinematic model.
    """
    heliostat_positions = torch.tensor([[0.0, 0.0, 0.0, 1.0],
                             [0.0, 0.0, 0.0, 1.0],
                             [0.0, 0.0, 0.0, 1.0],
                             [0.0, 0.0, 0.0, 1.0],
                             [0.0, 0.0, 0.0, 1.0],
                             [0.0, 1.0, 0.0, 1.0]], device=device)
    aim_points = torch.tensor([[0.0, -10.0, 0.0, 1.0],
                              [0.0, -10.0, 0.0, 1.0],
                              [0.0, -10.0, 0.0, 1.0],
                              [0.0, -10.0, 0.0, 1.0],
                              [0.0, -10.0, 0.0, 1.0],
                              [0.0, -9.0, 0.0, 1.0]], device=device)
    initial_orientation = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=device)
    return RigidBody(
        number_of_heliostats=6,
        heliostat_positions=heliostat_positions,
        aim_points=aim_points,
        actuator_parameters=actuator_parameters,
        initial_orientations=initial_orientation.expand(6, 4),
        deviation_parameters=deviation_parameters,
        device=device,
    )


@pytest.mark.parametrize(
    "incident_ray_directions, expected",
    [
        (
            torch.tensor([[0.0, 0.0, -1.0, 0.0],
                          [-1.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0, 0.0],
                          [0.0000, 0.7071, -0.7071, 0.0000],
                          [0.0, 0.0, -1.0, 0.0]]),
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
                    ]
                ]
            ),
        ),
    ],
)
def test_orientation_matrix(
    request: pytest.FixtureRequest,
    kinematic_model: RigidBody,
    incident_ray_directions: torch.Tensor,
    expected: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test that the alignment works as desired.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The pytest fixture used to consider different test cases.
    kinematic_model : RigidBody
        The kinematic model fixture used to select the kinematic model used in the test case.
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
    orientation_matrix = kinematic_model.incident_ray_directions_to_orientations(
        incident_ray_directions=incident_ray_directions.to(device),
        active_heliostats_indices=torch.arange(0, 6, device=device), 
        device=device
    )
    torch.testing.assert_close(
        orientation_matrix, expected.to(device), atol=5e-4, rtol=5e-4
    )
