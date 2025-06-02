from typing import Union
from unittest.mock import MagicMock, patch

import pytest
import torch

from artist.field.heliostat_field import HeliostatField
from artist.field.heliostat_group_rigid_body import HeliostatGroupRigidBody
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util.scenario import Scenario


@pytest.fixture(params=[(0, 0), (0, 4), (3, 2)])
def mock_scenario(
    request: pytest.FixtureRequest,
) -> Scenario:
    """
    Define a mock scenario used in tests.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The pytest fixture used to consider different test cases.

    Returns
    -------
    Scenario
        The mocked scenario.
    """
    mock_scenario = MagicMock(spec=Scenario)
    mock_heliostat_field = MagicMock(spec=HeliostatField)
    mock_heliostat_group = MagicMock(spec=HeliostatGroupRigidBody)
    mock_heliostat_group.number_of_heliostats = 4
    mock_heliostat_group.number_of_active_heliostats = request.param[0]
    mock_heliostat_group.number_of_aligned_heliostats = request.param[1]
    mock_scenario.heliostat_field = mock_heliostat_field
    mock_scenario.heliostat_field.heliostat_groups = [mock_heliostat_group]
    return mock_scenario


@pytest.fixture(params=[torch.tensor([0, 0, 0, 0]), None])
def target_area_mask(
    request: pytest.FixtureRequest, device: torch.device
) -> Union[torch.Tensor, None]:
    """
    Create a target area mask or None to use in the test.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The pytest fixture used to consider different test cases.
    device : torch.device
        The device on which to initialize tensors.

    Returns
    -------
    Union[torch.Tensor, None]
        The target area mask.
    """
    mask = request.param

    if mask is not None:
        mask = mask.to(device)
    return mask


def test_trace_rays_unaligned_heliostats_error(
    mock_scenario: Scenario,
    target_area_mask: Union[torch.Tensor, None],
    device: torch.device,
) -> None:
    """
    Test that unanligned heliostats raise ValueError while raytracing.

    Parameters
    ----------
    mock_scenario : Scenario
        A mocked scenario.
    target_area_mask : Union[torch.Tensor, None]
        The target area mask.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    mock_ray_tracer = MagicMock(spec=HeliostatRayTracer)
    mock_ray_tracer.scenario = mock_scenario
    mock_ray_tracer.heliostat_group = mock_scenario.heliostat_field.heliostat_groups[0]
    mock_ray_tracer.bitmap_resolution_e = 50
    mock_ray_tracer.bitmap_resolution_u = 50

    with patch(
        "artist.raytracing.heliostat_tracing.HeliostatRayTracer.trace_rays",
        wraps=HeliostatRayTracer.trace_rays,
    ) as mock_method:
        with pytest.raises(ValueError) as exc_info:
            mock_ray_tracer.trace_rays = MagicMock(wraps=HeliostatRayTracer.trace_rays)
            mock_ray_tracer.trace_rays(
                self=mock_ray_tracer,
                incident_ray_directions=torch.tensor(
                    [
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                    ],
                    device=device,
                ),
                target_area_mask=target_area_mask,
                device=device,
            )
            mock_method.assert_called_once()
        assert (
            "No heliostats are active or not all active heliostats have been aligned."
            in str(exc_info.value)
        )
