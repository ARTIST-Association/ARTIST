from unittest.mock import MagicMock, patch

import pytest
import torch

from artist.field.heliostat_field import HeliostatField
from artist.field.heliostat_group_rigid_body import HeliostatGroupRigidBody
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util.scenario import Scenario


def test_trace_rays_unaligned_error(device: torch.device) -> None:
    """
    Test that `some_method` raises ValueError under specific conditions.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    mock_scenario = MagicMock(spec=Scenario)
    mock_heliostat_field = MagicMock(spec=HeliostatField)
    mock_heliostat_group = MagicMock(spec=HeliostatGroupRigidBody)
    mock_heliostat_group.number_of_heliostats = 2
    mock_scenario.heliostat_field = mock_heliostat_field
    mock_scenario.heliostat_field.heliostat_groups = [mock_heliostat_group]
    mock_scenario.heliostat_field.heliostat_groups[0].aligned_heliostats = torch.tensor(
        [[0.0], [1.0]], device=device
    )

    mock_ray_tracer = MagicMock(spec=HeliostatRayTracer)
    mock_ray_tracer.scenario = mock_scenario
    mock_ray_tracer.heliostat_group = mock_heliostat_group
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
                    [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device=device
                ),
                active_heliostats_indices=torch.tensor([0, 1], device=device),
                target_area_indices=torch.tensor([0, 0], device=device),
                device=device,
            )
            mock_method.assert_called_once()
        assert "Not all active heliostats have been aligned." in str(exc_info.value)
