from unittest.mock import MagicMock, patch

import pytest
import torch

from artist.field.heliostat_field import HeliostatField
from artist.field.tower_target_area import TargetArea
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
    mock_scenario.heliostat_field = mock_heliostat_field
    mock_scenario.heliostat_field.all_aligned_heliostats = torch.tensor(
        [[0.0], [1.0]], device=device
    )

    mock_raytracer = MagicMock(spec=HeliostatRayTracer)
    mock_raytracer.scenario = mock_scenario
    mock_raytracer.bitmap_resolution_e = 50
    mock_raytracer.bitmap_resolution_u = 50

    mock_target_area = MagicMock(spec=TargetArea)

    with patch(
        "artist.raytracing.heliostat_tracing.HeliostatRayTracer.trace_rays",
        wraps=HeliostatRayTracer.trace_rays,
    ) as mock_method:
        with pytest.raises(ValueError) as exc_info:
            mock_raytracer.trace_rays = MagicMock(wraps=HeliostatRayTracer.trace_rays)
            mock_raytracer.trace_rays(
                self=mock_raytracer,
                incident_ray_direction=torch.tensor(
                    [0.0, 1.0, 0.0, 0.0], device=device
                ),
                target_area=mock_target_area,
                device=device,
            )
            mock_method.assert_called_once()
        assert "Not all heliostats have been aligned." in str(exc_info.value)
