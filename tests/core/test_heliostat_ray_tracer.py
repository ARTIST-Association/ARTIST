from unittest.mock import MagicMock, patch

import pytest
import torch

from artist.core.heliostat_ray_tracer import HeliostatRayTracer, RestrictedDistributedSampler
from artist.field.heliostat_field import HeliostatField
from artist.field.heliostat_group_rigid_body import HeliostatGroupRigidBody
from artist.scenario.scenario import Scenario

@pytest.mark.parametrize(
    "number_of_samples, number_of_heliostats, world_size, indices_per_rank",
    [
        (12, 4, 1, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]),
        (12, 4, 2, [[0, 1, 2, 6, 7, 8], [3, 4, 5, 9, 10, 11]]),
        (12, 4, 3, [[0, 1, 2, 9, 10, 11], [3, 4, 5], [6, 7, 8]]),
        (12, 4, 4, [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]),
        (4, 1, 3, [[0, 1, 2, 3], [], []]),
        (4, 2, 3, [[0, 1], [2, 3], []]),
    ],
)
def test_distributed_sampler(
    number_of_samples: int,
    number_of_heliostats: int,
    world_size: int,
    indices_per_rank: list[list[int]]
) -> None:
    """
    number_of_samples : int
        Number of samples to distribute among ranks.
    number_of_heliostats : int
        Number of heliostats.
    world_size : int
        Total number of processes.
    indices_per_rank : list[list[int]]
        Expected indices for each available rank.
    
    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    for rank in range(world_size): 
        sampler = RestrictedDistributedSampler(
            number_of_samples=number_of_samples,
            number_of_active_heliostats=number_of_heliostats,
            world_size=world_size,
            rank = rank
        )
        indices = list(sampler)

        assert indices == indices_per_rank[rank]
    

@pytest.fixture()
def mock_scenario() -> Scenario:
    """
    Define a mock scenario used in tests.

    Returns
    -------
    Scenario
        The mocked scenario.
    """
    mock_scenario = MagicMock(spec=Scenario)
    mock_heliostat_field = MagicMock(spec=HeliostatField)
    mock_heliostat_group = MagicMock(spec=HeliostatGroupRigidBody)
    mock_scenario.heliostat_field = mock_heliostat_field
    mock_scenario.heliostat_field.heliostat_groups = [mock_heliostat_group]
    return mock_scenario


@pytest.mark.parametrize(
    "active_heliostats_mask_scenario, active_heliostats_mask, expected",
    [
        (
            torch.tensor([0, 0, 1, 0], dtype=torch.int32),
            torch.tensor([0, 1, 1, 0], dtype=torch.int32),
            "Some heliostats were not aligned and cannot be raytraced.",
        ),
        (
            torch.tensor([0, 5, 1, 0], dtype=torch.int32),
            torch.tensor([1, 1, 0, 1], dtype=torch.int32),
            "Some heliostats were not aligned and cannot be raytraced.",
        ),
    ],
)
def test_trace_rays_unaligned_heliostats_error(
    mock_scenario: Scenario,
    active_heliostats_mask_scenario: torch.Tensor,
    active_heliostats_mask: torch.Tensor,
    expected: str,
    device: torch.device,
) -> None:
    """
    Test that unaligned heliostats raise ValueError while raytracing.

    Parameters
    ----------
    mock_scenario : Scenario
        A mocked scenario.
    active_heliostats_mask_scenario : torch.Tensor
        The active heliostats mask defined in the scenario.
    active_heliostats_mask : torch.Tensor
        The active heliostats mask given to the trace rays method.
    expected : str
        The expected error message.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    mock_scenario.heliostat_field.heliostat_groups[
        0
    ].active_heliostats_mask = active_heliostats_mask_scenario
    mock_ray_tracer = MagicMock(spec=HeliostatRayTracer)
    mock_ray_tracer.scenario = mock_scenario
    mock_ray_tracer.heliostat_group = mock_scenario.heliostat_field.heliostat_groups[0]
    mock_ray_tracer.bitmap_resolution_e = 50
    mock_ray_tracer.bitmap_resolution_u = 50

    with patch(
        "artist.core.heliostat_ray_tracer.HeliostatRayTracer.trace_rays",
        wraps=HeliostatRayTracer.trace_rays,
    ) as mock_method:
        with pytest.raises(AssertionError) as exc_info:
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
                active_heliostats_mask=active_heliostats_mask,
                target_area_mask=torch.tensor([0]),
                device=device,
            )
            mock_method.assert_called_once()

        assert expected in str(exc_info.value)
