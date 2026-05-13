import pytest

from artist.raytracing import RestrictedDistributedSampler


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
    indices_per_rank: list[list[int]],
) -> None:
    """
    Test the distributed sampler.

    Parameters
    ----------
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
            rank=rank,
        )
        indices = list(sampler)

        assert indices == indices_per_rank[rank]
