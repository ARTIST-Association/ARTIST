from typing import Iterator

import torch
from torch.utils.data import Dataset, Sampler

from artist.scene import LightSource
from artist.util import index_mapping


class DistortionsDataset(Dataset):
    """
    A dataset of distortions based on the model of the light source.

    Attributes
    ----------
    distortions_u : torch.Tensor
        The distortions in the up direction.
    distortions_e : torch.Tensor
        The distortions in the east direction.
    """

    def __init__(
        self,
        light_source: LightSource,
        number_of_points_per_heliostat: int,
        number_of_active_heliostats: int,
        random_seed: int = 7,
    ) -> None:
        """
        Initialize the dataset.

        This class implements a custom dataset according to the ``torch`` interface. The content of this
        dataset are the distortions. The distortions are used in our version of "heliostat"-tracing to
        indicate how each incoming ray must be multiplied and scattered on the heliostat. According to
        ``torch``, this dataset must implement a function to return the length of the dataset and one function
        to retrieve an element through an index.

        Parameters
        ----------
        light_source : LightSource
            The light source used to model the distortions.
        number_of_points_per_heliostat : int
            The number of points on the heliostats for which distortions are created.
        number_of_active_heliostats : int
            The number of active heliostats in the scenario.
        random_seed : int
            The random seed used for generating the distortions (default is 7).
        """
        self.distortions_u, self.distortions_e = light_source.get_distortions(
            number_of_points=number_of_points_per_heliostat,
            number_of_active_heliostats=number_of_active_heliostats,
            random_seed=random_seed,
        )

    def __len__(self) -> int:
        """
        Calculate the length of the dataset, i.e., the number of items contained.

        Returns
        -------
        int
            The length of the dataset.
        """
        return self.distortions_u.shape[index_mapping.heliostat_dimension]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select an item from the dataset.

        Parameters
        ----------
        idx : int
            The index of the item to select.

        Returns
        -------
        torch.Tensor
            The distortions in the up direction for the given index.
        torch.Tensor
            The distortions in the east direction for the given index.
        """
        return (
            self.distortions_u[idx],
            self.distortions_e[idx],
        )


class RestrictedDistributedSampler(Sampler):
    """
    Initializes a custom distributed sampler.

    The ``DistributedSampler`` from ``torch`` replicates samples if the size of the dataset
    is smaller than the world size, to assign data to each rank. This custom sampler
    can leave some ranks idle if the dataset is not large enough to distribute data to
    each rank. Replicated samples would mean replicated rays that physically do not exist.

    Attributes
    ----------
    rank_indices : int
        The indices corresponding to the ranks assigned samples.

    See Also
    --------
    :class:`torch.utils.data.Sampler` : Reference to the parent class.
    """

    def __init__(
        self,
        number_of_samples: int,
        number_of_active_heliostats: int,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        """
        Set up a custom distributed sampler to assign data to each rank or leave them idle.

        Parameters
        ----------
        number_of_samples : int
            Length of the dataset or total number of samples.
        number_of_active_heliostats : int
            Number of active heliostats.
        world_size : int
            World size or total number of processes (default is 1).
        rank : int
            Rank of the current process (default is 0).
        """
        super().__init__()
        number_of_active_ranks = min(number_of_active_heliostats, world_size)
        self.rank_indices = []

        if rank < number_of_active_ranks:
            number_of_samples_per_heliostat = (
                number_of_samples // number_of_active_heliostats
            )
            indices: list[int] = []

            for index in range(number_of_active_heliostats):
                if index % number_of_active_ranks == rank:
                    start = index * number_of_samples_per_heliostat
                    end = start + number_of_samples_per_heliostat
                    indices.extend(range(start, end))

            self.rank_indices = indices
        else:
            self.rank_indices = []

    def __iter__(self) -> Iterator[int]:
        """
        Generate a sequence of indices for the current rank's portion of the dataset.

        Returns
        -------
        Iterator[int]
            An iterator over indices for the current rank.
        """
        return iter(self.rank_indices)
