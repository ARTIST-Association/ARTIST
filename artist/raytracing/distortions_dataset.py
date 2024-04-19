from typing import Tuple

import torch
from torch.utils.data import Dataset

from artist.scene import LightSource


class DistortionsDataset(Dataset):
    """
    This class contains a data set of distortions based on the model of the light source.

    Attributes
    ----------
    distortions_u : torch.Tensor
        The distortions in the up direction.
    distortions_e : torth.Tensor
        The distortions in the east direction
    """

    def __init__(self, light_source: LightSource, number_of_points: int) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
        light_source : LightSource
             The light source used to model the distortions.
         number_of_points : int
             The number of points on the heliostat for which distortions are created.
        """
        self.distortions_u, self.distortions_e = light_source.get_distortions(
            number_of_points=number_of_points
        )

    def __len__(self) -> int:
        """
        Calculate the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.
        """
        return self.distortions_u.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select an item from the dataset.

        Parameters
        ----------
        idx : int
             The index of the item to select.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The distortions in the up and east direction for the given index.
        """
        return self.distortions_u[idx, :], self.distortions_e[idx, :]
