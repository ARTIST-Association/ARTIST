import logging

import h5py
import torch
from typing_extensions import Self

log = logging.getLogger(__name__)
"""A logger for the planar tower target areas."""


class TowerTargetAreas:
    """
    The tower target areas.

    Individual target areas are not saved as separate entities, instead separate tensors for each
    target area property exist. Each property tensor or list contains information about this property
    for all target areas.

    Attributes
    ----------
    names : list[str]
        The name of each target area.
    centers : torch.Tensor
        The center point coordinate of each target area.
        Tensor of shape [number_of_target_areas, 4].
    normal_vectors : torch.Tensor
        The normal vector of each planar target area.
        Tensor of shape [number_of_target_areas, 4].
    number_of_target_areas : int
        The total number of planar target areas on all towers in the scenario.

    Methods
    -------
    from_hdf5()
        Load all target areas from an HDF5 file.
    """

    def __init__(
        self,
        names: list[str],
        centers: torch.Tensor,
        normals: torch.Tensor,
    ) -> None:
        """
        Initialize the planar target areas.

        Parameters
        ----------
        names : list[str]
            The name of each target area.
        centers : torch.Tensor
            The center point coordinate of each target area.
            Tensor of shape [number_of_target_areas, 4].
        normal_vectors : torch.Tensor
            The normal vector of each target area.
            Tensor of shape [number_of_target_areas, 4].
        dimensions : torch.Tensor
            The dimensions of each target area (width, then height).
            Tensor of shape [number_of_target_areas, 2].
        """
        self.names = names
        self.centers = centers
        self.normals = normals

        self.number_of_target_areas = len(self.names)

    @classmethod
    def from_hdf5(
        cls, config_file: h5py.File, device: torch.device | None = None
    ) -> Self:
        """
        Load all planar target areas from an HDF5 file.

        Parameters
        ----------
        config_file : h5py.File
            The HDF5 file containing the configuration to be loaded.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Raises
        ------
        NotImplementedError
            This abstract method must be overridden.
        """
        raise NotImplementedError("Must be overridden!")