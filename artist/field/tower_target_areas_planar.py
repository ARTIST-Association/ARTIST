import logging

import h5py
import torch
from typing_extensions import Self

from artist.field.tower_target_areas import TowerTargetAreas
from artist.util import config_dictionary, index_mapping
from artist.util.environment_setup import get_device

log = logging.getLogger(__name__)
"""A logger for the planar tower target areas."""


class TowerTargetAreasPlanar(TowerTargetAreas):
    """
    The planar tower target areas.

    Individual planar target areas are not saved as separate entities, instead separate tensors for each
    planar target area property exist. Each property tensor or list contains information about this property
    for all planar target areas.

    Attributes
    ----------
    names : list[str]
        The name of each planar target area.
    centers : torch.Tensor
        The center point coordinate of each planar target area.
        Tensor of shape [number_of_target_areas, 4].
    normals : torch.Tensor
        The normal vector of each planar target area.
        Tensor of shape [number_of_target_areas, 4].
    dimensions : torch.Tensor
        The dimensions of each planar target area (width, then height).
        Tensor of shape [number_of_target_areas, 2].
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
        dimensions: torch.Tensor,
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
        normals : torch.Tensor
            The normal vector of each target area.
            Tensor of shape [number_of_target_areas, 4].
        dimensions : torch.Tensor
            The dimensions of each target area (width, then height).
            Tensor of shape [number_of_target_areas, 2].
        """
        super().__init__(
            names=names,
            centers=centers,
            normals=normals,
        )
        self.dimensions = dimensions

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

        Returns
        -------
        TowerTargetAreasPlanar
            The target areas loaded from the HDF5 file.
        """
        device = get_device(device=device)

        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        if rank == 0:
            log.info("Loading the planar tower target areas from an HDF5 file.")

        number_of_target_areas = len(
            config_file[config_dictionary.target_area_planar_key]
        )

        names = []
        centers = torch.zeros((number_of_target_areas, 4), device=device)
        normals = torch.zeros((number_of_target_areas, 4), device=device)
        dimensions = torch.zeros((number_of_target_areas, 2), device=device)

        for index, target_area_name in enumerate(
            config_file[config_dictionary.target_area_planar_key].keys()
        ):
            single_target_area_config = config_file[
                config_dictionary.target_area_planar_key
            ][target_area_name]
            names.append(target_area_name)
            centers[index] = torch.tensor(
                single_target_area_config[
                    config_dictionary.target_area_position_center
                ][()],
                dtype=torch.float,
                device=device,
            )
            normals[index] = torch.tensor(
                single_target_area_config[config_dictionary.target_area_normal_vector][
                    ()
                ],
                dtype=torch.float,
                device=device,
            )
            dimensions[index, index_mapping.target_area_plane_e] = float(
                single_target_area_config[config_dictionary.target_area_plane_e][()]
            )
            dimensions[index, index_mapping.target_area_plane_u] = float(
                single_target_area_config[config_dictionary.target_area_plane_u][()]
            )

        return cls(
            names=names,
            centers=centers,
            normals=normals,
            dimensions=dimensions,
        )
