import logging
from collections.abc import Sequence
from typing import cast

import h5py
import torch
from typing_extensions import Self

from artist.field.tower_target_areas import TowerTargetAreas
from artist.field.tower_target_areas_cylindrical import TowerTargetAreasCylindrical
from artist.field.tower_target_areas_planar import TowerTargetAreasPlanar
from artist.util import index_mapping
from artist.util.environment_setup import get_device

log = logging.getLogger(__name__)
"""A logger for the solar tower."""


class SolarTower:
    """
    The solar tower with its associated target areas.

    A solar tower holds two types of target areas (planar and cylindrical tower surfaces)
    onto which heliostats focus the reflected sunlight.
    Target areas are grouped by geometry type. Within each type, they are indexed
    consecutively, with planar areas assigned lower global indices than cylindrical ones.

    Attributes
    ----------
    target_areas : Sequence[TowerTargetAreas]
        List containing all target area groups, ordered as planar first, cylindrical second.
    number_of_target_area_types : int
        Number of distinct target area geometry types (e.g., 2 for planar and cylindrical).
    number_of_target_areas_per_type : torch.Tensor
        Number of individual target areas in each geometry type group.
        Shape is ``[number_of_target_area_types]``.
    target_name_to_index : dict[str, int]
        Mapping from a target area name to its global integer index.
    index_to_target_area : list[tuple[TowerTargetAreas, int]]
        Mapping from a global target area index to the corresponding target area object.

    Methods
    -------
    from_hdf5()
        Load a solar tower from an HDF5 file.
    get_centers_of_target_areas()
        Get the center coordinates of the specified target areas.
    """

    def __init__(
        self,
        target_areas: Sequence[TowerTargetAreas],
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize the solar tower with its target areas.

        Parameters
        ----------
        target_areas : Sequence[TowerTargetAreas]
            A list containing all target area groups. The expected order is planar
            target areas first, followed by cylindrical target areas.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        device = get_device(device=device)

        self.target_areas = target_areas
        self.number_of_target_area_types = len(self.target_areas)
        self.number_of_target_areas_per_type = torch.tensor(
            [
                target_area_type.number_of_target_areas
                for target_area_type in self.target_areas
            ],
            device=device,
        )

        self.target_name_to_index = {}
        idx = 0
        for target_area_type in self.target_areas:
            for name in target_area_type.names:
                self.target_name_to_index[name] = idx
                idx += 1

        self.index_to_target_area = []
        for target_area_type in self.target_areas:
            for local_idx, _ in enumerate(target_area_type.names):
                self.index_to_target_area.append((target_area_type, local_idx))

    @classmethod
    def from_hdf5(
        cls,
        config_file: h5py.File,
        device: torch.device | None = None,
    ) -> Self:
        """
        Load a solar tower from an HDF5 file.

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
        Self
            A ``SolarTower`` instance loaded from the HDF5 file.
        """
        device = get_device(device=device)

        target_areas_planar = TowerTargetAreasPlanar.from_hdf5(
            config_file=config_file, device=device
        )
        target_areas_cylindrical = TowerTargetAreasCylindrical.from_hdf5(
            config_file=config_file, device=device
        )

        # This defines the order of the target areas, first the planar ones, then the cylindrical ones.
        return cls(
            target_areas=[target_areas_planar, target_areas_cylindrical], device=device
        )

    def get_centers_of_target_areas(
        self,
        target_area_indices: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Get the center coordinates of the specified target areas.

        For planar target areas, the center is returned directly. For cylindrical target areas,
        the center is offset outward along the surface normal by the cylinder radius, giving
        the point on the curved surface facing the heliostats.

        Parameters
        ----------
        target_area_indices : torch.Tensor
            Global target area indices (planar first, cylindrical second) for which
            to retrieve the center coordinates.
            Shape is ``[number_of_active_heliostats]``.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            Center coordinates of the requested target areas in homogeneous coordinates.
            Shape is ``[number_of_active_heliostats, 4]``.
        """
        device = get_device(device=device)

        centers = torch.zeros((target_area_indices.shape[0], 4), device=device)

        planar_mask = (
            target_area_indices
            < self.number_of_target_areas_per_type[index_mapping.planar_target_areas]
        )
        if target_area_indices[planar_mask].numel() > 0:
            planar = cast(
                TowerTargetAreasPlanar,
                self.target_areas[index_mapping.planar_target_areas],
            )
            centers[planar_mask] = planar.centers[target_area_indices[planar_mask]]

        if target_area_indices[~planar_mask].numel() > 0:
            cylinder_indices = (
                target_area_indices[~planar_mask]
                - self.number_of_target_areas_per_type[
                    index_mapping.planar_target_areas
                ]
            )
            cylindrical = cast(
                TowerTargetAreasCylindrical,
                self.target_areas[index_mapping.cylindrical_target_areas],
            )
            centers[~planar_mask] = (
                cylindrical.centers[cylinder_indices]
                + cylindrical.radii[cylinder_indices][:, None]
                * cylindrical.normals[cylinder_indices]
            )
        centers[:, 3] = 1.0
        return centers
