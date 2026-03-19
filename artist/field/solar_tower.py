import logging
from collections.abc import Sequence

import h5py
import torch
from typing_extensions import Self

from artist.field.tower_target_areas import TowerTargetAreas
from artist.field.tower_target_areas_cylindrical import TowerTargetAreasCylindrical
from artist.field.tower_target_areas_planar import TowerTargetAreasPlanar
from artist.util.environment_setup import get_device

log = logging.getLogger(__name__)
"""A logger for the heliostat field."""


class SolarTower:
    """
    The heliostat field.

    A heliostat field consists of one or multiple heliostat groups. Each heliostat group contains all
    heliostats with a specific kinematics type and actuator type. The heliostats in the field are aligned
    individually to reflect the incoming light in a way that ensures maximum efficiency for the whole power plant.

    Attributes
    ----------
    heliostat_groups : Sequence[HeliostatGroup]
        A list containing all heliostat groups.
    number_of_heliostat_groups : int
        The number of different heliostat groups in the heliostat field.
    number_of_heliostats_per_group : torch.Tensor
        The number of heliostats per group.
        Tensor of shape [number_of_heliostat_groups].

    Methods
    -------
    from_hdf5()
        Load a heliostat field from an HDF5 file.
    update_surfaces()
        Update surface points and normals using new NURBS control points.
    """

    def __init__(
        self,
        target_areas: Sequence[TowerTargetAreas],
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize the heliostat field with heliostat groups.

        Parameters
        ----------
        heliostat_groups : Sequence[HeliostatGroup]
            A list containing all heliostat groups.
        device : device: torch.device | None
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
        for target_areas in self.target_areas:
            for name in target_areas.names:
                self.target_name_to_index[name] = idx
                idx += 1

        self.index_to_target_area = []
        for target_areas in self.target_areas:
            for local_idx, name in enumerate(target_areas.names):
                self.index_to_target_area.append((target_areas, local_idx))

        


    @classmethod
    def from_hdf5(
        cls,
        config_file: h5py.File,
        device: torch.device | None = None,
    ) -> Self:
        device = get_device(device=device)

        target_areas_planar = TowerTargetAreasPlanar.from_hdf5(
            config_file=config_file, device=device
        )
        target_areas_cylindrical = TowerTargetAreasCylindrical.from_hdf5(
            config_file=config_file, device=device
        )

        # This defines the order of the target areas, first the planar ones, then the cylindrical ones.
        return cls(target_areas=[target_areas_planar, target_areas_cylindrical], device=device)
    

    def get_centers_of_target_areas(self, target_area_indices: torch.Tensor) -> torch.Tensor:
        aim_points = torch.stack([
            target_area.centers[local_idx]
            for target_area, local_idx in (self.index_to_target_area[i] for i in target_area_indices)
        ], dim=0)
        return aim_points