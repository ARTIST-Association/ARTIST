import logging

import h5py
import torch
from typing_extensions import Self

from artist.field.tower_target_areas import TowerTargetAreas
from artist.util import config_dictionary
from artist.util.environment_setup import get_device

log = logging.getLogger(__name__)
"""A logger for the cylindrical tower target areas."""


class TowerTargetAreasCylindrical(TowerTargetAreas):
    """
    The cylindrical tower target areas.

    Individual cylindrical target areas are not saved as separate entities, instead separate tensors for each
    cylindrical target area property exist. Each property tensor or list contains information about this property
    for all cylindrical target areas.

    Attributes
    ----------
    names : list[str]
        Name of each cylindrical target area.
    centers : torch.Tensor
        Center coordinate of each cylindrical target area.
        The center is defined at the halfway point between top and bottom of the cylinder on the cylinder axis.
        Tensor of shape [number_of_target_areas, 4].
    normals : torch.Tensor
        Normal vector of each cylindrical target area in radians.
        Tensor of shape [number_of_target_areas, 4].
    axes : torch.Tensor
        Cylinder axes of all cylinder target areas.
        Tensor of shape [number_of_target_areas, 4].
    radii : torch.Tensor
        Radius of each cylindrical target area.
        Tensor of shape [number_of_target_areas].
    heights : torch.Tensor
        Height of each cylindrical target area.
        Tensor of shape [number_of_target_areas].
    opening_angles : torch.Tensor
        Opening angle of each cylindrical target area in radians.
        For full cylinders this is two pi or 360°.
        Tensor of shape [number_of_target_areas].
    number_of_target_areas : int
        Total number of cylindrical target areas on all towers in the scenario.

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
        axes: torch.Tensor,
        radii: torch.Tensor,
        heights: torch.Tensor,
        opening_angles: torch.Tensor,
    ) -> None:
        """
        Initialize the cylindrical target areas.

        Parameters
        ----------
        names : list[str]
            Name of each cylindrical target area.
        centers : torch.Tensor
            Center coordinate of each cylindrical target area.
            The center is defined at the halfway point between top and bottom of the cylinder on the cylinder axis.
            Tensor of shape [number_of_target_areas, 4].
        normals : torch.Tensor
            Normal vector of each cylindrical target area in radians.
            Tensor of shape [number_of_target_areas, 4].
        axes : torch.Tensor
            Cylinder axes of all cylinder target areas.
            Tensor of shape [number_of_target_areas, 4].
        radii : torch.Tensor
            Radius of each cylindrical target area.
            Tensor of shape [number_of_target_areas].
        heights : torch.Tensor
            Height of each cylindrical target area.
            Tensor of shape [number_of_target_areas].
        opening_angles : torch.Tensor
            Opening angle of each cylindrical target area in radians.
            For full cylinders this is two pi or 360°.
            Tensor of shape [number_of_target_areas].
        """
        super().__init__(
            names=names,
            centers=centers,
            normals=normals,
        )
        self.radii = radii
        self.heights = heights
        self.axes = axes
        self.opening_angles = opening_angles

    @classmethod
    def from_hdf5(
        cls, config_file: h5py.File, device: torch.device | None = None
    ) -> Self:
        """
        Load all cylindrical target areas from an HDF5 file.

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
        TowerTargetAreasCylindrical
            The target areas loaded from the HDF5 file.
        """
        device = get_device(device=device)

        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        if rank == 0:
            log.info("Loading the cylindrical tower target areas from an HDF5 file.")

        number_of_target_areas = len(
            config_file[config_dictionary.target_area_cylindrical_key]
        )

        names = []
        radii = torch.zeros((number_of_target_areas), device=device)
        centers = torch.zeros((number_of_target_areas, 4), device=device)
        heights = torch.zeros((number_of_target_areas), device=device)
        axes = torch.zeros((number_of_target_areas, 4), device=device)
        normals = torch.zeros((number_of_target_areas, 4), device=device)
        opening_angles = torch.zeros((number_of_target_areas), device=device)

        for index, target_area_name in enumerate(
            config_file[config_dictionary.target_area_cylindrical_key].keys()
        ):
            single_target_area_config = config_file[
                config_dictionary.target_area_cylindrical_key
            ][target_area_name]
            names.append(target_area_name)
            radii[index] = torch.tensor(
                single_target_area_config[
                    config_dictionary.target_area_cylinder_radius
                ][()],
                dtype=torch.float,
                device=device,
            )
            centers[index] = torch.tensor(
                single_target_area_config[
                    config_dictionary.target_area_cylinder_center
                ][()],
                dtype=torch.float,
                device=device,
            )
            heights[index] = torch.tensor(
                single_target_area_config[
                    config_dictionary.target_area_cylinder_height
                ][()],
                dtype=torch.float,
                device=device,
            )
            axes[index] = torch.tensor(
                single_target_area_config[config_dictionary.target_area_cylinder_axis][
                    ()
                ],
                dtype=torch.float,
                device=device,
            )
            normals[index] = torch.tensor(
                single_target_area_config[
                    config_dictionary.target_area_cylinder_normal
                ][()],
                dtype=torch.float,
                device=device,
            )
            opening_angles[index] = torch.tensor(
                single_target_area_config[
                    config_dictionary.target_area_cylinder_opening_angle
                ][()],
                dtype=torch.float,
                device=device,
            )

        return cls(
            names=names,
            radii=radii,
            centers=centers,
            heights=heights,
            axes=axes,
            normals=normals,
            opening_angles=opening_angles,
        )
