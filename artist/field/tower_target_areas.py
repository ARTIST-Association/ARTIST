import logging

import h5py
import torch
from typing_extensions import Self

from artist.util import config_dictionary
from artist.util.environment_setup import get_device

log = logging.getLogger(__name__)
"""A logger for the tower target areas."""


class TowerTargetAreas:
    """
    The tower target areas.

    Individual target areas are not saved as separate entities, instead separate tensors for each
    target area property exist. Each property tensor or list contains information about this property
    for all target areas.

    Attributes
    ----------
    names : list[str]
        The names of each target area.
    geometries : list[str]
        THe type of geometry of each target area.
    centers : torch.Tensor
        The center point coordinate of each target area.
        Tensor of shape [number_of_target_areas, 4].
    normal_vectors : torch.Tensor
        The normal vector of each target area.
        Tensor of shape [number_of_target_areas, 4].
    dimensions : torch.Tensor
        The dimensions of each target area (width, then height).
        Tensor of shape [number_of_target_areas, 2].
    curvatures : torch.Tensor
        The curvature of the target area in 2 dimensions (0.0 if not applicable).
        Tensor of shape [number_of_target_areas, 2].
    number_of_target_areas : int
        The total number of target areas on all towers in the scenario.

    Methods
    -------
    from_hdf5()
        Load all target areas from an HDF5 file.
    """

    def __init__(
        self,
        names: list[str],
        geometries: list[str],
        centers: torch.Tensor,
        normal_vectors: torch.Tensor,
        dimensions: torch.Tensor,
        curvatures: torch.Tensor,
    ):
        """
        Initialize the target area array.

        A target area array consists of one or more target areas that are positioned
        on the solar tower, in front of the heliostats. The target area array is provided
        with a list of target areas to initialize the target areas.

        Parameters
        ----------
        names : list[str]
            The names of each target area.
        geometries : list[str]
            The type of geometry of each target area.
        centers : torch.Tensor
            The center point coordinate of each target area.
            Tensor of shape [number_of_target_areas, 4].
        normal_vectors : torch.Tensor
            The normal vector of each target area.
            Tensor of shape [number_of_target_areas, 4].
        dimensions : torch.Tensor
            The dimensions of each target area (width, then height).
            Tensor of shape [number_of_target_areas, 2].
        curvatures : torch.Tensor
            The curvature of the target area in 2 dimensions (0.0 if not applicable).
            Tensor of shape [number_of_target_areas, 2].
        """
        self.names = names
        self.geometries = geometries
        self.centers = centers
        self.normal_vectors = normal_vectors
        self.dimensions = dimensions
        self.curvatures = curvatures

        self.number_of_target_areas = len(self.names)

    @classmethod
    def from_hdf5(
        cls, config_file: h5py.File, device: torch.device | None = None
    ) -> Self:
        """
        Load all target areas from an HDF5 file.

        Parameters
        ----------
        config_file : h5py.File
            The HDF5 file containing the configuration to be loaded.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        TowerTargetAreas
            The target areas loaded from the HDF5 file.
        """
        device = get_device(device=device)

        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        if rank == 0:
            log.info("Loading the tower target areas from an HDF5 file.")

        number_of_target_areas = len(config_file[config_dictionary.target_area_key])

        names = []
        geometries = []
        centers = torch.zeros((number_of_target_areas, 4), device=device)
        normal_vectors = torch.zeros((number_of_target_areas, 4), device=device)
        dimensions = torch.zeros((number_of_target_areas, 2), device=device)
        curvatures = torch.zeros((number_of_target_areas, 2), device=device)

        for index, target_area_name in enumerate(
            config_file[config_dictionary.target_area_key].keys()
        ):
            single_target_area_config = config_file[config_dictionary.target_area_key][
                target_area_name
            ]
            names.append(target_area_name)
            geometries.append(
                single_target_area_config[config_dictionary.target_area_geometry][
                    ()
                ].decode("utf-8")
            )
            centers[index] = torch.tensor(
                single_target_area_config[
                    config_dictionary.target_area_position_center
                ][()],
                dtype=torch.float,
                device=device,
            )
            normal_vectors[index] = torch.tensor(
                single_target_area_config[config_dictionary.target_area_normal_vector][
                    ()
                ],
                dtype=torch.float,
                device=device,
            )
            dimensions[index, 0] = float(
                single_target_area_config[config_dictionary.target_area_plane_e][()]
            )
            dimensions[index, 1] = float(
                single_target_area_config[config_dictionary.target_area_plane_u][()]
            )

            if (
                config_dictionary.target_area_curvature_e
                in single_target_area_config.keys()
            ):
                curvatures[index, 0] = float(
                    single_target_area_config[
                        config_dictionary.target_area_curvature_e
                    ][()]
                )
            else:
                if rank == 0:
                    log.warning(
                        f"No curvature in the east direction set for the {target_area_name}!"
                    )
            if (
                config_dictionary.target_area_curvature_u
                in single_target_area_config.keys()
            ):
                curvatures[index, 1] = float(
                    single_target_area_config[
                        config_dictionary.target_area_curvature_u
                    ][()]
                )
            else:
                if rank == 0:
                    log.warning(
                        f"No curvature in the up direction set for the {target_area_name}!"
                    )

        return cls(
            names=names,
            geometries=geometries,
            centers=centers,
            normal_vectors=normal_vectors,
            dimensions=dimensions,
            curvatures=curvatures,
        )
