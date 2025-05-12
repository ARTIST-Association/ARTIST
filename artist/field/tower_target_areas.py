import logging
from typing import Union

import h5py
import torch.nn
from typing_extensions import Self

from artist.util import config_dictionary

log = logging.getLogger(__name__)
"""A logger for the tower target area array."""


class TowerTargetAreas(torch.nn.Module):
    """
    Wrap the list of tower target areas as a ``torch.nn.Module`` to allow gradient calculation.

    Attributes
    ----------
    target_area_list : list[TargetArea]
        A list of target areas included in the scenario.

    Methods
    -------
    from_hdf5()
        Load the list of target areas from an HDF5 file.
    forward()
        Specify the forward pass.
    """

    def __init__(
        self,
        names: list[str],
        geometries: list[str],
        centers: torch.Tensor,
        normal_vectors: torch.Tensor,
        dimensions: torch.Tensor,
        curvatures: torch.Tensor):
        """
        Initialize the target area array.

        A target area array consists of one or more target areas that are positioned
        on the solar tower, in front of the heliostats. The target area array is provided
        with a list of target areas to initialize the target areas.

        Parameters
        ----------
        target_area_list : List[TargetArea]
            The list of target areas included in the scenario.
        """
        super().__init__()
        self.names = names
        self.geometries = geometries
        self.centers = centers
        self.normal_vectors = normal_vectors
        self.dimensions = dimensions
        self.curvatures = curvatures

    @classmethod
    def from_hdf5(
        cls, config_file: h5py.File, device: Union[torch.device, str] = "cuda"
    ) -> Self:
        """
        Load a tower target array from an HDF5 file.

        Parameters
        ----------
        config_file : h5py.File
            The HDF5 file containing the configuration to be loaded.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        TargetAreaArray
            The target area array loaded from the HDF5 file.
        """
        log.info("Loading the tower target area array from an HDF5 file.")
        device = torch.device(device)

        number_of_target_areas = len(config_file[config_dictionary.target_area_key])

        names = []
        geometries = []
        centers = torch.zeros((number_of_target_areas, 4), device=device)
        normal_vectors = torch.zeros((number_of_target_areas, 4), device=device)
        dimensions = torch.zeros((number_of_target_areas, 2), device=device)
        curvatures = torch.zeros((number_of_target_areas, 2), device=device)

        for index, target_area_name in (
            enumerate(config_file[config_dictionary.target_area_key].keys())
        ):
            single_target_area_config = config_file[config_dictionary.target_area_key][target_area_name]
            names.append(target_area_name)
            geometries.append(single_target_area_config[config_dictionary.target_area_geometry][()].decode(
                "utf-8"
            ))
            centers[index] = torch.tensor(
                single_target_area_config[config_dictionary.target_area_position_center][()],
                dtype=torch.float,
                device=device,
            )
            normal_vectors[index] = torch.tensor(
                single_target_area_config[config_dictionary.target_area_normal_vector][()],
                dtype=torch.float,
                device=device,
            )
            dimensions[index, 0] = float(single_target_area_config[config_dictionary.target_area_plane_e][()])
            dimensions[index, 1] = float(single_target_area_config[config_dictionary.target_area_plane_u][()])

            if config_dictionary.target_area_curvature_e in single_target_area_config.keys():
                curvatures[index, 0] = float(
                    single_target_area_config[config_dictionary.target_area_curvature_e][()]
                )
            else:
                log.warning(
                    f"No curvature in the east direction set for the {target_area_name}!"
                )
            if config_dictionary.target_area_curvature_u in single_target_area_config.keys():
                curvatures[index, 1] = float(
                    single_target_area_config[config_dictionary.target_area_curvature_u][()]
                )
            else:
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

    def forward(self) -> None:
        """
        Specify the forward pass.

        Raises
        ------
        NotImplementedError
            Whenever called.
        """
        raise NotImplementedError("Not Implemented!")
