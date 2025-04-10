import logging
from typing import Union

import h5py
import torch.nn
from typing_extensions import Self

from artist.field.tower_target_area import TargetArea
from artist.util import config_dictionary

log = logging.getLogger(__name__)
"""A logger for the tower target area array."""


class TargetAreaArray(torch.nn.Module):
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

    def __init__(self, target_area_list: list[TargetArea]):
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
        self.target_area_list = target_area_list

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
        target_area_array = [
            TargetArea.from_hdf5(
                config_file=config_file[config_dictionary.target_area_key][
                    target_area_name
                ],
                target_area_name=target_area_name,
                device=device,
            )
            for target_area_name in config_file[
                config_dictionary.target_area_key
            ].keys()
        ]
        return cls(target_area_list=target_area_array)

    def forward(self) -> None:
        """
        Specify the forward pass.

        Raises
        ------
        NotImplementedError
            Whenever called.
        """
        raise NotImplementedError("Not Implemented!")
