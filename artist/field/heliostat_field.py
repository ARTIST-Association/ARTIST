from typing import List

import h5py
import torch.nn
from typing_extensions import Self

from artist.field import Heliostat
from artist.util import config_dictionary


class HeliostatField(torch.nn.Module):
    """
    This class wraps the heliostat list as a torch.nn.Module to allow gradient calculation.

    Attributes
    ----------
    heliostat_list : List[Heliostat]
        A list of heliostats included in the scenario.

    Methods
    -------
    from_hdf5()
        Load the list of heliostats from an HDF5 file.
    """

    def __init__(self, heliostat_list: List[Heliostat]):
        """
        Initialize the heliostat field.

        Parameters
        ----------
        heliostat_list : List[Heliostat]
            The list of heliostats included in the scenario.
        """
        super(HeliostatField, self).__init__()
        self.heliostat_list = heliostat_list

    @classmethod
    def from_hdf5(cls, config_file: h5py.File) -> Self:
        """
        Load a heliostat field from an HDF5 file.

        Parameters
        ----------
        config_file : h5py.File
            The HDF5 file containing the configuration to be loaded.

        Returns
        -------
        HeliostatField
            The heliostat field loaded from the HDF5 file.
        """
        heliostat_list = [
            Heliostat.from_hdf5(
                config_file=config_file,
                heliostat_name=heliostat_name.decode("utf-8"),
            )
            for heliostat_name in config_file[config_dictionary.heliostat_prefix][
                config_dictionary.heliostat_names
            ]
        ]
        return cls(heliostat_list=heliostat_list)
