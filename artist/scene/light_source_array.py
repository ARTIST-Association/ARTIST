from typing import List

import h5py
import torch.nn
from typing_extensions import Self

from artist.scene import LightSource
from artist.util import artist_type_mapping_dict, config_dictionary


class LightSourceArray(torch.nn.Module):
    """
    This class wraps the list of light sources as a torch.nn.Module to allow gradient calculation.

    Attributes
    ----------
    light_source_list : List[LightSource]
        A list of light sources included in the scenario.

    Methods
    -------
    from_hdf5()
        Load the list of light sources from an HDF5 file.
    """

    def __init__(self, light_source_list: List[LightSource]):
        """
        Initialize the heliostat field.

        Parameters
        ----------
        light_source_list : List[LightSource]
            A list of light sources included in the scenario.
        """
        super(LightSourceArray, self).__init__()
        self.light_source_list = light_source_list

    @classmethod
    def from_hdf5(cls, config_file: h5py.File) -> Self:
        """
        Load a light source array from an HDF5 file.

        Parameters
        ----------
        config_file : h5py.File
            The HDF5 file containing the configuration to be loaded.

        Returns
        -------
        LightSourceArray
            The light source array loaded from the HDF5 file.
        """
        light_source_array = []
        for ls in config_file[config_dictionary.light_source_key].keys():
            mapping_key = config_file[config_dictionary.light_source_key][ls][
                config_dictionary.light_source_type
            ][()].decode("utf-8")
            try:
                ls_object = artist_type_mapping_dict.light_source_type_mapping[
                    mapping_key
                ]
                light_source_array.append(
                    ls_object.from_hdf5(
                        config_file=config_file[config_dictionary.light_source_key][ls]
                    )
                )
            except KeyError:
                raise KeyError(
                    f"Currently the selected light source: {mapping_key} is not supported."
                )
        return cls(light_source_list=light_source_array)
