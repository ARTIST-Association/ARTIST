import logging

import h5py
import torch.nn
from typing_extensions import Self

from artist.scene.light_source import LightSource
from artist.scene.sun import Sun
from artist.util import config_dictionary

light_source_type_mapping = {config_dictionary.sun_key: Sun}
"""A type mapping dictionary that allows ARTIST to automatically infer the correct light source type."""

log = logging.getLogger(__name__)
"""A logger for the light source array."""


class LightSourceArray(torch.nn.Module):
    """
    Wrap the list of light sources as a ``torch.nn.Module`` to allow gradient calculation.

    Attributes
    ----------
    light_source_list : list[LightSource]
        A list of light sources included in the scenario.

    Methods
    -------
    from_hdf5()
        Load the list of light sources from an HDF5 file.
    """

    def __init__(self, light_source_list: list[LightSource]):
        """
        Initialize the light sources included in the considered scenario.

        The light source array bundles all light sources considered in the scenario. The light source
        sends out the rays that are then used for raytracing. The position of the light sources can be
        dynamic throughout a time span.

        Parameters
        ----------
        light_source_list : list[LightSource]
            A list of light sources included in the scenario.
        """
        super(LightSourceArray, self).__init__()
        self.light_source_list = light_source_list

    @classmethod
    def from_hdf5(cls, 
                  config_file: h5py.File,
                  device: torch.device="cpu") -> Self:
        """
        Load a light source array from an HDF5 file.

        Parameters
        ----------
        config_file : h5py.File
            The HDF5 file containing the configuration to be loaded.
        device : torch.device
            The device on which to initialize tensors (default is CPU).

        Returns
        -------
        LightSourceArray
            The light source array loaded from the HDF5 file.
        """
        log.info("Loading a light source array from an HDF5 file.")
        light_source_array = []
        # Iterate through each light source configuration in the list of light source configurations.
        for ls in config_file[config_dictionary.light_source_key].keys():
            mapping_key = config_file[config_dictionary.light_source_key][ls][
                config_dictionary.light_source_type
            ][()].decode("utf-8")
            # Try to load a light source from the given configuration. This will fail, if ARTIST
            # does not recognize the light source type defined in the configuration.
            try:
                ls_object = light_source_type_mapping[mapping_key]
                light_source_array.append(
                    ls_object.from_hdf5(
                        config_file=config_file[config_dictionary.light_source_key][ls],
                        light_source_name=ls,
                        device=device
                    )
                )
            except KeyError:
                raise KeyError(
                    f"Currently the selected light source: {mapping_key} is not supported."
                )
        return cls(light_source_list=light_source_array)
