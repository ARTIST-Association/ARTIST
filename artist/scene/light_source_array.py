import logging

import h5py
import torch.nn
from typing_extensions import Self

from artist.scene.light_source import LightSource
from artist.util import config_dictionary, type_mappings
from artist.util.environment_setup import get_device

log = logging.getLogger(__name__)
"""A logger for the light source array."""


class LightSourceArray:
    """
    The light source array.

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
        sends out the rays that are then used for ray tracing. The position of the light sources can be
        dynamic throughout a time span.

        Parameters
        ----------
        light_source_list : list[LightSource]
            A list of light sources included in the scenario.
        """
        super(LightSourceArray, self).__init__()
        self.light_source_list = light_source_list

    @classmethod
    def from_hdf5(
        cls, config_file: h5py.File, device: torch.device | None = None
    ) -> Self:
        """
        Load a light source array from an HDF5 file.

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
        LightSourceArray
            The light source array loaded from the HDF5 file.
        """
        device = get_device(device=device)

        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        if rank == 0:
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
                ls_object = type_mappings.light_source_type_mapping[mapping_key]
                light_source_array.append(
                    ls_object.from_hdf5(
                        config_file=config_file[config_dictionary.light_source_key][ls],
                        light_source_name=ls,
                        device=device,
                    )
                )
            except KeyError:
                raise KeyError(
                    f"Currently the selected light source: {mapping_key} is not supported."
                )
        return cls(light_source_list=light_source_array)
