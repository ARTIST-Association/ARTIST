import logging
from typing import List

import h5py
import torch.nn
from typing_extensions import Self

from artist.field.receiver import Receiver
from artist.util import config_dictionary

log = logging.getLogger(__name__)


class ReceiverField(torch.nn.Module):
    """
    This class wraps the list of receivers as a ``torch.nn.Module`` to allow gradient calculation.

    Attributes
    ----------
    receiver_list : List[Receiver]
        A list of receivers included in the scenario.

    Methods
    -------
    from_hdf5()
        Load the list of receivers from an HDF5 file.
    """

    def __init__(self, receiver_list: List[Receiver]):
        """
        Initialize the heliostat field.

        Parameters
        ----------
        receiver_list : List[Heliostat]
            The list of heliostats included in the scenario.
        """
        super(ReceiverField, self).__init__()
        self.receiver_list = receiver_list

    @classmethod
    def from_hdf5(cls, config_file: h5py.File) -> Self:
        """
        Load a receiver field from an HDF5 file.

        Parameters
        ----------
        config_file : h5py.File
            The HDF5 file containing the configuration to be loaded.

        Returns
        -------
        ReceiverField
            The receiver field loaded from the HDF5 file.
        """
        log.info("Loading a receiver field from an HDF5 file.")
        receiver_field = [
            Receiver.from_hdf5(
                config_file=config_file[config_dictionary.receiver_key][receiver_name],
                receiver_name=receiver_name,
            )
            for receiver_name in config_file[config_dictionary.receiver_key].keys()
        ]
        return cls(receiver_list=receiver_field)
