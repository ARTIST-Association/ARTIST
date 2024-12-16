import logging
from typing import Union

import h5py
import torch.nn
from typing_extensions import Self

from artist.field.receiver import Receiver
from artist.util import config_dictionary

log = logging.getLogger(__name__)
"""A logger for the receiver field."""


class ReceiverField(torch.nn.Module):
    """
    Wrap the list of receivers as a ``torch.nn.Module`` to allow gradient calculation.

    Attributes
    ----------
    receiver_list : list[Receiver]
        A list of receivers included in the scenario.

    Methods
    -------
    from_hdf5()
        Load the list of receivers from an HDF5 file.
    forward()
        Specify the forward pass.
    """

    def __init__(self, receiver_list: list[Receiver]):
        """
        Initialize the receiver field.

        A receiver field consists of one or more receivers that are positioned in front of the heliostats.
        The receiver field is provided with a list of receivers to initialize the receivers.

        Parameters
        ----------
        receiver_list : List[Receiver]
            The list of receivers included in the scenario.
        """
        super(ReceiverField, self).__init__()
        self.receiver_list = receiver_list

    @classmethod
    def from_hdf5(
        cls, config_file: h5py.File, device: Union[torch.device, str] = "cuda"
    ) -> Self:
        """
        Load a receiver field from an HDF5 file.

        Parameters
        ----------
        config_file : h5py.File
            The HDF5 file containing the configuration to be loaded.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        ReceiverField
            The receiver field loaded from the HDF5 file.
        """
        log.info("Loading a receiver field from an HDF5 file.")
        device = torch.device(device)
        receiver_field = [
            Receiver.from_hdf5(
                config_file=config_file[config_dictionary.receiver_key][receiver_name],
                receiver_name=receiver_name,
                device=device,
            )
            for receiver_name in config_file[config_dictionary.receiver_key].keys()
        ]
        return cls(receiver_list=receiver_field)

    def forward(self) -> None:
        """
        Specify the forward pass.

        Raises
        ------
        NotImplementedError
            Whenever called.
        """
        raise NotImplementedError("Not Implemented!")
