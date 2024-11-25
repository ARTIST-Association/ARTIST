from logging import log
from typing import Union
from artist.util import config_dictionary
from typing_extensions import Self
from artist.field.tower_area import TowerArea
import h5py
import torch

class Towers(torch.nn.Module):
    """
    Wrap the list of tower areas as a ``torch.nn.Module`` to allow gradient calculation.

    Attributes
    ----------
    tower_area_list : list[TowerArea]
        A list of tower areas included in the scenario.

    Methods
    -------
    from_hdf5()
        Load the list of tower areas from an HDF5 file.
    """

    def __init__(self, tower_area_list: list[TowerArea]):
        """
        Initialize the Towers.

        The towers consist of one or more tower areas. The towers are positioned in front of the heliostats.

        Parameters
        ----------
        receiver_list : List[Receiver]
            The list of receivers included in the scenario.
        """
        super(Towers, self).__init__()
        self.tower_area_list = tower_area_list

    @classmethod
    def from_hdf5(
        cls, config_file: h5py.File, device: Union[torch.device, str] = "cuda"
    ) -> Self:
        """
        Load towers from an HDF5 file.

        Parameters
        ----------
        config_file : h5py.File
            The HDF5 file containing the configuration to be loaded.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        Towers
            The towers loaded from the HDF5 file.
        """
        log.info("Loading the towers from an HDF5 file.")
        device = torch.device(device)
        tower_areas = [
            TowerArea.from_hdf5(
                config_file=config_file[config_dictionary.tower_areas_key][area_name],
                area_name=area_name,
                device=device,
            )
            for area_name in config_file[config_dictionary.tower_areas_key].keys()
        ]
        return cls(tower_area_list=tower_areas)
