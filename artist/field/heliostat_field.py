import logging
from typing import Union

import h5py
import torch.nn
from typing_extensions import Self

from artist.field.heliostat import Heliostat
from artist.util import config_dictionary
from artist.util.configuration_classes import (
    ActuatorListConfig,
    KinematicLoadConfig,
    SurfaceConfig,
)

log = logging.getLogger(__name__)
"""A logger for the heliostat field."""


class HeliostatField(torch.nn.Module):
    """
    Wrap the heliostat list as a ``torch.nn.Module`` to allow gradient calculation.

    Attributes
    ----------
    heliostat_list : list[Heliostat]
        A list of heliostats included in the scenario.

    Methods
    -------
    from_hdf5()
        Load the list of heliostats from an HDF5 file.
    forward()
        Specify the forward pass.
    """

    def __init__(self, heliostat_list: list[Heliostat]):
        """
        Initialize the heliostat field.

        A heliostat field consists of many heliostats that have a unique position in the field. The
        heliostats in the field are aligned individually to reflect the incoming light in a way that
        ensures maximum efficiency for the whole power plant.

        Parameters
        ----------
        heliostat_list : list[Heliostat]
            The list of heliostats included in the scenario.
        """
        super(HeliostatField, self).__init__()
        self.heliostat_list = heliostat_list

    @classmethod
    def from_hdf5(
        cls,
        config_file: h5py.File,
        prototype_surface: SurfaceConfig,
        prototype_kinematic: KinematicLoadConfig,
        prototype_actuator: ActuatorListConfig,
        device: Union[torch.device, str] = "cuda",
    ) -> Self:
        """
        Load a heliostat field from an HDF5 file.

        Parameters
        ----------
        config_file : h5py.File
            The HDF5 file containing the configuration to be loaded.
        prototype_surface : SurfaceConfig
            The prototype for the surface configuration to be used if the heliostat has no individual surface.
        prototype_kinematic : KinematicLoadConfig
            The prototype for the kinematic configuration to be used if the heliostat has no individual kinematic.
        prototype_actuator : ActuatorListConfig
            The prototype for the actuator configuration to be used if the heliostat has no individual actuators.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        HeliostatField
            The heliostat field loaded from the HDF5 file.
        """
        log.info("Loading a heliostat field from an HDF5 file.")
        device = torch.device(device)
        heliostat_list = [
            Heliostat.from_hdf5(
                config_file=config_file[config_dictionary.heliostat_key][
                    heliostat_name
                ],
                prototype_surface=prototype_surface,
                prototype_kinematic=prototype_kinematic,
                prototype_actuator=prototype_actuator,
                heliostat_name=heliostat_name,
                device=device,
            )
            for heliostat_name in config_file[config_dictionary.heliostat_key].keys()
        ]
        return cls(heliostat_list=heliostat_list)

    def forward(self) -> None:
        """
        Specify the forward pass.

        Raises
        ------
        NotImplementedError
            Whenever called.
        """
        raise NotImplementedError("Not Implemented!")
