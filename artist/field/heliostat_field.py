import logging
from typing import List

import h5py
import torch.nn
from typing_extensions import Self

from artist.field import Heliostat
from artist.util import config_dictionary
from artist.util.configuration_classes import (
    ActuatorListConfig,
    KinematicConfig,
    SurfaceConfig,
)

log = logging.getLogger(__name__)


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
    def from_hdf5(
        cls,
        config_file: h5py.File,
        prototype_surface: SurfaceConfig,
        prototype_kinematic: KinematicConfig,
        prototype_actuator: ActuatorListConfig,
    ) -> Self:
        """
        Load a heliostat field from an HDF5 file.

        Parameters
        ----------
        config_file : h5py.File
            The HDF5 file containing the configuration to be loaded.
        prototype_surface  : SurfaceConfig
            The prototype for the surface configuration to be used if the heliostat has no individual surface.
        prototype_kinematic : KinematicConfig
            The prototype for the kinematic configuration for when the heliostat has no individual kinematic.
        prototype_actuator : ActuatorListConfig
            The prototype for the actuator configuration for when the heliostat has no individual actuators.

        Returns
        -------
        HeliostatField
            The heliostat field loaded from the HDF5 file.
        """
        log.info("Loading a heliostat field from an HDF5 file.")
        heliostat_list = [
            (
                log.info(f"Loading {heliostat_name} from an HDF5 file"),
                Heliostat.from_hdf5(
                    config_file=config_file[config_dictionary.heliostat_key][
                        heliostat_name
                    ],
                    prototype_surface=prototype_surface,
                    prototype_kinematic=prototype_kinematic,
                    prototype_actuator=prototype_actuator,
                ),
            )
            for heliostat_name in config_file[config_dictionary.heliostat_key].keys()
        ]
        return cls(heliostat_list=heliostat_list)
