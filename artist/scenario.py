from typing import List, Self

import h5py

from artist.field import Heliostat
from artist.field.receiver import Receiver
from artist.scene import LightSource, Sun
from artist.util import config_dictionary


class Scenario:
    """
    This class represents a scenario that is loaded by ARTIST.

    Attributes
    ----------
    receiver : Receiver
        The receiver for the scenario.
    light_source : LightSource
        The light source for the scenario.
    heliostats : List[Heliostat]
        A list of heliostats included in the scenario.

    Methods
    -------
    load_scenario_from_hdf5()
        Class method to initialize the scenario from an hdf5 file.
    """

    def __init__(
        self, receiver: Receiver, light_source: LightSource, heliostats: List[Heliostat]
    ) -> None:
        """
        Initialize the scenario.

        Parameters
        ----------
        receiver : Receiver
            The receiver for the scenario.
        light_source : LightSource
            The light source for the scenario.
        heliostats : List[Heliostat]
            A list of heliostats included in the scenario.
        """
        self.receiver = receiver
        self.light_source = light_source
        self.heliostats = heliostats

    @classmethod
    def load_scenario_from_hdf5(cls, scenario_file: h5py.File) -> Self:
        """
        Class method to load the scenario from an hdf5 file.

        Parameters
        ----------
        scenario_file : h5py.File
            The config file containing all the information about the scenario being loaded.

        Returns
        -------
        Scenario
            The ARTIST scenario loaded from the hdf5 file.
        """
        receiver = Receiver.from_hdf5(config_file=scenario_file)
        light_source = Sun.from_hdf5(config_file=scenario_file)
        heliostat_list = []
        for heliostat_name in scenario_file[config_dictionary.heliostat_prefix][
            config_dictionary.heliostat_names
        ]:
            heliostat_list.append(
                Heliostat.from_hdf5(
                    config_file=scenario_file,
                    heliostat_name=heliostat_name.decode("utf-8"),
                )
            )

        return cls(
            receiver=receiver, light_source=light_source, heliostats=heliostat_list
        )
