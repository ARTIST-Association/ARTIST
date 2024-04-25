import h5py
from typing_extensions import Self

from artist.field.heliostat_field import HeliostatField
from artist.field.receiver import Receiver
from artist.scene import LightSource, Sun


class Scenario:
    """
    This class represents a scenario that is loaded by ARTIST.

    Attributes
    ----------
    receiver : Receiver
        The receiver for the scenario.
    light_source : LightSource
        The light source for the scenario.
    heliostats : HeliostatField
        The heliostat field for the scenario.

    Methods
    -------
    load_scenario_from_hdf5()
        Class method to initialize the scenario from an HDF5 file.
    """

    def __init__(
        self,
        receiver: Receiver,
        light_source: LightSource,
        heliostat_field: HeliostatField,
    ) -> None:
        """
        Initialize the scenario.

        Parameters
        ----------
        receiver : Receiver
            The receiver for the scenario.
        light_source : LightSource
            The light source for the scenario.
        heliostat_field : HeliostatField
            A field of heliostats included in the scenario.
        """
        self.receiver = receiver
        self.light_source = light_source
        self.heliostats = heliostat_field

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
        heliostat_field = HeliostatField.from_hdf5(config_file=scenario_file)

        return cls(
            receiver=receiver,
            light_source=light_source,
            heliostat_field=heliostat_field,
        )
