import h5py
from typing_extensions import Self

from artist.field.heliostat_field import HeliostatField
from artist.field.receiver_field import ReceiverField
from artist.scene.light_source_array import LightSourceArray


class Scenario:
    """
    This class represents a scenario loaded by ARTIST.

    Attributes
    ----------
    receivers : ReceiverField
        A list of receivers included in the scenario.
    light_sources : LightSourceArray
        A list of light sources included in the scenario.
    heliostats : HeliostatField
        The heliostat field for the scenario.

    Methods
    -------
    load_scenario_from_hdf5()
        Class method to initialize the scenario from an HDF5 file.
    """

    def __init__(
        self,
        receivers: ReceiverField,
        light_sources: LightSourceArray,
        heliostat_field: HeliostatField,
    ) -> None:
        """
        Initialize the scenario.

        Parameters
        ----------
        receivers : ReceiverField
            A list of receivers included in the scenario.
        light_sources : LightSourceArray
            A list of light sources included in the scenario.
        heliostat_field : HeliostatField
            A field of heliostats included in the scenario.
        """
        self.receivers = receivers
        self.light_sources = light_sources
        self.heliostats = heliostat_field

    @classmethod
    def load_scenario_from_hdf5(cls, scenario_file: h5py.File) -> Self:
        """
        Class method to load the scenario from an HDF5 file.

        Parameters
        ----------
        scenario_file : h5py.File
            The config file containing all the information about the scenario being loaded.

        Returns
        -------
        Scenario
            The ARTIST scenario loaded from the HDF5 file.
        """
        receivers = ReceiverField.from_hdf5(config_file=scenario_file)
        light_sources = LightSourceArray.from_hdf5(config_file=scenario_file)
        heliostat_field = HeliostatField.from_hdf5(config_file=scenario_file)

        return cls(
            receivers=receivers,
            light_sources=light_sources,
            heliostat_field=heliostat_field,
        )
