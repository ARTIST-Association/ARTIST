import logging
from typing import Union
from artist.util import config_dictionary, utils_load_h5
from artist.util.configuration_classes import ActuatorConfig, ActuatorListConfig, ActuatorParameters, KinematicDeviations, KinematicLoadConfig, SurfaceConfig
from typing_extensions import Self
from artist.field.heliostat_field import HeliostatField
from artist.field.tower_target_area_array import TargetAreaArray
from artist.scene.light_source_array import LightSourceArray
import h5py
import torch

log = logging.getLogger(__name__)
"""A logger for the scenario."""

class Scenario:
    """
    Define a scenario loaded by ARTIST.

    Attributes
    ----------
    power_plant_position : torch.Tensor
        The position of the power plant as latitude, longitude, altitude.
    target_areas : TargetAreaArray
        A list of tower target areas included in the scenario.
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
        power_plant_position: torch.Tensor,
        target_areas: TargetAreaArray,
        light_sources: LightSourceArray,
        heliostat_field: HeliostatField,
    ) -> None:
        """
        Initialize the scenario.

        A scenario defines the physical objects and scene to be used by ``ARTIST``. Therefore, a scenario contains at
        least one target area that is a receiver, at least one light source and at least one heliostat in a heliostat field.
        ``ARTIST`` also supports scenarios that contain multiple target areas, multiple light sources, and multiple heliostats.

        Parameters
        ----------
        power_plant_position : torch.Tensor,
            The position of the power plant as latitude, longitude, altitude.
        target_areas : TargetAreaArray
            A list of tower target areas included in the scenario.
        light_sources : LightSourceArray
            A list of light sources included in the scenario.
        heliostat_field : HeliostatField
            A field of heliostats included in the scenario.
        """
        self.power_plant_position = power_plant_position
        self.target_areas = target_areas
        self.light_sources = light_sources
        self.heliostat_field = heliostat_field

    @classmethod
    def load_scenario_from_hdf5(
        cls, scenario_file: h5py.File, device: Union[torch.device, str] = "cuda"
    ) -> Self:
        """
        Class method to load the scenario from an HDF5 file.

        Parameters
        ----------
        scenario_file : h5py.File
            The config file containing all the information about the scenario being loaded.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        Scenario
            The ``ARTIST`` scenario loaded from the HDF5 file.
        """
        log.info(
            f"Loading an ``ARTIST`` scenario HDF5 file. This scenario file is version {scenario_file.attrs['version']}."
        )
        device = torch.device(device)
        
        power_plant_position = torch.tensor(
            scenario_file[config_dictionary.power_plant_key][
                config_dictionary.power_plant_position
            ][()]
        )
        target_areas = TargetAreaArray.from_hdf5(
            config_file=scenario_file, device=device
        )
        light_sources = LightSourceArray.from_hdf5(
            config_file=scenario_file, device=device
        )

        prototype_surface = utils_load_h5.load_surface_config(prototype=True,
                                                              scenario_file=scenario_file,
                                                              device=device)

        prototype_initial_orientation = torch.tensor(
            scenario_file[config_dictionary.prototype_key][
                config_dictionary.kinematic_prototype_key
            ][config_dictionary.kinematic_initial_orientation][()],
            dtype=torch.float,
            device=device,
        )

        prototype_kinematic_deviations_rigid_body = utils_load_h5.load_kinematic_deviations_rigid_body(
            prototype=True,
            scenario_file=scenario_file,
            log=log,
            device=device
        )

        prototype_actuators = utils_load_h5.load_actuators(
            prototype=True,
            scenario_file=scenario_file,
            log=log,
            device=device
        )

        heliostat_field = HeliostatField.from_hdf5(
            config_file=scenario_file,
            prototype_surface=prototype_surface,
            prototype_initial_orientation=prototype_initial_orientation,
            prototype_kinematic_deviations_rigid_body=prototype_kinematic_deviations_rigid_body,
            prototype_actuators=prototype_actuators,
            device=device,
        )

        return cls(
            power_plant_position=power_plant_position,
            target_areas=target_areas,
            light_sources=light_sources,
            heliostat_field=heliostat_field,
        )

    def __repr__(self) -> str:
        """Return a string representation of the scenario."""
        return (
            f"ARTIST Scenario containing:\n\tA Power Plant located at: {self.power_plant_position.tolist()}"
            f" with {len(self.target_areas.target_area_list)} Target Area(s),"
            f" {len(self.light_sources.light_source_list)} Light Source(s),"
            f" and {self.heliostat_field.all_surface_points.shape[0]} Heliostat(s)."
        )
