import logging
from typing import Optional, Union

import h5py
import torch
from typing_extensions import Self

from artist.field.heliostat_field import HeliostatField
from artist.field.heliostat_group import HeliostatGroup
from artist.field.tower_target_areas import TowerTargetAreas
from artist.scene.light_source_array import LightSourceArray
from artist.util import config_dictionary, utils_load_h5

log = logging.getLogger(__name__)
"""A logger for the scenario."""


class Scenario:
    """
    Define a scenario loaded by ARTIST.

    Attributes
    ----------
    power_plant_position : torch.Tensor
        The position of the power plant as latitude, longitude, altitude.
    target_areas : TowerTargetAreas
        All target areas on all towers of the power plant.
    light_sources : LightSourceArray
        A list of light sources included in the scenario.
    heliostat_field : HeliostatField
        The heliostat field for the scenario.

    Methods
    -------
    load_scenario_from_hdf5()
        Class method to load the scenario from an HDF5 file.
    index_mapping()
        Create an index mapping from heliostat names, target area names and incident ray directions.
    """

    def __init__(
        self,
        power_plant_position: torch.Tensor,
        target_areas: TowerTargetAreas,
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
        target_areas = TowerTargetAreas.from_hdf5(
            config_file=scenario_file, device=device
        )
        light_sources = LightSourceArray.from_hdf5(
            config_file=scenario_file, device=device
        )

        prototype_surface = utils_load_h5.surface_config(
            prototype=True, scenario_file=scenario_file, device=device
        )

        prototype_initial_orientation = torch.tensor(
            scenario_file[config_dictionary.prototype_key][
                config_dictionary.kinematic_prototype_key
            ][config_dictionary.kinematic_initial_orientation][()],
            dtype=torch.float,
            device=device,
        )

        prototype_kinematic_type = scenario_file[config_dictionary.prototype_key][
            config_dictionary.kinematic_prototype_key
        ][config_dictionary.kinematic_type][()].decode("utf-8")

        prototype_kinematic_deviations, number_of_actuators = (
            utils_load_h5.kinematic_deviations(
                prototype=True,
                kinematic_type=prototype_kinematic_type,
                scenario_file=scenario_file,
                log=log,
                device=device,
            )
        )

        prototype_kinematic = {
            config_dictionary.kinematic_type: prototype_kinematic_type,
            config_dictionary.kinematic_initial_orientation: prototype_initial_orientation,
            config_dictionary.kinematic_deviations: prototype_kinematic_deviations,
        }

        prototype_actuator_keys = list(
            scenario_file[config_dictionary.prototype_key][
                config_dictionary.actuators_prototype_key
            ].keys()
        )

        prototype_actuator_type = scenario_file[config_dictionary.prototype_key][
            config_dictionary.actuators_prototype_key
        ][prototype_actuator_keys[0]][config_dictionary.actuator_type_key][()].decode(
            "utf-8"
        )

        prototype_actuator_parameters = utils_load_h5.actuator_parameters(
            prototype=True,
            scenario_file=scenario_file,
            actuator_type=prototype_actuator_type,
            number_of_actuators=number_of_actuators,
            initial_orientation=prototype_initial_orientation,
            log=log,
            device=device,
        )

        prototype_actuators = {
            config_dictionary.actuator_type_key: prototype_actuator_type,
            config_dictionary.actuator_parameters_key: prototype_actuator_parameters,
        }

        heliostat_field = HeliostatField.from_hdf5(
            config_file=scenario_file,
            prototype_surface=prototype_surface,
            prototype_kinematic=prototype_kinematic,
            prototype_actuators=prototype_actuators,
            device=device,
        )

        return cls(
            power_plant_position=power_plant_position,
            target_areas=target_areas,
            light_sources=light_sources,
            heliostat_field=heliostat_field,
        )

    def index_mapping(
        self,
        string_mapping: Optional[list[tuple[str, str, torch.Tensor]]],
        heliostat_group: HeliostatGroup,
        default_incident_ray_direction: torch.Tensor = torch.tensor(
            [0.0, 1.0, 0.0, 0.0]
        ),
        default_target_area_index: int = 0,
        device: Union[torch.device, str] = "cuda",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create an index mapping from heliostat names, target area names and incident ray directions.

        Parameters
        ----------
        string_mapping : (Optional[list[tuple[str, str, torch.Tensor]]])
            Strings that map heliostats to target areas and incident ray direction tensors.
        heliostat_group_index : int
            The index of the current heliostat group.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        torch.Tensor
            All incident ray directions for the heliostats in this scenario.
        torch.Tensor
            The indices of all active heliostats in order.
        torch.Tensor
            The indices of target areas for all heliostats in order.
        """
        device = torch.device(device)

        active_heliostats_mask = torch.zeros(
            heliostat_group.number_of_heliostats, dtype=torch.int32, device=device
        )
        target_area_mask = torch.zeros(
            len(string_mapping), dtype=torch.int32, device=device
        )
        incident_ray_directions = torch.zeros((len(string_mapping), 4), device=device)

        heliostat_to_target = {
            heliostat: (target, light_direction)
            for heliostat, target, light_direction in string_mapping
        }

        active_index = 0
        for heliostat_index, name in enumerate(heliostat_group.names):
            if name in heliostat_to_target:
                target_name, direction = heliostat_to_target[name]
                active_heliostats_mask[heliostat_index] = 1
                target_area_mask[active_index] = self.target_areas.names.index(
                    target_name
                )
                incident_ray_directions[active_index] = direction
                active_index += 1
            else:
                active_heliostats_mask[heliostat_index] = 0

        return (
            active_heliostats_mask,
            target_area_mask,
            incident_ray_directions,
        )

    def __repr__(self) -> str:
        """Return a string representation of the scenario."""
        return (
            f"ARTIST Scenario containing:\n\tA Power Plant located at: {self.power_plant_position.tolist()}"
            f" with {len(self.target_areas.names)} Target Area(s),"
            f" {len(self.light_sources.light_source_list)} Light Source(s),"
            f" and {sum(len(group.names) for group in self.heliostat_field.heliostat_groups)} Heliostat(s)."
        )
