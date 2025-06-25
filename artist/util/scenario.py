import logging
from collections import defaultdict
from typing import Optional

import h5py
import torch
from typing_extensions import Self

from artist.field.heliostat_field import HeliostatField
from artist.field.heliostat_group import HeliostatGroup
from artist.field.tower_target_areas import TowerTargetAreas
from artist.scene.light_source_array import LightSourceArray
from artist.util import config_dictionary, utils_load_h5
from artist.util.environment_setup import get_device

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
        (Note: Currently only a single light source can be provided.)

        Parameters
        ----------
        power_plant_position : torch.Tensor,
            The position of the power plant as latitude, longitude, altitude.
        target_areas : TargetAreaArray
            A list of tower target areas included in the scenario.
        light_sources : LightSourceArray
            A list of light sources included in the scenario.
            Currently only a single light source can be provided.
        heliostat_field : HeliostatField
            A field of heliostats included in the scenario.
        """
        self.power_plant_position = power_plant_position
        self.target_areas = target_areas
        self.light_sources = light_sources
        self.heliostat_field = heliostat_field

    @classmethod
    def load_scenario_from_hdf5(
        cls, scenario_file: h5py.File, device: Optional[torch.device] = None
    ) -> Self:
        """
        Class method to load the scenario from an HDF5 file.

        Parameters
        ----------
        scenario_file : h5py.File
            The config file containing all the information about the scenario being loaded.
        device : Optional[torch.device]
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA, MPS, or CPU) based on availability and OS.

        Returns
        -------
        Scenario
            The ``ARTIST`` scenario loaded from the HDF5 file.
        """
        device = get_device(device=device)

        if torch.distributed.get_rank() == 0:
            log.info(
                f"Loading an ``ARTIST`` scenario HDF5 file. This scenario file is version {scenario_file.attrs['version']}."
            )

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
        heliostat_group: HeliostatGroup,
        string_mapping: Optional[list[tuple[str, str, torch.Tensor]]] = None,
        single_incident_ray_direction: torch.Tensor = torch.tensor(
            [0.0, 1.0, 0.0, 0.0]
        ),
        single_target_area_index: int = 0,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create an index mapping from heliostat names, target area names and incident ray directions.

        If no mapping is provided, a default mapping for all heliostats within this group will be created.
        The default mapping will map all heliostats to the default ``single_incident_ray_direction``, which
        simualtes a light source positioned in the south and the default ``single_target_area_index``, which
        is 0. To overwrite these defaults, please provide a ``single_incident_ray_direction`` or a
        ``single_target_area_index``.

        Parameters
        ----------
        heliostat_group : HeliostatGroup
            The current heliostat group.
        string_mapping : Optional[list[tuple[str, str, torch.Tensor]]]
            Strings that map heliostats to target areas and incident ray direction tensors (default is None).
        single_incident_ray_direction : torch.Tensor
            The default incident ray direction (defualt is torch.tensor([0.0, 1.0, 0.0, 0.0])).
        single_target_area_index : int
            The default target area index (default is 0).
        device : Optional[torch.device]
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA, MPS, or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            All incident ray directions for the heliostats in this scenario.
        torch.Tensor
            The indices of all active heliostats in order.
        torch.Tensor
            The indices of target areas for all heliostats in order.
        """
        device = get_device(device=device)

        data_per_heliostat = defaultdict(list)

        if string_mapping is None:
            if (
                single_incident_ray_direction.shape != torch.Size([4])
                or single_incident_ray_direction[3] != 0.0
                or torch.norm(single_incident_ray_direction) != 1.0
            ):
                raise ValueError(
                    "The specified single incident ray direction is invalid. Please provide a normalized 4D tensor with last element 0.0."
                )
            if single_target_area_index >= self.target_areas.number_of_target_areas:
                raise ValueError(
                    f"The specified single target area index is invalid. Only {self.target_areas.number_of_target_areas} target areas exist in this scenario."
                )
            active_heliostats_mask = torch.ones(
                heliostat_group.number_of_heliostats, dtype=torch.int32, device=device
            )
            target_area_mask = torch.tensor(
                single_target_area_index, dtype=torch.int32, device=device
            ).expand(heliostat_group.number_of_heliostats)
            incident_ray_directions = single_incident_ray_direction.expand(
                heliostat_group.number_of_heliostats, -1
            ).to(device)
        else:
            filtered_mapping = [
                mapping
                for mapping in string_mapping
                if mapping[0] in heliostat_group.names
            ]
            errors = []
            for i, (heliostat_name, target_name, light_direction) in enumerate(
                filtered_mapping
            ):
                if target_name not in self.target_areas.names:
                    errors.append(
                        f"Invalid target '{target_name}' (Found at index {i} of provided mapping) not found in this scenario."
                    )
                if (
                    light_direction.shape != torch.Size([4])
                    or light_direction[3] != 0.0
                    or torch.norm(light_direction) != 1.0
                ):
                    errors.append(
                        f"Invalid incident ray direction (Found at index {i} of provided mapping). This must be a normalized 4D tensor with last element 0.0."
                    )
            if errors:
                raise ValueError(" ".join(errors))

            heliostat_name_to_index = {
                heliostat_name: index
                for index, heliostat_name in enumerate(heliostat_group.names)
            }
            active_heliostats_mask = torch.zeros(
                heliostat_group.number_of_heliostats, dtype=torch.int32, device=device
            )
            target_area_mask = torch.empty(
                len(filtered_mapping), dtype=torch.int32, device=device
            )
            incident_ray_directions = torch.empty(
                (len(filtered_mapping), 4), device=device
            )
            heliostat_name_to_index = {
                heliostat_name: index
                for index, heliostat_name in enumerate(heliostat_group.names)
            }
            active_heliostats_mask = torch.zeros(
                heliostat_group.number_of_heliostats, dtype=torch.int32, device=device
            )
            target_area_mask = torch.empty(
                len(filtered_mapping), dtype=torch.int32, device=device
            )
            incident_ray_directions = torch.empty(
                (len(filtered_mapping), 4), device=device
            )

            for i, (heliostat_name, target_name, incident_ray_direction) in enumerate(
                filtered_mapping
            ):
                if heliostat_name in heliostat_group.names:
                    active_heliostats_mask[heliostat_name_to_index[heliostat_name]] += 1
                    data_per_heliostat[heliostat_name].append(
                        [
                            self.target_areas.names.index(target_name),
                            incident_ray_direction,
                        ]
                    )
            index = 0
            for name in heliostat_group.names:
                for target_area_index, incident_ray_direction in data_per_heliostat.get(
                    name, []
                ):
                    target_area_mask[index] = target_area_index
                    incident_ray_directions[index] = incident_ray_direction
                    index += 1

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
