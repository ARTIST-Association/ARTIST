import logging
from typing import Optional, Union

import h5py
import torch
from typing_extensions import Self

from artist.field.heliostat_field import HeliostatField
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
    target_areas : TargetAreaArray
        A list of tower target areas included in the scenario.
    light_sources : LightSourceArray
        A list of light sources included in the scenario.
    heliostat_field : HeliostatField
        The heliostat field for the scenario.

    Methods
    -------
    load_scenario_from_hdf5()
        Class method to load the scenario from an HDF5 file.
    get_target_area()
        Retrieve a specified target area from the scenario.
    create_calibration_scenario()
        Create a calibration scenario with a single heliostat from an existing scenario.
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

        prototype_actuators = utils_load_h5.actuator_parameters(
            prototype=True,
            scenario_file=scenario_file,
            actuator_type=prototype_actuator_type,
            number_of_actuators=number_of_actuators,
            initial_orientation=prototype_initial_orientation,
            log=log,
            device=device,
        )

        heliostat_field = HeliostatField.from_hdf5(
            config_file=scenario_file,
            prototype_surface=prototype_surface,
            prototype_initial_orientation=prototype_initial_orientation,
            prototype_kinematic_deviations=prototype_kinematic_deviations,
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
        mapping: Optional[list[tuple[str, str, torch.Tensor]]],
        heliostat_group_index: int,
        device: Union[torch.device, str] = "cuda"
    ) -> None:
        device = torch.device(device)

        all_heliostat_name_indices = {heliostat_name: index for index, heliostat_name in enumerate(self.heliostat_field.heliostat_groups[heliostat_group_index].names)}
        all_target_area_indices = {target_area_name: index for index, target_area_name in enumerate(self.target_areas.names)}

        single_target_area_index = 1
        single_incident_ray_direction_index = 0

        if mapping is None:
            all_incident_ray_directions = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)
            heliostat_target_mapping = torch.tensor([
                [i, single_target_area_index, single_incident_ray_direction_index]
                for i in range(self.heliostat_field.heliostat_groups[heliostat_group_index].number_of_heliostats)
            ], device=device)
        else:
            all_incident_ray_directions = torch.stack([t for _, _, t in mapping])
            heliostat_target_mapping = torch.tensor([
                [all_heliostat_name_indices[heliostat_name], all_target_area_indices[target_area_name], index]
                for index, (heliostat_name, target_area_name, _) in enumerate(mapping)
                if heliostat_name in all_heliostat_name_indices and target_area_name in all_target_area_indices
            ], device=device)
        
        active_heliostats_indices = heliostat_target_mapping[:, 0]
        target_area_indices = heliostat_target_mapping[:, 1]
        incident_ray_direction_indices = heliostat_target_mapping[:, 2]

        return all_incident_ray_directions, incident_ray_direction_indices, active_heliostats_indices, target_area_indices
        

    def create_calibration_scenario(
        self,
        heliostat_index: int,
        device: Union[torch.device, str] = "cuda",
    ) -> "Scenario":
        """
        Create a calibration scenario with a single heliostat from an existing scenario.

        Parameters
        ----------
        heliostat_index : int
            The index of the heliostat from the original scenario.
        device : device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        Scenario
            The calibration scenario.
        """
        device = torch.device(device)

        heliostat_index = torch.tensor([heliostat_index], device=device)

        heliostat_field = HeliostatField(
            number_of_heliostats=1,
            all_heliostat_names=[
                self.heliostat_field.all_heliostat_names[heliostat_index]
            ],
            all_heliostat_positions=self.heliostat_field.all_heliostat_positions[
                heliostat_index
            ],
            all_aim_points=self.heliostat_field.all_aim_points[heliostat_index],
            all_surface_points=self.heliostat_field.all_surface_points[heliostat_index],
            all_surface_normals=self.heliostat_field.all_surface_normals[
                heliostat_index
            ],
            all_initial_orientations=self.heliostat_field.all_initial_orientations[
                heliostat_index
            ],
            all_kinematic_deviation_parameters=self.heliostat_field.all_kinematic_deviation_parameters[
                heliostat_index
            ],
            all_actuator_parameters=self.heliostat_field.all_actuator_parameters[
                heliostat_index
            ],
            device=device,
        )

        return Scenario(
            power_plant_position=self.power_plant_position,
            target_areas=self.target_areas,
            light_sources=self.light_sources,
            heliostat_field=heliostat_field,
        )

    def __repr__(self) -> str:
        """Return a string representation of the scenario."""
        return (
            f"ARTIST Scenario containing:\n\tA Power Plant located at: {self.power_plant_position.tolist()}"
            f" with {len(self.target_areas.target_area_list)} Target Area(s),"
            f" {len(self.light_sources.light_source_list)} Light Source(s),"
            f" and {self.heliostat_field.number_of_heliostats} Heliostat(s)."
        )
