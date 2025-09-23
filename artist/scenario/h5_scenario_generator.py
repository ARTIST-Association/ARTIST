import logging
import pathlib
from collections.abc import MutableMapping
from typing import Any, Generator

import h5py
import torch

from artist.scenario.configuration_classes import (
    ActuatorConfig,
    ActuatorListConfig,
    HeliostatListConfig,
    KinematicConfig,
    LightSourceListConfig,
    PowerPlantConfig,
    PrototypeConfig,
    TargetAreaListConfig,
)
from artist.util import config_dictionary

log = logging.getLogger(__name__)
"""A logger for the scenario generator."""


class H5ScenarioGenerator:
    """
    Generate an ``ARTIST`` scenario, saving it as an HDF5 file.

    Attributes
    ----------
    file_path : pathlib.Path
        File path to the HDF5 to be saved.
    power_plant_config : PowerPlantConfig
        The power plant configuration object.
    target_area_list_config : TargetAreaListConfig
        The target area list configuration object.
    light_source_list_config : LightSourceListConfig
        The light source list configuration object.
    heliostat_list_config : HeliostatListConfig
        The heliostat_list configuration object.
    prototype_config : PrototypeConfig
        The prototype configuration object,
    version : Optional[float]
        The version of the scenario generator being used.

    Methods
    -------
    generate_scenario()
        Generate the scenario and save it as an HDF5 file.
    """

    def __init__(
        self,
        file_path: pathlib.Path,
        power_plant_config: PowerPlantConfig,
        target_area_list_config: TargetAreaListConfig,
        light_source_list_config: LightSourceListConfig,
        heliostat_list_config: HeliostatListConfig,
        prototype_config: PrototypeConfig,
        version: float = 1.0,
    ) -> None:
        """
        Initialize the scenario generator.

        Scenarios in ``ARTIST`` describe the whole environment and all the components of a solar tower power
        plant. The scenario generator creates the scenarios. A scenario encompasses the tower target area(s), the
        light source(s), prototypes, and the heliostat(s). The generated scenarios are then saved in HDF5
        files.

        Parameters
        ----------
        file_path : pathlib.Path
            File path to the HDF5 to be saved.
        power_plant_config : PowerPlantConfig
            The power plant configuration object.
        target_area_list_config : TargetAreaListConfig
            The target area list configuration object.
        light_source_list_config : LightSourceListConfig
            The light source list configuration object.
        heliostat_list_config : HeliostatListConfig
            The heliostat_list configuration object.
        prototype_config : PrototypeConfig
            The prototype configuration object.
        version : float
            The version of the scenario generator being used (default is 1.0).
        """
        self.file_path = file_path
        if not self.file_path.parent.is_dir():
            raise FileNotFoundError(
                f"The folder ``{self.file_path.parent}`` selected to save the scenario does not exist. "
                "Please create the folder or adjust the file path before running again!"
            )
        self.power_plant_config = power_plant_config
        self.target_area_list_config = target_area_list_config
        self.light_source_list_config = light_source_list_config
        self.heliostat_list_config = heliostat_list_config
        self.prototype_config = prototype_config
        self._check_equal_facet_numbers()
        self.version = version

    def _get_number_of_heliostat_groups(self) -> int:
        """
        Get the number of heliostat groups in the scenario.

        Returns
        -------
        int
            Number of heliostat groups in the scenario.
        """
        unique_groups = set()
        for heliostat_config in self.heliostat_list_config.heliostat_list:
            if isinstance(heliostat_config.kinematic, KinematicConfig):
                selected_kinematic_type = heliostat_config.kinematic.type
            else:
                selected_kinematic_type = self.prototype_config.kinematic_prototype.type
            if isinstance(heliostat_config.actuators, ActuatorListConfig):
                for actuator_config in heliostat_config.actuators.actuator_list:
                    assert isinstance(actuator_config, ActuatorConfig)
                    selected_actuator_type = actuator_config.type
                    unique_groups.add((selected_kinematic_type, selected_actuator_type))
            else:
                for (
                    actuator_config
                ) in self.prototype_config.actuators_prototype.actuator_list:
                    assert isinstance(actuator_config, ActuatorConfig)
                    selected_actuator_type = actuator_config.type
                    unique_groups.add((selected_kinematic_type, selected_actuator_type))
        return len(unique_groups)

    def _check_equal_facet_numbers(self):
        """
        Check that each heliostat has the same number of facets.

        Raises
        ------
        ValueError
            If at least one heliostat has a different number of facets.
        """
        # Define accepted number of facets based on the prototype.
        accepted_number_of_facets = len(
            self.prototype_config.surface_prototype.facet_list
        )

        # Check that every heliostat has the same number of facets.
        for heliostat in self.heliostat_list_config.heliostat_list:
            if heliostat.surface:
                if len(heliostat.surface.facet_list) != accepted_number_of_facets:
                    raise ValueError(
                        "Individual heliostats must all have the same number of facets!"
                    )

    def _flatten_dict(
        self, dictionary: MutableMapping, parent_key: str = "", sep: str = "/"
    ) -> dict[str, Any]:
        """
        Flatten nested dictionaries to first-level keys.

        Parameters
        ----------
        dictionary : MutableMapping
            Original nested dictionary to flatten.
        parent_key : str
            The parent key of nested dictionaries. Should be empty upon initialization.
        sep : str
            The separator used to separate keys in nested dictionaries.

        Returns
        -------
        dict[str, Any]
            A flattened version of the original dictionary.
        """
        return dict(self._flatten_dict_gen(dictionary, parent_key, sep))

    def _flatten_dict_gen(
        self, d: MutableMapping, parent_key: str, sep: str
    ) -> Generator:
        # Flattens the keys in a nested dictionary so that the resulting key is a concatenation of all nested keys
        # separated by a defined separator.
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, MutableMapping):
                yield from self._flatten_dict(v, new_key, sep=sep).items()
            else:
                yield new_key, v

    @staticmethod
    def _include_parameters(file: h5py.File, prefix: str, parameters: dict) -> None:
        """
        Include the parameters from a parameter dictionary.

        Parameters
        ----------
        file : h5py.File
            The HDF5 file to write to.
        prefix : str
            The prefix used for naming the parameters.
        parameters : dict
            The parameters to be included into the HFD5 file.
        """
        for key, value in parameters.items():
            if torch.is_tensor(value):
                value = value.cpu()
            file[f"{prefix}/{key}"] = value

    def generate_scenario(self) -> None:
        """Generate the scenario and save it as an HDF5 file."""
        log.info(f"Generating a scenario saved to: {self.file_path}.")
        if self.file_path.suffix == ".h5":
            save_name = self.file_path
        elif self.file_path.suffix == "":
            save_name = self.file_path.with_suffix(".h5")
        else:
            log.error(
                f"```ARTIST``` only supports HDF5 files in the scenario generator, your extension {self.file_path.suffix} is unsupported!"
            )
        with h5py.File(save_name, "w") as f:
            # Set scenario version as attribute.
            log.info(f"Using scenario generator version {self.version}.")
            f.attrs["version"] = self.version

            # Include number of heliostat groups in the top level
            f[config_dictionary.number_of_heliostat_groups] = (
                self._get_number_of_heliostat_groups()
            )

            # Include parameters for the power plant.
            log.info("Including parameters for the power plant.")
            self._include_parameters(
                file=f,
                prefix=config_dictionary.power_plant_key,
                parameters=self._flatten_dict(
                    self.power_plant_config.create_power_plant_dict()
                ),
            )

            # Include parameters for the tower target areas.
            log.info("Including parameters for the target areas.")
            self._include_parameters(
                file=f,
                prefix=config_dictionary.target_area_key,
                parameters=self._flatten_dict(
                    self.target_area_list_config.create_target_area_list_dict()
                ),
            )

            # Include parameters for the light sources.
            log.info("Including parameters for the light sources.")
            self._include_parameters(
                file=f,
                prefix=config_dictionary.light_source_key,
                parameters=self._flatten_dict(
                    self.light_source_list_config.create_light_source_list_dict()
                ),
            )

            # Include parameters for the prototype.
            log.info("Including parameters for the prototype.")
            self._include_parameters(
                file=f,
                prefix=config_dictionary.prototype_key,
                parameters=self._flatten_dict(
                    self.prototype_config.create_prototype_dict()
                ),
            )

            # Include heliostat parameters.
            log.info("Including parameters for the heliostats.")
            self._include_parameters(
                file=f,
                prefix=config_dictionary.heliostat_key,
                parameters=self._flatten_dict(
                    self.heliostat_list_config.create_heliostat_list_dict()
                ),
            )
