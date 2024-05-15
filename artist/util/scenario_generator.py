import logging
import sys
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Dict, Optional

import colorlog
import h5py

from artist.util import config_dictionary
from artist.util.configuration_classes import (
    HeliostatListConfig,
    LightSourceListConfig,
    PrototypeConfig,
    ReceiverListConfig,
)


class ScenarioGenerator:
    """
    Generate an ARTIST scenario, saving it as an HDF5 file.

    Attributes
    ----------
    file_path : Path
        File path to the HDF5 to be saved.
    receiver_list_config : ReceiverListConfig
        The receiver list configuration object.
    light_source_list_config : LightSourceListConfig
        The light source list configuration object.
    heliostat_list_config : HeliostatListConfig
        The heliostat_list configuration object.
    prototype_config : PrototypeConfig
        The prototype configuration object,
    version : Optional[float]
        The version of the scenario generator being used.
    log : logging.Logger
        The logger.

    Methods
    -------
    flatten_dict()
        Flatten nested dictionaries to first-level keys.
    include_parameters()
        Include the parameters from a parameter dictionary.
    generate_scenario()
        Generate the scenario according to the given parameters.
    """

    def __init__(
        self,
        file_path: str,
        receiver_list_config: ReceiverListConfig,
        light_source_list_config: LightSourceListConfig,
        heliostat_list_config: HeliostatListConfig,
        prototype_config: PrototypeConfig,
        version: Optional[float] = 1.0,
        log_level: Optional[int] = logging.INFO,
    ) -> None:
        """
        Initialize the scenario generator.

        Parameters
        ----------
        file_path : str
            File path to the HDF5 to be saved.
        receiver_list_config : ReceiverListConfig
            The receiver list configuration object.
        light_source_list_config : LightSourceListConfig
            The light source list configuration object.
        heliostat_list_config : HeliostatListConfig
            The heliostat_list configuration object.
        prototype_config : PrototypeConfig
            The prototype configuration object,
        version : Optional[float]
            The version of the scenario generator being used.
        log_level : Optional[int]
            The log level applied to the logger.
        """
        self.file_path = Path(file_path)
        if not self.file_path.parent.is_dir():
            raise FileNotFoundError(
                f"The folder ``{self.file_path.parent}`` selected to save the scenario does not exist. "
                "Please create the folder or adjust the file path before running again!"
            )
        self.receiver_list_config = receiver_list_config
        self.light_source_list_config = light_source_list_config
        self.heliostat_list_config = heliostat_list_config
        self.prototype_config = prototype_config
        self.check_facet_and_point_size()
        self.version = version
        log = logging.getLogger("scenario-generator")  # Get logger instance.
        log_formatter = colorlog.ColoredFormatter(
            fmt="[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s]"
            "[%(log_color)s%(levelname)s%(reset)s] - %(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
            secondary_log_colors={},
        )
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(log_formatter)
        log.addHandler(handler)
        log.setLevel(log_level)
        self.log = log

    def check_facet_and_point_size(self):
        """
        Check that each heliostat has the same number of facets and each facet the same number of evaluation points.

        Raises
        ------
        AssertionError
            If at least one heliostat has a different number of facets or one facet has a different number of evaluation
            points.
        """
        # Define accepted number of facets based on the prototype
        accepted_number_of_facets = len(
            self.prototype_config.surface_prototype.facets_list
        )
        # Define accepted number of points based on the prototype
        accepted_number_of_points = (
            self.prototype_config.surface_prototype.facets_list[0].number_eval_points_e
            * self.prototype_config.surface_prototype.facets_list[
                0
            ].number_eval_points_n
        )
        # Check that every facet in the prototype has the same number of evaluation points
        assert all(
            self.prototype_config.surface_prototype.facets_list[i].number_eval_points_e
            * self.prototype_config.surface_prototype.facets_list[
                i
            ].number_eval_points_n
            == accepted_number_of_points
            for i in range(accepted_number_of_facets)
        ), "The number of evaluation points for each facet is different in the surface prototype!"
        # Check that every heliostat has the same number of facets and evaluation points
        for heliostat in self.heliostat_list_config.heliostat_list:
            if heliostat.heliostat_surface:
                assert (
                    len(heliostat.heliostat_surface.facets_list)
                    == accepted_number_of_facets
                ), "Individual heliostats must all have the same number of facets!"
                assert all(
                    heliostat.heliostat_surface.facets_list[i].number_eval_points_e
                    * heliostat.heliostat_surface.facets_list[i].number_eval_points_n
                    == accepted_number_of_points
                    for i in range(accepted_number_of_facets)
                ), "The number of evaluation points for each facet is different in the individual heliostat!"

    def flatten_dict(
        self, dictionary: MutableMapping, parent_key: str = "", sep: str = "/"
    ) -> Dict[str, Any]:
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
        Dict
            A flattened version of the original dictionary.
        """
        return dict(self._flatten_dict_gen(dictionary, parent_key, sep))

    def _flatten_dict_gen(self, d: MutableMapping, parent_key: str, sep: str) -> None:
        # Flattens the keys in a nested dictionary so that the resulting key is a concatenation of all nested keys
        # separated by a defined separator.
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, MutableMapping):
                yield from self.flatten_dict(v, new_key, sep=sep).items()
            else:
                yield new_key, v

    @staticmethod
    def include_parameters(file: h5py.File, prefix: str, parameters: Dict) -> None:
        """
        Include the parameters from a parameter dictionary.

        Parameters
        ----------
        file : h5py.File
            The HDF5 file to write to.
        prefix : str
            The prefix used for naming the parameters.
        parameters : Dict
            The parameters to be included into the HFD5 file.
        """
        for key, value in parameters.items():
            file[f"{prefix}/{key}"] = value

    def generate_scenario(self) -> None:
        """Generate the scenario according to the given parameters."""
        self.log.info(f"Generating a scenario saved to: {self.file_path}")
        save_name = self.file_path.parent / (self.file_path.name + ".h5")
        with h5py.File(save_name, "w") as f:
            # Set scenario version as attribute.
            self.log.info(f"Using scenario generator version {self.version}")
            f.attrs["version"] = self.version

            # Include parameters for the receivers.
            self.log.info("Including parameters for the receivers")
            self.include_parameters(
                file=f,
                prefix=config_dictionary.receiver_key,
                parameters=self.flatten_dict(
                    self.receiver_list_config.create_receiver_list_dict()
                ),
            )

            # Include parameters for the light sources.
            self.log.info("Including parameters for the light sources")
            self.include_parameters(
                file=f,
                prefix=config_dictionary.light_source_key,
                parameters=self.flatten_dict(
                    self.light_source_list_config.create_light_source_list_dict()
                ),
            )

            # Include parameters for the prototype.
            self.log.info("Including parameters for the prototype")
            self.include_parameters(
                file=f,
                prefix=config_dictionary.prototype_key,
                parameters=self.flatten_dict(
                    self.prototype_config.create_prototype_dict()
                ),
            )

            # Include heliostat parameters.
            self.log.info("Including parameters for the heliostats")
            self.include_parameters(
                file=f,
                prefix=config_dictionary.heliostat_key,
                parameters=self.flatten_dict(
                    self.heliostat_list_config.create_heliostat_list_dict()
                ),
            )
