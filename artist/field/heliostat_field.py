import logging
from collections import defaultdict
from typing import Any

import h5py
import torch.nn
from typing_extensions import Self

from artist.field.heliostat_group import HeliostatGroup
from artist.field.surface import Surface
from artist.util import config_dictionary, type_mappings, utils_load_h5
from artist.util.configuration_classes import (
    SurfaceConfig,
)
from artist.util.environment_setup import get_device

log = logging.getLogger(__name__)
"""A logger for the heliostat field."""


class HeliostatField(torch.nn.Module):
    """
    The heliostat field.

    A heliostat field consists of one or multiple heliostat groups. Each heliostat group contains all
    heliostats with a specific kinematic type and actuator type. The heliostats in the field are aligned
    individually to reflect the incoming light in a way that ensures maximum efficiency for the whole power plant.

    Attributes
    ----------
    heliostat_groups : list[HeliostatGroup]
        A list containing all heliostat groups.
    number_of_heliostat_groups : int
        The number of different heliostat groups in the heliostat field.

    Methods
    -------
    from_hdf5()
        Load a heliostat field from an HDF5 file.
    forward()
        Specify the forward pass.
    """

    def __init__(
        self,
        heliostat_groups: list[HeliostatGroup],
    ) -> None:
        """
        Initialize the heliostat field with heliostat groups.

        Parameters
        ----------
        heliostat_groups : list[HeliostatGroup]
            A list containing all heliostat groups.
        """
        super().__init__()

        self.heliostat_groups = heliostat_groups
        self.number_of_heliostat_groups = len(self.heliostat_groups)

    @classmethod
    def from_hdf5(
        cls,
        config_file: h5py.File,
        prototype_surface: SurfaceConfig,
        prototype_kinematic: dict[str, str | torch.Tensor],
        prototype_actuators: dict[str, str | torch.Tensor],
        device: torch.device | None = None,
    ) -> Self:
        """
        Load a heliostat field from an HDF5 file.

        Parameters
        ----------
        config_file : h5py.File
            The HDF5 file containing the configuration to be loaded.
        prototype_surface : SurfaceConfig
            The prototype for the surface configuration to be used if a heliostat has no individual surface.
        prototype_kinematic : dict[str, str | torch.Tensor]
            The prototype for the kinematic, including type, initial orientation and deviations.
        prototype_actuators : dict[str, str | torch.Tensor]
            The prototype for the actuators, including type and parameters.
        device : device: torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA, MPS, or CPU) based on availability and OS.

        Raises
        ------
        ValueError
            If neither prototypes nor individual heliostat parameters are provided.

        Returns
        -------
        HeliostatField
            The heliostat field loaded from the HDF5 file.
        """
        device = get_device(device=device)

        log.info("Loading a heliostat field from an HDF5 file.")

        grouped_field_data: defaultdict[str, defaultdict[str, list[Any]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for heliostat_name in config_file[config_dictionary.heliostat_key].keys():
            single_heliostat_config = config_file[config_dictionary.heliostat_key][
                heliostat_name
            ]

            if (
                config_dictionary.heliostat_surface_key
                in single_heliostat_config.keys()
            ):
                surface_config = utils_load_h5.surface_config(
                    prototype=False,
                    scenario_file=single_heliostat_config,
                    device=device,
                )
            else:
                if prototype_surface is None:
                    raise ValueError(
                        "If the heliostat does not have individual surface parameters, a surface prototype must be provided!"
                    )
                log.info(
                    "Individual surface parameters not provided - loading a heliostat with the surface prototype."
                )
                surface_config = prototype_surface

            if (
                config_dictionary.heliostat_kinematic_key
                in single_heliostat_config.keys()
            ):
                initial_orientation = torch.tensor(
                    single_heliostat_config[config_dictionary.heliostat_kinematic_key][
                        config_dictionary.kinematic_initial_orientation
                    ][()],
                    dtype=torch.float,
                    device=device,
                )
                kinematic_type = single_heliostat_config[
                    config_dictionary.heliostat_kinematic_key
                ][config_dictionary.kinematic_type][()].decode("utf-8")

                kinematic_deviations, number_of_actuators = (
                    utils_load_h5.kinematic_deviations(
                        prototype=False,
                        kinematic_type=kinematic_type,
                        scenario_file=single_heliostat_config,
                        log=log,
                        heliostat_name=heliostat_name,
                        device=device,
                    )
                )
            else:
                if prototype_kinematic is None:
                    raise ValueError(
                        "If the heliostat does not have an individual kinematic, a kinematic prototype must be provided!"
                    )
                log.info(
                    "Individual kinematic configuration not provided - loading a heliostat with the kinematic prototype."
                )
                kinematic_type = prototype_kinematic[config_dictionary.kinematic_type]
                initial_orientation = prototype_kinematic[
                    config_dictionary.kinematic_initial_orientation
                ]
                kinematic_deviations = prototype_kinematic[
                    config_dictionary.kinematic_deviations
                ]

            if (
                config_dictionary.heliostat_actuator_key
                in single_heliostat_config.keys()
            ):
                actuator_keys = list(
                    single_heliostat_config[
                        config_dictionary.heliostat_actuator_key
                    ].keys()
                )

                actuator_type = single_heliostat_config[
                    config_dictionary.heliostat_actuator_key
                ][actuator_keys[0]][config_dictionary.actuator_type_key][()].decode(
                    "utf-8"
                )

                actuator_parameters = utils_load_h5.actuator_parameters(
                    prototype=False,
                    scenario_file=single_heliostat_config,
                    actuator_type=actuator_type,
                    number_of_actuators=number_of_actuators,
                    initial_orientation=initial_orientation,
                    log=log,
                    heliostat_name=heliostat_name,
                    device=device,
                )
            else:
                if prototype_actuators is None:
                    raise ValueError(
                        "If the heliostat does not have individual actuators, an actuator prototype must be provided!"
                    )
                log.info(
                    "Individual actuator configurations not provided - loading a heliostat with the actuator prototype."
                )
                actuator_type = prototype_actuators[config_dictionary.actuator_type_key]
                actuator_parameters = prototype_actuators[
                    config_dictionary.actuator_parameters_key
                ]

            surface = Surface(surface_config)

            heliostat_group_key = f"{kinematic_type}_{actuator_type}"

            grouped_field_data[heliostat_group_key][config_dictionary.names].append(
                heliostat_name
            )
            grouped_field_data[heliostat_group_key][config_dictionary.positions].append(
                torch.tensor(
                    single_heliostat_config[config_dictionary.heliostat_position][()],
                    dtype=torch.float,
                    device=device,
                )
            )
            grouped_field_data[heliostat_group_key][
                config_dictionary.aim_points
            ].append(
                torch.tensor(
                    single_heliostat_config[config_dictionary.heliostat_aim_point][()],
                    dtype=torch.float,
                    device=device,
                )
            )
            grouped_field_data[heliostat_group_key][
                config_dictionary.surface_points
            ].append(
                surface.get_surface_points_and_normals(device=device)[0].reshape(-1, 4)
            )
            grouped_field_data[heliostat_group_key][
                config_dictionary.surface_normals
            ].append(
                surface.get_surface_points_and_normals(device=device)[1].reshape(-1, 4)
            )
            grouped_field_data[heliostat_group_key][
                config_dictionary.initial_orientations
            ].append(initial_orientation)
            grouped_field_data[heliostat_group_key][
                config_dictionary.kinematic_deviation_parameters
            ].append(kinematic_deviations)
            grouped_field_data[heliostat_group_key][
                config_dictionary.actuator_parameters
            ].append(actuator_parameters)

        for group in grouped_field_data:
            for key in grouped_field_data[group]:
                if key != config_dictionary.names:
                    grouped_field_data[group][key] = torch.stack(
                        grouped_field_data[group][key]
                    )

        heliostat_groups = []
        for heliostat_group_name in grouped_field_data.keys():
            heliostat_groups.append(
                type_mappings.heliostat_group_type_mapping[heliostat_group_name](
                    names=grouped_field_data[heliostat_group_name][
                        config_dictionary.names
                    ],
                    positions=grouped_field_data[heliostat_group_name][
                        config_dictionary.positions
                    ],
                    aim_points=grouped_field_data[heliostat_group_name][
                        config_dictionary.aim_points
                    ],
                    surface_points=grouped_field_data[heliostat_group_name][
                        config_dictionary.surface_points
                    ],
                    surface_normals=grouped_field_data[heliostat_group_name][
                        config_dictionary.surface_normals
                    ],
                    initial_orientations=grouped_field_data[heliostat_group_name][
                        config_dictionary.initial_orientations
                    ],
                    kinematic_deviation_parameters=grouped_field_data[
                        heliostat_group_name
                    ][config_dictionary.kinematic_deviation_parameters],
                    actuator_parameters=grouped_field_data[heliostat_group_name][
                        config_dictionary.actuator_parameters
                    ],
                    device=device,
                )
            )
            log.info(
                f"Added a heliostat group with kinematic type: {kinematic_type}, and actuator type: {actuator_type}, to the heliostat field."
            )

        return cls(
            heliostat_groups=heliostat_groups,
        )

    def forward(self) -> None:
        """
        Specify the forward pass.

        Raises
        ------
        NotImplementedError
            Whenever called.
        """
        raise NotImplementedError("Not Implemented!")
