import logging
from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import h5py
import torch
from typing_extensions import Self

from artist.data_loader import h5_loader
from artist.field.heliostat_group import HeliostatGroup
from artist.field.surface import Surface

if TYPE_CHECKING:
    from artist.scenario.configuration_classes import (
        SurfaceConfig,
    )
from artist.util import config_dictionary, type_mappings
from artist.util.environment_setup import get_device

log = logging.getLogger(__name__)
"""A logger for the heliostat field."""


class HeliostatField:
    """
    The heliostat field.

    A heliostat field consists of one or multiple heliostat groups. Each heliostat group contains all
    heliostats with a specific kinematic type and actuator type. The heliostats in the field are aligned
    individually to reflect the incoming light in a way that ensures maximum efficiency for the whole power plant.

    Attributes
    ----------
    heliostat_groups : Sequence[HeliostatGroup]
        A list containing all heliostat groups.
    number_of_heliostat_groups : int
        The number of different heliostat groups in the heliostat field.

    Methods
    -------
    from_hdf5()
        Load a heliostat field from an HDF5 file.
    """

    def __init__(
        self,
        heliostat_groups: Sequence[HeliostatGroup],
    ) -> None:
        """
        Initialize the heliostat field with heliostat groups.

        Parameters
        ----------
        heliostat_groups : Sequence[HeliostatGroup]
            A list containing all heliostat groups.
        """
        self.heliostat_groups = heliostat_groups
        self.number_of_heliostat_groups = len(self.heliostat_groups)

    @classmethod
    def from_hdf5(
        cls,
        config_file: h5py.File,
        prototype_surface: "SurfaceConfig",
        prototype_kinematic: dict[str, str | torch.Tensor],
        prototype_actuators: dict[str, str | torch.Tensor],
        number_of_points_per_facet: torch.Tensor,
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
        number_of_points_per_facet : torch.Tensor
            The number of surface points per facet in east and then in north direction.
            Tensor of shape [2].
        device : device: torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

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

        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        if rank == 0:
            log.info("Loading a heliostat field from an HDF5 file.")

        grouped_field_data: defaultdict[
            str, defaultdict[str, list[str | torch.Tensor]]
        ] = defaultdict(lambda: defaultdict(list))

        for heliostat_name in config_file[config_dictionary.heliostat_key].keys():
            single_heliostat_config = config_file[config_dictionary.heliostat_key][
                heliostat_name
            ]

            if (
                config_dictionary.heliostat_surface_key
                in single_heliostat_config.keys()
            ):
                surface_config = h5_loader.surface_config(
                    prototype=False,
                    scenario_file=single_heliostat_config,
                    device=device,
                )
            else:
                if prototype_surface is None:
                    raise ValueError(
                        "If the heliostat does not have individual surface parameters, a surface prototype must be provided!"
                    )
                if rank == 0:
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
                    h5_loader.kinematic_deviations(
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
                if rank == 0:
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

                actuator_type_list = []
                for key in actuator_keys:
                    actuator_type_list.append(
                        single_heliostat_config[
                            config_dictionary.heliostat_actuator_key
                        ][key][config_dictionary.actuator_type_key][()].decode("utf-8")
                    )

                unique_actuators = {a for a in actuator_type_list}

                if kinematic_type == config_dictionary.rigid_body_key:
                    if len(unique_actuators) > 1:
                        raise ValueError(
                            "When using the Rigid Body Kinematic, all actuators for a given heliostat must have the same type."
                        )
                    else:
                        actuator_type = actuator_type_list[0]

                actuator_parameters = h5_loader.actuator_parameters(
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
                if rank == 0:
                    log.info(
                        "Individual actuator configurations not provided - loading a heliostat with the actuator prototype."
                    )
                actuator_type = prototype_actuators[config_dictionary.actuator_type_key]
                actuator_parameters = prototype_actuators[
                    config_dictionary.actuator_parameters_key
                ]

            surface = Surface(surface_config, device=device)
            
            number_of_facets = len(surface_config.facet_list)
            degrees = torch.empty(2, dtype=torch.int32, device=device)
            # Each facet automatically has the same control points dimensions. This is required in ARTIST.
            # control_points: Tensor of shape [number_of_surfaces, number_of_facets_per_surface, number_of_control_points_u_direction, number_of_control_points_v_direction, 3].
            control_points = torch.empty(
                (
                    number_of_facets,
                    surface_config.facet_list[0].control_points.shape[0],
                    surface_config.facet_list[0].control_points.shape[1],
                    3,
                ),
                device=device,
            )
            for i in range(number_of_facets):
                degrees = surface_config.facet_list[i].degrees
                control_points[i] = surface_config.facet_list[i].control_points

            heliostat_group_key = f"{kinematic_type}_{actuator_type}"

            grouped_field_data[heliostat_group_key][config_dictionary.names].append(
                heliostat_name
            )
            grouped_field_data[heliostat_group_key][
                config_dictionary.heliostat_group_type
            ] = [kinematic_type, actuator_type]

            grouped_field_data[heliostat_group_key][config_dictionary.positions].append(
                torch.tensor(
                    single_heliostat_config[config_dictionary.heliostat_position][()],
                    dtype=torch.float,
                    device=device,
                )
            )
            grouped_field_data[heliostat_group_key][
                config_dictionary.surface_points
            ].append(
                surface.get_surface_points_and_normals(
                    number_of_points_per_facet=number_of_points_per_facet, device=device
                )[0]
            )
            grouped_field_data[heliostat_group_key][
                config_dictionary.surface_normals
            ].append(
                surface.get_surface_points_and_normals(
                    number_of_points_per_facet=number_of_points_per_facet, device=device
                )[1]
            )
            grouped_field_data[heliostat_group_key][
                config_dictionary.facet_control_points
            ].append(control_points)
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
                if key not in [
                    config_dictionary.names,
                    config_dictionary.kinematic_type,
                    config_dictionary.actuator_type_key,
                ]:
                    grouped_field_data[group][key] = torch.stack(
                        grouped_field_data[group][key]
                    )

        heliostat_groups = []
        for heliostat_group_name in grouped_field_data.keys():
            number_of_heliostats_in_group = len(
                grouped_field_data[heliostat_group_name][config_dictionary.names]
            )
            heliostat_groups.append(
                type_mappings.heliostat_group_type_mapping[heliostat_group_name](
                    names=grouped_field_data[heliostat_group_name][
                        config_dictionary.names
                    ],
                    positions=grouped_field_data[heliostat_group_name][
                        config_dictionary.positions
                    ],
                    surface_points=cast(
                        torch.Tensor,
                        grouped_field_data[heliostat_group_name][
                            config_dictionary.surface_points
                        ],
                    ).reshape(number_of_heliostats_in_group, -1, 4),
                    surface_normals=cast(
                        torch.Tensor,
                        grouped_field_data[heliostat_group_name][
                            config_dictionary.surface_normals
                        ],
                    ).reshape(number_of_heliostats_in_group, -1, 4),
                    nurbs_control_points=grouped_field_data[heliostat_group_name][
                        config_dictionary.facet_control_points
                    ],
                    nurbs_degrees=degrees,
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
            if rank == 0:
                log.info(
                    f"Added a heliostat group with kinematic type: {grouped_field_data[heliostat_group_name][config_dictionary.heliostat_group_type][0]}, and actuator type: {grouped_field_data[heliostat_group_name][config_dictionary.heliostat_group_type][1]}, to the heliostat field."
                )

        return cls(
            heliostat_groups=heliostat_groups,
        )
