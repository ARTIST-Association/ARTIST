from typing import Any, Dict, Tuple

import h5py
from yacs.config import CfgNode

import torch

from artist.io.datapoint import HeliostatDataPoint
from artist.physics_objects.heliostats.concentrator.concentrator import (
    ConcentratorModule,
)
from artist.physics_objects.heliostats.alignment.alignment import AlignmentModule
from artist.physics_objects.module import AModule
from artist.util import config_dictionary


class HeliostatModule(AModule):
    """
    Implementation of the Heliostat as a module.

    Attributes
    ----------
    aim_point : torch.Tensor
        The aim point on the receiver.
    incident_ray_direction : torch.Tensor
        The direction of the rays.
    concentrator : ConcentratorModule
        The surface of the heliostat.
    alignment : AlignmentModule
        The alignment module of the heliostat.

    Methods
    -------
    get_aligned_surface()
        Compute the aligned surface points and aligned surface normals of the heliostat.

    See also
    --------
    :class:AModule : Reference to the parent class.
    """

    def __init__(
        self,
        id: int,
        position: torch.Tensor,
        alignment_type: str,
        actuator_type: str,
        aim_point: torch.Tensor,
        facet_type: str,
        surface_points: torch.Tensor,
        surface_normals: torch.Tensor,
        incident_ray_direction: torch.Tensor,
        kinematic_deviation_parameters: Dict[str, torch.Tensor],
        kinematic_initial_orientation_offset: float,
    ) -> None:
        """
        Initialize the heliostat.

        Parameters
        ----------
        incident_ray_direction : torch.Tensor
            The direction of the incident ray as seen from the heliostat.
        heliostat_name : str
            The name of the heliostat being initialized.
        """
        super().__init__()
        self.id = id
        self.incident_ray_direction = incident_ray_direction
        self.concentrator = ConcentratorModule(
            facets_type=facet_type,
            surface_points=surface_points,
            surface_normals=surface_normals,
        )
        self.alignment = AlignmentModule(
            alignment_type=alignment_type,
            actuator_type=actuator_type,
            position=position,
            aim_point=aim_point,
            kinematic_deviation_parameters=kinematic_deviation_parameters,
            kinematic_initial_orientation_offset=kinematic_initial_orientation_offset,
        )

    @classmethod
    def from_hdf5(cls, config_file: h5py.File, incident_ray_direction, heliostat_name):
        heliostat_id = config_file[config_dictionary.heliostat_prefix][heliostat_name][
            config_dictionary.heliostat_id
        ][()]
        heliostat_position = torch.tensor(
            config_file[config_dictionary.heliostat_prefix][heliostat_name][
                config_dictionary.heliostat_position
            ][()],
            dtype=torch.float,
        )
        alignment_type = config_file[config_dictionary.heliostat_prefix][
            heliostat_name
        ][config_dictionary.alignment_type_key][()].decode("utf-8")
        actuator_type = config_file[config_dictionary.heliostat_prefix][heliostat_name][
            config_dictionary.actuator_type_key
        ][()].decode("utf-8")
        aim_point = torch.tensor(
            config_file[config_dictionary.heliostat_prefix][heliostat_name][
                config_dictionary.heliostat_aim_point
            ][()],
            dtype=torch.float,
        )
        facet_type = config_file[config_dictionary.heliostat_prefix][heliostat_name][
            config_dictionary.facets_type_key
        ][()].decode("utf-8")
        surface_points = torch.tensor(
            config_file[config_dictionary.heliostat_prefix][heliostat_name][
                config_dictionary.heliostat_individual_surface_points
            ][()]
        )
        surface_normals = torch.tensor(
            config_file[config_dictionary.heliostat_prefix][heliostat_name][
                config_dictionary.heliostat_individual_surface_normals
            ][()]
        )

        if surface_points.dtype == torch.bool and not surface_points:
            surface_points = torch.tensor(
                config_file[config_dictionary.heliostat_prefix][
                    config_dictionary.general_surface_points
                ][()],
                dtype=torch.float,
            )
        elif surface_points.dtype != torch.float:
            surface_points = surface_points.type(torch.float)

        if surface_normals.dtype == torch.bool and not surface_normals:
            surface_normals = torch.tensor(
                config_file[config_dictionary.heliostat_prefix][
                    config_dictionary.general_surface_normals
                ][()],
                dtype=torch.float,
            )
        elif surface_normals.dtype != torch.float:
            surface_normals = surface_normals.type(torch.float)

        kinematic_deviation_parameters = {
            config_dictionary.first_joint_translation_e: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.kinematic_deviation_key
                ][config_dictionary.first_joint_translation_e][()],
                dtype=torch.float,
            ),
            config_dictionary.first_joint_translation_n: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.kinematic_deviation_key
                ][config_dictionary.first_joint_translation_n][()],
                dtype=torch.float,
            ),
            config_dictionary.first_joint_translation_u: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.kinematic_deviation_key
                ][config_dictionary.first_joint_translation_u][()],
                dtype=torch.float,
            ),
            config_dictionary.first_joint_tilt_e: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.kinematic_deviation_key
                ][config_dictionary.first_joint_tilt_e][()],
                dtype=torch.float,
            ),
            config_dictionary.first_joint_tilt_n: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.kinematic_deviation_key
                ][config_dictionary.first_joint_tilt_n][()],
                dtype=torch.float,
            ),
            config_dictionary.first_joint_tilt_u: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.kinematic_deviation_key
                ][config_dictionary.first_joint_tilt_u][()],
                dtype=torch.float,
            ),
            config_dictionary.second_joint_translation_e: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.kinematic_deviation_key
                ][config_dictionary.second_joint_translation_e][()],
                dtype=torch.float,
            ),
            config_dictionary.second_joint_translation_n: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.kinematic_deviation_key
                ][config_dictionary.second_joint_translation_n][()],
                dtype=torch.float,
            ),
            config_dictionary.second_joint_translation_u: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.kinematic_deviation_key
                ][config_dictionary.second_joint_translation_u][()],
                dtype=torch.float,
            ),
            config_dictionary.second_joint_tilt_e: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.kinematic_deviation_key
                ][config_dictionary.second_joint_tilt_e][()],
                dtype=torch.float,
            ),
            config_dictionary.second_joint_tilt_n: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.kinematic_deviation_key
                ][config_dictionary.second_joint_tilt_n][()],
                dtype=torch.float,
            ),
            config_dictionary.second_joint_tilt_u: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.kinematic_deviation_key
                ][config_dictionary.second_joint_tilt_u][()],
                dtype=torch.float,
            ),
            config_dictionary.concentrator_translation_e: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.kinematic_deviation_key
                ][config_dictionary.concentrator_translation_e][()],
                dtype=torch.float,
            ),
            config_dictionary.concentrator_translation_n: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.kinematic_deviation_key
                ][config_dictionary.concentrator_translation_n][()],
                dtype=torch.float,
            ),
            config_dictionary.concentrator_translation_u: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.kinematic_deviation_key
                ][config_dictionary.concentrator_translation_u][()],
                dtype=torch.float,
            ),
            config_dictionary.concentrator_tilt_e: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.kinematic_deviation_key
                ][config_dictionary.concentrator_tilt_e][()],
                dtype=torch.float,
            ),
            config_dictionary.concentrator_tilt_n: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.kinematic_deviation_key
                ][config_dictionary.concentrator_tilt_n][()],
                dtype=torch.float,
            ),
            config_dictionary.concentrator_tilt_u: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.kinematic_deviation_key
                ][config_dictionary.concentrator_tilt_u][()],
                dtype=torch.float,
            ),
        }

        kinematic_initial_orientation_offset = float(
            config_file[config_dictionary.heliostat_prefix][heliostat_name][
                config_dictionary.kinematic_initial_orientation_offset_key
            ][()]
        )

        return cls(
            id=heliostat_id,
            position=heliostat_position,
            alignment_type=alignment_type,
            actuator_type=actuator_type,
            aim_point=aim_point,
            facet_type=facet_type,
            surface_points=surface_points,
            surface_normals=surface_normals,
            incident_ray_direction=incident_ray_direction,
            kinematic_deviation_parameters=kinematic_deviation_parameters,
            kinematic_initial_orientation_offset=kinematic_initial_orientation_offset,
        )

    # def __init__(
    #     self,
    #     heliostat_name: str,
    #     incident_ray_direction: torch.Tensor,
    #     config_file: h5py.File = None,
    # ) -> None:
    #     """
    #     Initialize the heliostat.

    #     Parameters
    #     ----------
    #     heliostat_name : str
    #         The name of the heliostat being initialized.
    #     incident_ray_direction : torch.Tensor
    #         The direction of the incident ray as seen from the heliostat.
    #     config_file : h5py.File
    #         An open hdf5 file containing the scenario configuration.
    #     """
    #     super().__init__()
    #     self.position = torch.tensor(
    #         config_file[config_dictionary.heliostat_prefix][config_dictionary.heliostats_list][heliostat_name][config_dictionary.heliostat_position][
    #             ()
    #         ],
    #         dtype=torch.float,
    #     )
    #     self.incident_ray_direction = incident_ray_direction
    #     self.concentrator = ConcentratorModule(
    #         heliostat_name=heliostat_name, config_file=config_file
    #     )
    #     self.alignment = AlignmentModule(
    #         heliostat_name=heliostat_name, config_file=config_file
    #     )

    def get_aligned_surface(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the aligned surface points and aligned surface normals of the heliostat.

        Parameters
        ----------
        aim_point : torch.Tensor
            The desired aim point.
        incident_ray_direction : torch.Tensor
            The direction of the rays.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The aligned surface points and aligned surface normals.
        """
        surface_points, surface_normals = (
            self.concentrator.facets.surface_points,
            self.concentrator.facets.surface_normals,
        )
        aligned_surface_points, aligned_surface_normals = self.alignment.align_surface(
            self.incident_ray_direction, surface_points, surface_normals
        )
        return aligned_surface_points, aligned_surface_normals
