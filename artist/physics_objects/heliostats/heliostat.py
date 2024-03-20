from typing import Dict, Tuple
from typing_extensions import Self

import h5py
import torch

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
    id : int
        Unique ID of the heliostat.
    incident_ray_direction : torch.Tensor
        The direction of the rays.
    concentrator : ConcentratorModule
        The surface of the heliostat.
    alignment : AlignmentModule
        The alignment module of the heliostat.

    Methods
    -------
    from_hdf5()
        Classmethod to initialize helisotat from an h5 file.
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
        id : int
            Unique ID of the heliostat.
        position : torch.Tensor
            The position of the heliostat in the field.
        alignment_type : str
            The method by which the helisotat is aligned, currently only rigid-body is possible.
        actuator_type : str
            The type of the actuators of the heliostat.
        aim_point : torch.Tensor
            The aimpoint.
        facet_type : str
            The type of the facets, for example point cloud facets or NURBS.
        surface_points : torch.Tensor
            The points on the surface of the heliostat.
        surface_normals : torch.Tensor
            The normals corresponding to the points on the surface.
        incident_ray_direction : torch.Tensor
            The direction of the incident ray as seen from the heliostat.
        kinematic_deviation_parameters : Dict[str, torch.Tensor]
            The 18 deviation parameters of the kinematic module.
        kinematic_initial_orientation_offset : float
            The initial orientation-rotation angle of the heliostat.
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
    def from_hdf5(cls, config_file: h5py.File, incident_ray_direction: torch.Tensor, heliostat_name: str) -> Self:
        """
        Classmethod to initialize helisotat from an h5 file.

        Parameters
        ----------
        config_file : h5py.File
            The config file containing all the information about the heliostat and the environment.
        incident_ray_direction : torch.Tensor
            The direction of the incident ray as seen from the heliostat.
        helisotat_name : str
            The name of the heliostat, for identification.
        
        Returns
        -------
        HeliostatModule
            A heliostat initialized from a h5 file.
        """
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

        if config_file[config_dictionary.heliostat_prefix][heliostat_name][
                config_dictionary.has_individual_surface_points
            ][()]:
            surface_points = torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.heliostat_individual_surface_points
                ][()], dtype=torch.float
            )
        else:
            surface_points = torch.tensor(
                config_file[config_dictionary.heliostat_prefix][
                    config_dictionary.general_surface_points
                ][()],
                dtype=torch.float,
            )
        if config_file[config_dictionary.heliostat_prefix][heliostat_name][
                config_dictionary.has_individual_surface_normals
            ][()]:
            surface_normals = torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.heliostat_individual_surface_normals
                ][()], dtype=torch.float
            )
        else:
            surface_normals = torch.tensor(
                config_file[config_dictionary.heliostat_prefix][
                    config_dictionary.general_surface_normals
                ][()],
                dtype=torch.float,
            )
        
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

    def get_aligned_surface(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the aligned surface points and aligned surface normals of the heliostat.

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
