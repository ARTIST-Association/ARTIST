from typing import Any, Dict, Tuple

import h5py
import torch
from typing_extensions import Self

from artist.field.alignment import Alignment
from artist.field.concentrator import Concentrator
from artist.util import artist_type_mapping_dict, config_dictionary


class Heliostat(torch.nn.Module):
    """
    Implementation of the heliostat as a module.

    Attributes
    ----------
    id : int
        Unique ID of the heliostat.
    incident_ray_direction : torch.Tensor
        The direction of the rays.
    concentrator : Concentrator
        The surface of the heliostat.
    alignment : Alignment
        The alignment module of the heliostat.

    Methods
    -------
    from_hdf5()
        Class method to initialize heliostat from an hdf5 file.
    get_aligned_surface()
        Compute the aligned surface points and aligned surface normals of the heliostat.
    """

    def __init__(
        self,
        id: int,
        position: torch.Tensor,
        alignment_type: Any,
        actuator_type: Any,
        aim_point: torch.Tensor,
        facet_type: Any,
        surface_points: torch.Tensor,
        surface_normals: torch.Tensor,
        incident_ray_direction: torch.Tensor,
        kinematic_deviation_parameters: Dict[str, torch.Tensor],
        kinematic_initial_orientation_offsets: float,
        actuator_parameters: Dict[str, torch.Tensor],
    ) -> None:
        """
        Initialize the heliostat.

        Parameters
        ----------
        id : int
            Unique ID of the heliostat.
        position : torch.Tensor
            The position of the heliostat in the field.
        alignment_type : Any
            The method by which the heliostat is aligned, currently only rigid-body is possible.
        actuator_type : Any
            The type of the actuators of the heliostat.
        aim_point : torch.Tensor
            The aimpoint.
        facet_type : Any
            The type of the facets, for example point cloud facets or NURBS.
        surface_points : torch.Tensor
            The points on the surface of the heliostat.
        surface_normals : torch.Tensor
            The normals corresponding to the points on the surface.
        incident_ray_direction : torch.Tensor
            The direction of the incident ray as seen from the heliostat.
        kinematic_deviation_parameters : Dict[str, torch.Tensor]
            The 18 deviation parameters of the kinematic module.
        kinematic_initial_orientation_offsets : Dict[str, torch.Tensor]
            The initial orientation-rotation angles of the heliostat.
        actuator_parameters : Dict[str, torch.Tensor]
            Parameters describing the imperfect actuators.
        """
        super().__init__()
        self.id = id
        self.incident_ray_direction = incident_ray_direction
        self.concentrator = Concentrator(
            facets_type=facet_type,
            surface_points=surface_points,
            surface_normals=surface_normals,
        )
        self.alignment = Alignment(
            alignment_type=alignment_type,
            actuator_type=actuator_type,
            position=position,
            aim_point=aim_point,
            kinematic_deviation_parameters=kinematic_deviation_parameters,
            kinematic_initial_orientation_offsets=kinematic_initial_orientation_offsets,
            actuator_parameters=actuator_parameters,
        )

    @classmethod
    def from_hdf5(
        cls,
        config_file: h5py.File,
        incident_ray_direction: torch.Tensor,
        heliostat_name: str,
    ) -> Self:
        """
        Class method to initialize heliostat from an hdf5 file.

        Parameters
        ----------
        config_file : h5py.File
            The config file containing all the information about the heliostat and the environment.
        incident_ray_direction : torch.Tensor
            The direction of the incident ray as seen from the heliostat.
        heliostat_name : str
            The name of the heliostat, for identification.

        Returns
        -------
        Heliostat
            A heliostat initialized from an h5 file.
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

        try:
            alignment_type = artist_type_mapping_dict.alignment_type_mapping[
                alignment_type
            ]
        except KeyError:
            raise KeyError(
                f"Currently the selected alignment type: {alignment_type} is not supported."
            )

        actuator_type = config_file[config_dictionary.heliostat_prefix][heliostat_name][
            config_dictionary.actuator_type_key
        ][()].decode("utf-8")

        try:
            actuator_type = artist_type_mapping_dict.actuator_type_mapping[
                actuator_type
            ]
        except KeyError:
            raise KeyError(
                f"Currently the selected actuator type: {actuator_type} is not supported."
            )
        aim_point = torch.tensor(
            config_file[config_dictionary.heliostat_prefix][heliostat_name][
                config_dictionary.heliostat_aim_point
            ][()],
            dtype=torch.float,
        )
        facet_type = config_file[config_dictionary.heliostat_prefix][heliostat_name][
            config_dictionary.facets_type_key
        ][()].decode("utf-8")

        try:
            facet_type = artist_type_mapping_dict.facet_type_mapping[facet_type]
        except KeyError:
            raise KeyError(
                f"Currently the selected facet type: {facet_type} is not supported."
            )

        if config_file[config_dictionary.heliostat_prefix][heliostat_name][
            config_dictionary.has_individual_surface_points
        ][()]:
            surface_points = torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.heliostat_individual_surface_points
                ][()],
                dtype=torch.float,
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
                ][()],
                dtype=torch.float,
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

        kinematic_initial_orientation_offsets = {
            config_dictionary.kinematic_initial_orientation_offset_e: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.kinematic_initial_orientation_offset_key
                ][config_dictionary.kinematic_initial_orientation_offset_e][()],
                dtype=torch.float,
            ),
            config_dictionary.kinematic_initial_orientation_offset_n: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.kinematic_initial_orientation_offset_key
                ][config_dictionary.kinematic_initial_orientation_offset_n][()],
                dtype=torch.float,
            ),
            config_dictionary.kinematic_initial_orientation_offset_u: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.kinematic_initial_orientation_offset_key
                ][config_dictionary.kinematic_initial_orientation_offset_u][()],
                dtype=torch.float,
            ),
        }
        actuator_parameters = {
            config_dictionary.first_joint_increment: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.actuator_parameters_key
                ][config_dictionary.first_joint_increment][()],
                dtype=torch.float,
            ),
            config_dictionary.first_joint_initial_stroke_length: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.actuator_parameters_key
                ][config_dictionary.first_joint_initial_stroke_length][()],
                dtype=torch.float,
            ),
            config_dictionary.first_joint_actuator_offset: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.actuator_parameters_key
                ][config_dictionary.first_joint_actuator_offset][()],
                dtype=torch.float,
            ),
            config_dictionary.first_joint_radius: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.actuator_parameters_key
                ][config_dictionary.first_joint_radius][()],
                dtype=torch.float,
            ),
            config_dictionary.first_joint_phi_0: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.actuator_parameters_key
                ][config_dictionary.first_joint_phi_0][()],
                dtype=torch.float,
            ),
            config_dictionary.second_joint_increment: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.actuator_parameters_key
                ][config_dictionary.second_joint_increment][()],
                dtype=torch.float,
            ),
            config_dictionary.second_joint_initial_stroke_length: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.actuator_parameters_key
                ][config_dictionary.second_joint_initial_stroke_length][()],
                dtype=torch.float,
            ),
            config_dictionary.second_joint_actuator_offset: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.actuator_parameters_key
                ][config_dictionary.second_joint_actuator_offset][()],
                dtype=torch.float,
            ),
            config_dictionary.second_joint_radius: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.actuator_parameters_key
                ][config_dictionary.second_joint_radius][()],
                dtype=torch.float,
            ),
            config_dictionary.second_joint_phi_0: torch.tensor(
                config_file[config_dictionary.heliostat_prefix][heliostat_name][
                    config_dictionary.actuator_parameters_key
                ][config_dictionary.second_joint_phi_0][()],
                dtype=torch.float,
            ),
        }

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
            kinematic_initial_orientation_offsets=kinematic_initial_orientation_offsets,
            actuator_parameters=actuator_parameters,
        )

    def get_aligned_surface(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the aligned surface points and aligned surface normals of the heliostat.

        Returns
        -------
        torch.Tensor
            The aligned surface points.
        torch.Tensor
            The aligned surface normals.
        """
        surface_points, surface_normals = (
            self.concentrator.facets.surface_points,
            self.concentrator.facets.surface_normals,
        )
        aligned_surface_points, aligned_surface_normals = self.alignment.align_surface(
            self.incident_ray_direction, surface_points, surface_normals
        )
        return aligned_surface_points, aligned_surface_normals
