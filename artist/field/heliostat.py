from typing import Optional

import h5py
import torch
from typing_extensions import Self

from artist.field.surface import Surface
from artist.util import artist_type_mapping_dict, config_dictionary, utils
from artist.util.configuration_classes import (
    ActuatorConfig,
    ActuatorListConfig,
    ActuatorParameters,
    FacetConfig,
    KinematicConfig,
    KinematicDeviations,
    KinematicOffsets,
    SurfaceConfig,
)


class Heliostat(torch.nn.Module):
    """
    This class realizes a heliostat.

    Attributes
    ----------
    heliostat_id : int
        Unique ID of the heliostat.
    position : torch.Tensor
        The position of the heliostat in the field.
    aim_point : torch.Tensor
        The aim point of the heliostat.
    surface : Surface
        The surface of the heliostat.
    kinematic : Union[RigidBodyKinematic,...]
        The kinematic used in the heliostat.
    current_aligned_surface_points : torch.Tensor
        The current aligned surface points for the heliostat.
    current_aligned_surface_normals : torch.Tensor
        The current aligned surface normals for the heliostat.
    is_aligned : bool
        Boolean indicating if the heliostat is aligned.
    preferred_reflection_direction : torch.Tensor
        The preferred reflection direction for rays reflecting off the heliostat.

    Methods
    -------
    from_hdf5()
        Class method to initialize heliostat from an HDF5 file.
    set_aligned_surface()
        Compute the aligned surface points and aligned surface normals of the heliostat.
    set_preferred_reflection_direction()
        Compute the preferred reflection direction for each normal vector given an incident ray direction.
    """

    def __init__(
        self,
        heliostat_id: int,
        position: torch.Tensor,
        aim_point: torch.Tensor,
        surface_config: SurfaceConfig,
        kinematic_config: KinematicConfig,
        actuator_config: ActuatorListConfig,
    ) -> None:
        """
        Initialize the heliostat.

        Parameters
        ----------
        heliostat_id : int
            Unique ID of the heliostat.
        position : torch.Tensor
            The position of the heliostat in the field.
        aim_point : torch.Tensor
            The aim point of the heliostat.
        surface_config : SurfaceConfig
            The configuration parameters to use for the heliostat surface.
        kinematic_config : KinematicConfig
            The configuration parameters to use for the heliostat kinematic.
        actuator_config : ActuatorListConfig
            The configuration parameters to use for the list of actuators.
        """
        super().__init__()
        self.heliostat_id = heliostat_id
        self.position = position
        self.aim_point = aim_point
        self.surface = Surface(surface_config=surface_config)
        try:
            kinematic_object = artist_type_mapping_dict.kinematic_type_mapping[
                kinematic_config.kinematic_type
            ]
        except KeyError:
            raise KeyError(
                f"Currently the selected kinematic type: {kinematic_config.kinematic_type} is not supported."
            )
        self.kinematic = kinematic_object(
            position=position,
            aim_point=aim_point,
            actuator_config=actuator_config,
            initial_orientation_offsets=kinematic_config.kinematic_initial_orientation_offsets,
            deviation_parameters=kinematic_config.kinematic_deviations,
        )

        self.current_aligned_surface_points = None
        self.current_aligned_surface_normals = None
        self.is_aligned = False
        self.preferred_reflection_direction = None

    @classmethod
    def from_hdf5(
        cls,
        config_file: h5py.File,
        prototype_surface: Optional[SurfaceConfig] = None,
        prototype_kinematic: Optional[KinematicConfig] = None,
        prototype_actuator: Optional[ActuatorListConfig] = None,
    ) -> Self:
        """
        Class method to initialize heliostat from an HDF5 file.

        Parameters
        ----------
        config_file : h5py.File
            The config file containing all the information about the heliostat.
        prototype_surface  : Optional[SurfaceConfig]
            An optional prototype for the surface configuration.
        prototype_kinematic : Optional[KinematicConfig]
            An optional prototype for the kinematic configuration.
        prototype_actuator : Optional[ActuatorListConfig]
            An optional prototype for the actuator configuration.

        Returns
        -------
        Heliostat
            A heliostat initialized from an HDF5 file.
        """
        heliostat_id = int(config_file[config_dictionary.heliostat_id][()])
        position = torch.tensor(
            config_file[config_dictionary.heliostat_position][()], dtype=torch.float
        )
        aim_point = torch.tensor(
            config_file[config_dictionary.heliostat_aim_point][()], dtype=torch.float
        )

        if config_dictionary.heliostat_surface_key in config_file.keys():
            facets_list = [
                FacetConfig(
                    facet_key="",
                    control_points_e=torch.tensor(
                        config_file[config_dictionary.heliostat_surface_key][
                            config_dictionary.facets_key
                        ][facet][config_dictionary.facet_control_points_e][()],
                        dtype=torch.float,
                    ),
                    control_points_u=torch.tensor(
                        config_file[config_dictionary.heliostat_surface_key][
                            config_dictionary.facets_key
                        ][facet][config_dictionary.facet_control_points_u][()],
                        dtype=torch.float,
                    ),
                    knots_e=torch.tensor(
                        config_file[config_dictionary.heliostat_surface_key][
                            config_dictionary.facets_key
                        ][facet][config_dictionary.facet_knots_e][()],
                        dtype=torch.float,
                    ),
                    knots_u=torch.tensor(
                        config_file[config_dictionary.heliostat_surface_key][
                            config_dictionary.facets_key
                        ][facet][config_dictionary.facet_knots_u][()],
                        dtype=torch.float,
                    ),
                    width=float(
                        config_file[config_dictionary.heliostat_surface_key][
                            config_dictionary.facets_key
                        ][facet][config_dictionary.facets_width][()]
                    ),
                    height=float(
                        config_file[config_dictionary.heliostat_surface_key][
                            config_dictionary.facets_key
                        ][facet][config_dictionary.facets_height][()]
                    ),
                    position=torch.tensor(
                        config_file[config_dictionary.heliostat_surface_key][
                            config_dictionary.facets_key
                        ][facet][config_dictionary.facets_position][()],
                        dtype=torch.float,
                    ),
                    canting_e=torch.tensor(
                        config_file[config_dictionary.heliostat_surface_key][
                            config_dictionary.facets_key
                        ][facet][config_dictionary.facets_canting_e][()],
                        dtype=torch.float,
                    ),
                    canting_u=torch.tensor(
                        config_file[config_dictionary.heliostat_surface_key][
                            config_dictionary.facets_key
                        ][facet][config_dictionary.facets_canting_u][()],
                        dtype=torch.float,
                    ),
                )
                for facet in config_file[config_dictionary.heliostat_surface_key][
                    config_dictionary.facets_key
                ].keys()
            ]
            surface_config = SurfaceConfig(facets_list=facets_list)
        else:
            assert (
                prototype_surface is not None
            ), "If the heliostat does not have individual surface parameters a surface prototype must be provided!"
            surface_config = prototype_surface

        if config_dictionary.heliostat_kinematic_key in config_file.keys():
            kinematic_initial_orientation_offset_e = torch.tensor(0.0)
            kinematic_initial_orientation_offset_n = torch.tensor(0.0)
            kinematic_initial_orientation_offset_u = torch.tensor(0.0)
            if (
                config_dictionary.kinematic_offsets_key
                in config_file[config_dictionary.heliostat_kinematic_key].keys()
            ):
                if (
                    config_dictionary.kinematic_initial_orientation_offset_e
                    in config_file[config_dictionary.heliostat_kinematic_key][
                        config_dictionary.kinematic_offsets_key
                    ].keys()
                ):
                    kinematic_initial_orientation_offset_e = torch.tensor(
                        config_file[config_dictionary.heliostat_kinematic_key][
                            config_dictionary.kinematic_offsets_key
                        ][config_dictionary.kinematic_initial_orientation_offset_e][()],
                        dtype=torch.float,
                    )
                if (
                    config_dictionary.kinematic_initial_orientation_offset_n
                    in config_file[config_dictionary.heliostat_kinematic_key][
                        config_dictionary.kinematic_offsets_key
                    ].keys()
                ):
                    kinematic_initial_orientation_offset_n = torch.tensor(
                        config_file[config_dictionary.heliostat_kinematic_key][
                            config_dictionary.kinematic_offsets_key
                        ][config_dictionary.kinematic_initial_orientation_offset_n][()],
                        dtype=torch.float,
                    )
                if (
                    config_dictionary.kinematic_initial_orientation_offset_u
                    in config_file[config_dictionary.heliostat_kinematic_key][
                        config_dictionary.kinematic_offsets_key
                    ].keys()
                ):
                    kinematic_initial_orientation_offset_u = torch.tensor(
                        config_file[config_dictionary.heliostat_kinematic_key][
                            config_dictionary.kinematic_offsets_key
                        ][config_dictionary.kinematic_initial_orientation_offset_u][()],
                        dtype=torch.float,
                    )
            kinematic_offsets = KinematicOffsets(
                kinematic_initial_orientation_offset_e=kinematic_initial_orientation_offset_e,
                kinematic_initial_orientation_offset_n=kinematic_initial_orientation_offset_n,
                kinematic_initial_orientation_offset_u=kinematic_initial_orientation_offset_u,
            )
            first_joint_translation_e = torch.tensor(0.0)
            first_joint_translation_n = torch.tensor(0.0)
            first_joint_translation_u = torch.tensor(0.0)
            first_joint_tilt_e = torch.tensor(0.0)
            first_joint_tilt_n = torch.tensor(0.0)
            first_joint_tilt_u = torch.tensor(0.0)
            second_joint_translation_e = torch.tensor(0.0)
            second_joint_translation_n = torch.tensor(0.0)
            second_joint_translation_u = torch.tensor(0.0)
            second_joint_tilt_e = torch.tensor(0.0)
            second_joint_tilt_n = torch.tensor(0.0)
            second_joint_tilt_u = torch.tensor(0.0)
            concentrator_translation_e = torch.tensor(0.0)
            concentrator_translation_n = torch.tensor(0.0)
            concentrator_translation_u = torch.tensor(0.0)
            concentrator_tilt_e = torch.tensor(0.0)
            concentrator_tilt_n = torch.tensor(0.0)
            concentrator_tilt_u = torch.tensor(0.0)
            if (
                config_dictionary.kinematic_deviations_key
                in config_file[config_dictionary.heliostat_kinematic_key].keys()
            ):
                if (
                    config_dictionary.first_joint_translation_e
                    in config_file[config_dictionary.heliostat_kinematic_key][
                        config_dictionary.kinematic_deviations_key
                    ].keys()
                ):
                    first_joint_translation_e = torch.tensor(
                        config_file[config_dictionary.heliostat_kinematic_key][
                            config_dictionary.kinematic_deviations_key
                        ][config_dictionary.first_joint_translation_e][()],
                        dtype=torch.float,
                    )
                if (
                    config_dictionary.first_joint_translation_n
                    in config_file[config_dictionary.heliostat_kinematic_key][
                        config_dictionary.kinematic_deviations_key
                    ].keys()
                ):
                    first_joint_translation_n = torch.tensor(
                        config_file[config_dictionary.heliostat_kinematic_key][
                            config_dictionary.kinematic_deviations_key
                        ][config_dictionary.first_joint_translation_n][()],
                        dtype=torch.float,
                    )
                if (
                    config_dictionary.first_joint_translation_u
                    in config_file[config_dictionary.heliostat_kinematic_key][
                        config_dictionary.kinematic_deviations_key
                    ].keys()
                ):
                    first_joint_translation_u = torch.tensor(
                        config_file[config_dictionary.heliostat_kinematic_key][
                            config_dictionary.kinematic_deviations_key
                        ][config_dictionary.first_joint_translation_u][()],
                        dtype=torch.float,
                    )
                if (
                    config_dictionary.first_joint_tilt_e
                    in config_file[config_dictionary.heliostat_kinematic_key][
                        config_dictionary.kinematic_deviations_key
                    ].keys()
                ):
                    first_joint_tilt_e = torch.tensor(
                        config_file[config_dictionary.heliostat_kinematic_key][
                            config_dictionary.kinematic_deviations_key
                        ][config_dictionary.first_joint_tilt_e][()],
                        dtype=torch.float,
                    )
                if (
                    config_dictionary.first_joint_tilt_n
                    in config_file[config_dictionary.heliostat_kinematic_key][
                        config_dictionary.kinematic_deviations_key
                    ].keys()
                ):
                    first_joint_tilt_n = torch.tensor(
                        config_file[config_dictionary.heliostat_kinematic_key][
                            config_dictionary.kinematic_deviations_key
                        ][config_dictionary.first_joint_tilt_n][()],
                        dtype=torch.float,
                    )
                if (
                    config_dictionary.first_joint_tilt_u
                    in config_file[config_dictionary.heliostat_kinematic_key][
                        config_dictionary.kinematic_deviations_key
                    ].keys()
                ):
                    first_joint_tilt_u = torch.tensor(
                        config_file[config_dictionary.heliostat_kinematic_key][
                            config_dictionary.kinematic_deviations_key
                        ][config_dictionary.first_joint_tilt_u][()],
                        dtype=torch.float,
                    )
                if (
                    config_dictionary.second_joint_translation_e
                    in config_file[config_dictionary.heliostat_kinematic_key][
                        config_dictionary.kinematic_deviations_key
                    ].keys()
                ):
                    second_joint_translation_e = torch.tensor(
                        config_file[config_dictionary.heliostat_kinematic_key][
                            config_dictionary.kinematic_deviations_key
                        ][config_dictionary.second_joint_translation_e][()],
                        dtype=torch.float,
                    )
                if (
                    config_dictionary.second_joint_translation_n
                    in config_file[config_dictionary.heliostat_kinematic_key][
                        config_dictionary.kinematic_deviations_key
                    ].keys()
                ):
                    second_joint_translation_n = torch.tensor(
                        config_file[config_dictionary.heliostat_kinematic_key][
                            config_dictionary.kinematic_deviations_key
                        ][config_dictionary.second_joint_translation_n][()],
                        dtype=torch.float,
                    )
                if (
                    config_dictionary.second_joint_translation_u
                    in config_file[config_dictionary.heliostat_kinematic_key][
                        config_dictionary.kinematic_deviations_key
                    ].keys()
                ):
                    second_joint_translation_u = torch.tensor(
                        config_file[config_dictionary.heliostat_kinematic_key][
                            config_dictionary.kinematic_deviations_key
                        ][config_dictionary.second_joint_translation_u][()],
                        dtype=torch.float,
                    )
                if (
                    config_dictionary.second_joint_tilt_e
                    in config_file[config_dictionary.heliostat_kinematic_key][
                        config_dictionary.kinematic_deviations_key
                    ].keys()
                ):
                    second_joint_tilt_e = torch.tensor(
                        config_file[config_dictionary.heliostat_kinematic_key][
                            config_dictionary.kinematic_deviations_key
                        ][config_dictionary.second_joint_tilt_e][()],
                        dtype=torch.float,
                    )
                if (
                    config_dictionary.second_joint_tilt_n
                    in config_file[config_dictionary.heliostat_kinematic_key][
                        config_dictionary.kinematic_deviations_key
                    ].keys()
                ):
                    second_joint_tilt_n = torch.tensor(
                        config_file[config_dictionary.heliostat_kinematic_key][
                            config_dictionary.kinematic_deviations_key
                        ][config_dictionary.second_joint_tilt_n][()],
                        dtype=torch.float,
                    )
                if (
                    config_dictionary.second_joint_tilt_u
                    in config_file[config_dictionary.heliostat_kinematic_key][
                        config_dictionary.kinematic_deviations_key
                    ].keys()
                ):
                    second_joint_tilt_u = torch.tensor(
                        config_file[config_dictionary.heliostat_kinematic_key][
                            config_dictionary.kinematic_deviations_key
                        ][config_dictionary.second_joint_tilt_u][()],
                        dtype=torch.float,
                    )
                if (
                    config_dictionary.concentrator_translation_e
                    in config_file[config_dictionary.heliostat_kinematic_key][
                        config_dictionary.kinematic_deviations_key
                    ].keys()
                ):
                    concentrator_translation_e = torch.tensor(
                        config_file[config_dictionary.heliostat_kinematic_key][
                            config_dictionary.kinematic_deviations_key
                        ][config_dictionary.concentrator_translation_e][()],
                        dtype=torch.float,
                    )
                if (
                    config_dictionary.concentrator_translation_n
                    in config_file[config_dictionary.heliostat_kinematic_key][
                        config_dictionary.kinematic_deviations_key
                    ].keys()
                ):
                    concentrator_translation_n = torch.tensor(
                        config_file[config_dictionary.heliostat_kinematic_key][
                            config_dictionary.kinematic_deviations_key
                        ][config_dictionary.concentrator_translation_n][()],
                        dtype=torch.float,
                    )
                if (
                    config_dictionary.concentrator_translation_u
                    in config_file[config_dictionary.heliostat_kinematic_key][
                        config_dictionary.kinematic_deviations_key
                    ].keys()
                ):
                    concentrator_translation_u = torch.tensor(
                        config_file[config_dictionary.heliostat_kinematic_key][
                            config_dictionary.kinematic_deviations_key
                        ][config_dictionary.concentrator_translation_u][()],
                        dtype=torch.float,
                    )
                if (
                    config_dictionary.concentrator_tilt_e
                    in config_file[config_dictionary.heliostat_kinematic_key][
                        config_dictionary.kinematic_deviations_key
                    ].keys()
                ):
                    concentrator_tilt_e = torch.tensor(
                        config_file[config_dictionary.heliostat_kinematic_key][
                            config_dictionary.kinematic_deviations_key
                        ][config_dictionary.concentrator_tilt_e][()],
                        dtype=torch.float,
                    )
                if (
                    config_dictionary.concentrator_tilt_n
                    in config_file[config_dictionary.heliostat_kinematic_key][
                        config_dictionary.kinematic_deviations_key
                    ].keys()
                ):
                    concentrator_tilt_n = torch.tensor(
                        config_file[config_dictionary.heliostat_kinematic_key][
                            config_dictionary.kinematic_deviations_key
                        ][config_dictionary.concentrator_tilt_n][()],
                        dtype=torch.float,
                    )
                if (
                    config_dictionary.concentrator_tilt_u
                    in config_file[config_dictionary.heliostat_kinematic_key][
                        config_dictionary.kinematic_deviations_key
                    ].keys()
                ):
                    concentrator_tilt_u = torch.tensor(
                        config_file[config_dictionary.heliostat_kinematic_key][
                            config_dictionary.kinematic_deviations_key
                        ][config_dictionary.concentrator_tilt_u][()],
                        dtype=torch.float,
                    )
            kinematic_deviations = KinematicDeviations(
                first_joint_translation_e=first_joint_translation_e,
                first_joint_translation_n=first_joint_translation_n,
                first_joint_translation_u=first_joint_translation_u,
                first_joint_tilt_e=first_joint_tilt_e,
                first_joint_tilt_n=first_joint_tilt_n,
                first_joint_tilt_u=first_joint_tilt_u,
                second_joint_translation_e=second_joint_translation_e,
                second_joint_translation_n=second_joint_translation_n,
                second_joint_translation_u=second_joint_translation_u,
                second_joint_tilt_e=second_joint_tilt_e,
                second_joint_tilt_n=second_joint_tilt_n,
                second_joint_tilt_u=second_joint_tilt_u,
                concentrator_translation_e=concentrator_translation_e,
                concentrator_translation_n=concentrator_translation_n,
                concentrator_translation_u=concentrator_translation_u,
                concentrator_tilt_e=concentrator_tilt_e,
                concentrator_tilt_n=concentrator_tilt_n,
                concentrator_tilt_u=concentrator_tilt_u,
            )
            kinematic_config = KinematicConfig(
                kinematic_type=str(
                    config_file[config_dictionary.heliostat_kinematic_key][
                        config_dictionary.kinematic_type
                    ][()].decode("utf-8")
                ),
                kinematic_initial_orientation_offsets=kinematic_offsets,
                kinematic_deviations=kinematic_deviations,
            )
        else:
            assert (
                prototype_kinematic is not None
            ), "If the heliostat does not have an individual kinematic then a kinematic prototype must be provided!"
            kinematic_config = prototype_kinematic

        if config_dictionary.heliostat_actuator_key in config_file.keys():
            actuator_list = []
            for ac in config_file[config_dictionary.heliostat_actuator_key].keys():
                increment = torch.tensor(0.0)
                initial_stroke_length = torch.tensor(0.0)
                offset = torch.tensor(0.0)
                radius = torch.tensor(0.0)
                phi_0 = torch.tensor(0.0)
                if (
                    config_dictionary.actuator_parameters_key
                    in config_file[config_dictionary.heliostat_actuator_key][ac].keys()
                ):
                    if (
                        config_dictionary.actuator_increment
                        in config_file[config_dictionary.heliostat_actuator_key][ac][
                            config_dictionary.actuator_parameters_key
                        ].keys()
                    ):
                        increment = torch.tensor(
                            config_file[config_dictionary.heliostat_actuator_key][ac][
                                config_dictionary.actuator_parameters_key
                            ][config_dictionary.actuator_increment][()],
                            dtype=torch.float,
                        )
                    if (
                        config_dictionary.actuator_initial_stroke_length
                        in config_file[config_dictionary.heliostat_actuator_key][ac][
                            config_dictionary.actuator_parameters_key
                        ].keys()
                    ):
                        initial_stroke_length = torch.tensor(
                            config_file[config_dictionary.heliostat_actuator_key][ac][
                                config_dictionary.actuator_parameters_key
                            ][config_dictionary.actuator_initial_stroke_length][()],
                            dtype=torch.float,
                        )
                    if (
                        config_dictionary.actuator_offset
                        in config_file[config_dictionary.heliostat_actuator_key][ac][
                            config_dictionary.actuator_parameters_key
                        ].keys()
                    ):
                        offset = torch.tensor(
                            config_file[config_dictionary.heliostat_actuator_key][ac][
                                config_dictionary.actuator_parameters_key
                            ][config_dictionary.actuator_offset][()],
                            dtype=torch.float,
                        )
                    if (
                        config_dictionary.actuator_radius
                        in config_file[config_dictionary.heliostat_actuator_key][ac][
                            config_dictionary.actuator_parameters_key
                        ].keys()
                    ):
                        radius = torch.tensor(
                            config_file[config_dictionary.heliostat_actuator_key][ac][
                                config_dictionary.actuator_parameters_key
                            ][config_dictionary.actuator_radius][()],
                            dtype=torch.float,
                        )
                    if (
                        config_dictionary.actuator_phi_0
                        in config_file[config_dictionary.heliostat_actuator_key][ac][
                            config_dictionary.actuator_parameters_key
                        ].keys()
                    ):
                        phi_0 = torch.tensor(
                            config_file[config_dictionary.heliostat_actuator_key][ac][
                                config_dictionary.actuator_parameters_key
                            ][config_dictionary.actuator_phi_0][()],
                            dtype=torch.float,
                        )

                actuator_parameters = ActuatorParameters(
                    increment=increment,
                    initial_stroke_length=initial_stroke_length,
                    offset=offset,
                    radius=radius,
                    phi_0=phi_0,
                )
                actuator_list.append(
                    ActuatorConfig(
                        actuator_key="",
                        actuator_type=str(
                            config_file[config_dictionary.heliostat_actuator_key][ac][
                                config_dictionary.actuator_type_key
                            ][()].decode("utf-8")
                        ),
                        actuator_clockwise=bool(
                            config_file[config_dictionary.heliostat_actuator_key][ac][
                                config_dictionary.actuator_clockwise
                            ][()]
                        ),
                        actuator_parameters=actuator_parameters,
                    )
                )
            actuator_list_config = ActuatorListConfig(actuator_list=actuator_list)
        else:
            assert (
                prototype_actuator is not None
            ), "If the heliostat does not have individual actuators then a actuator prototype must be provided!"
            actuator_list_config = prototype_actuator

        return cls(
            heliostat_id=heliostat_id,
            position=position,
            aim_point=aim_point,
            surface_config=surface_config,
            kinematic_config=kinematic_config,
            actuator_config=actuator_list_config,
        )

    def set_aligned_surface(self, incident_ray_direction: torch.Tensor) -> None:
        """
        Compute the aligned surface points and aligned surface normals of the heliostat.

        Parameters
        ----------
        incident_ray_direction : torch.Tensor
            The incident ray direction.
        """
        surface_points, surface_normals = (
            self.concentrator.facets.surface_points,
            self.concentrator.facets.surface_normals,
        )
        (
            self.current_aligned_surface_points,
            self.current_aligned_surface_normals,
        ) = self.alignment.align_surface(
            incident_ray_direction, surface_points, surface_normals
        )
        self.is_aligned = True

    def set_preferred_reflection_direction(self, rays: torch.Tensor) -> None:
        """
        Reflect incoming rays according to a normal vector.

        Parameters
        ----------
        rays : torch.Tensor
            The incoming rays (from the sun) to be reflected.

        Raises
        ------
        AssertionError
            If the heliostat has not yet been aligned.
        """
        assert self.is_aligned, "Heliostat has not yet been aligned."

        self.preferred_reflection_direction = (
            rays
            - 2
            * utils.batch_dot(rays, self.current_aligned_surface_normals)
            * self.current_aligned_surface_normals
        )
