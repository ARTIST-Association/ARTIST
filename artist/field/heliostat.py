import logging
from typing import Optional

import h5py
import torch
from typing_extensions import Self

from artist.field.kinematic_rigid_body import RigidBody
from artist.field.surface import Surface
from artist.raytracing.raytracing_utils import reflect
from artist.util import config_dictionary
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

kinematic_type_mapping = {config_dictionary.rigid_body_key: RigidBody}
"""A type mapping dictionary that allows ARTIST to automatically infer the correct kinematic type."""

log = logging.getLogger(__name__)
"""A logger for the heliostat."""


class Heliostat(torch.nn.Module):
    """
    Implements the behavior of a heliostat.

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
        Class method to initialize a heliostat from an HDF5 file.
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
        Implement the behavior of a heliostat.

        A Heliostat is used to reflect light onto the receiver. A Heliostat has a position within the field and an
        aim point where it aims to reflect the light. Furthermore, each heliostat must be initialized with a surface
        configuration which contains information on the heliostat surface, a kinematic configuration containing
        information on the applied kinematic, and an actuator configuration that contains the configurations of the
        actuators used in the heliostat.

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
            kinematic_object = kinematic_type_mapping[kinematic_config.kinematic_type]
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
        heliostat_name: Optional[str] = None,
    ) -> Self:
        """
        Class method to initialize a heliostat from an HDF5 file.

        Parameters
        ----------
        config_file : h5py.File
            The config file containing all the information about the heliostat.
        prototype_surface  : SurfaceConfig, optional
            An optional prototype for the surface configuration.
        prototype_kinematic : KinematicConfig, optional
            An optional prototype for the kinematic configuration.
        prototype_actuator : ActuatorConfig, optional
            An optional prototype for the actuator configuration.
        heliostat_name : str, optional
            The name of the heliostat being loaded - used for logging.

        Returns
        -------
        Heliostat
            A heliostat initialized from an HDF5 file.
        """
        if heliostat_name:
            log.info(f"Loading {heliostat_name} from an HDF5 file.")
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
                    control_points=torch.tensor(
                        config_file[config_dictionary.heliostat_surface_key][
                            config_dictionary.facets_key
                        ][facet][config_dictionary.facet_control_points][()],
                        dtype=torch.float,
                    ),
                    degree_e=int(
                        config_file[config_dictionary.heliostat_surface_key][
                            config_dictionary.facets_key
                        ][facet][config_dictionary.facet_degree_e][()]
                    ),
                    degree_n=int(
                        config_file[config_dictionary.heliostat_surface_key][
                            config_dictionary.facets_key
                        ][facet][config_dictionary.facet_degree_n][()]
                    ),
                    number_eval_points_e=int(
                        config_file[config_dictionary.heliostat_surface_key][
                            config_dictionary.facets_key
                        ][facet][config_dictionary.facet_number_eval_e][()]
                    ),
                    number_eval_points_n=int(
                        config_file[config_dictionary.heliostat_surface_key][
                            config_dictionary.facets_key
                        ][facet][config_dictionary.facet_number_eval_n][()]
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
                    translation_vector=torch.tensor(
                        config_file[config_dictionary.heliostat_surface_key][
                            config_dictionary.facets_key
                        ][facet][config_dictionary.facets_translation_vector][()],
                        dtype=torch.float,
                    ),
                    canting_e=torch.tensor(
                        config_file[config_dictionary.heliostat_surface_key][
                            config_dictionary.facets_key
                        ][facet][config_dictionary.facets_canting_e][()],
                        dtype=torch.float,
                    ),
                    canting_n=torch.tensor(
                        config_file[config_dictionary.heliostat_surface_key][
                            config_dictionary.facets_key
                        ][facet][config_dictionary.facets_canting_n][()],
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
            ), "If the heliostat does not have individual surface parameters, a surface prototype must be provided!"
            log.info(
                "Individual surface parameters not provided - loading a heliostat with the surface prototype."
            )
            surface_config = prototype_surface

        if config_dictionary.heliostat_kinematic_key in config_file.keys():
            kinematic_initial_orientation_offset_e = config_file.get(
                f"{config_dictionary.heliostat_kinematic_key}/"
                f"{config_dictionary.kinematic_offsets_key}/{config_dictionary.kinematic_initial_orientation_offset_e}"
            )
            kinematic_initial_orientation_offset_n = config_file.get(
                f"{config_dictionary.heliostat_kinematic_key}/"
                f"{config_dictionary.kinematic_offsets_key}/{config_dictionary.kinematic_initial_orientation_offset_n}"
            )
            kinematic_initial_orientation_offset_u = config_file.get(
                f"{config_dictionary.heliostat_kinematic_key}/"
                f"{config_dictionary.kinematic_offsets_key}/{config_dictionary.kinematic_initial_orientation_offset_n}"
            )
            if kinematic_initial_orientation_offset_e is None:
                log.warning(
                    f"No individual kinematic {config_dictionary.kinematic_initial_orientation_offset_e} for "
                    f"{heliostat_name} set."
                    f"Using default values!"
                )
            if kinematic_initial_orientation_offset_n is None:
                log.warning(
                    f"No individual kinematic {config_dictionary.kinematic_initial_orientation_offset_n} for "
                    f"{heliostat_name} set."
                    f"Using default values!"
                )
            if kinematic_initial_orientation_offset_u is None:
                log.warning(
                    f"No individual kinematic {config_dictionary.kinematic_initial_orientation_offset_u} for "
                    f"{heliostat_name} set."
                    f"Using default values!"
                )
            kinematic_offsets = KinematicOffsets(
                kinematic_initial_orientation_offset_e=(
                    torch.tensor(
                        kinematic_initial_orientation_offset_e[()], dtype=torch.float
                    )
                    if kinematic_initial_orientation_offset_e
                    else torch.tensor(0.0)
                ),
                kinematic_initial_orientation_offset_n=(
                    torch.tensor(
                        kinematic_initial_orientation_offset_n[()], dtype=torch.float
                    )
                    if kinematic_initial_orientation_offset_n
                    else torch.tensor(0.0)
                ),
                kinematic_initial_orientation_offset_u=(
                    torch.tensor(
                        kinematic_initial_orientation_offset_u[()], dtype=torch.float
                    )
                    if kinematic_initial_orientation_offset_u
                    else torch.tensor(0.0)
                ),
            )

            first_joint_translation_e = config_file.get(
                f"{config_dictionary.heliostat_kinematic_key}/"
                f"{config_dictionary.kinematic_deviations_key}/"
                f"{config_dictionary.first_joint_translation_e}"
            )
            first_joint_translation_n = config_file.get(
                f"{config_dictionary.heliostat_kinematic_key}/"
                f"{config_dictionary.kinematic_deviations_key}/"
                f"{config_dictionary.first_joint_translation_n}"
            )
            first_joint_translation_u = config_file.get(
                f"{config_dictionary.heliostat_kinematic_key}/"
                f"{config_dictionary.kinematic_deviations_key}/"
                f"{config_dictionary.first_joint_translation_u}"
            )
            first_joint_tilt_e = config_file.get(
                f"{config_dictionary.heliostat_kinematic_key}/"
                f"{config_dictionary.kinematic_deviations_key}/"
                f"{config_dictionary.first_joint_tilt_e}"
            )
            first_joint_tilt_n = config_file.get(
                f"{config_dictionary.heliostat_kinematic_key}/"
                f"{config_dictionary.kinematic_deviations_key}/"
                f"{config_dictionary.first_joint_tilt_n}"
            )
            first_joint_tilt_u = config_file.get(
                f"{config_dictionary.heliostat_kinematic_key}/"
                f"{config_dictionary.kinematic_deviations_key}/"
                f"{config_dictionary.first_joint_tilt_u}"
            )
            second_joint_translation_e = config_file.get(
                f"{config_dictionary.heliostat_kinematic_key}/"
                f"{config_dictionary.kinematic_deviations_key}/"
                f"{config_dictionary.second_joint_translation_e}"
            )
            second_joint_translation_n = config_file.get(
                f"{config_dictionary.heliostat_kinematic_key}/"
                f"{config_dictionary.kinematic_deviations_key}/"
                f"{config_dictionary.second_joint_translation_n}"
            )
            second_joint_translation_u = config_file.get(
                f"{config_dictionary.heliostat_kinematic_key}/"
                f"{config_dictionary.kinematic_deviations_key}/"
                f"{config_dictionary.second_joint_translation_u}"
            )
            second_joint_tilt_e = config_file.get(
                f"{config_dictionary.heliostat_kinematic_key}/"
                f"{config_dictionary.kinematic_deviations_key}/"
                f"{config_dictionary.second_joint_tilt_e}"
            )
            second_joint_tilt_n = config_file.get(
                f"{config_dictionary.heliostat_kinematic_key}/"
                f"{config_dictionary.kinematic_deviations_key}/"
                f"{config_dictionary.second_joint_tilt_n}"
            )
            second_joint_tilt_u = config_file.get(
                f"{config_dictionary.heliostat_kinematic_key}/"
                f"{config_dictionary.kinematic_deviations_key}/"
                f"{config_dictionary.second_joint_tilt_u}"
            )
            concentrator_translation_e = config_file.get(
                f"{config_dictionary.heliostat_kinematic_key}/"
                f"{config_dictionary.kinematic_deviations_key}/"
                f"{config_dictionary.concentrator_translation_e}"
            )
            concentrator_translation_n = config_file.get(
                f"{config_dictionary.heliostat_kinematic_key}/"
                f"{config_dictionary.kinematic_deviations_key}/"
                f"{config_dictionary.concentrator_translation_n}"
            )
            concentrator_translation_u = config_file.get(
                f"{config_dictionary.heliostat_kinematic_key}/"
                f"{config_dictionary.kinematic_deviations_key}/"
                f"{config_dictionary.concentrator_translation_u}"
            )
            concentrator_tilt_e = config_file.get(
                f"{config_dictionary.heliostat_kinematic_key}/"
                f"{config_dictionary.kinematic_deviations_key}/"
                f"{config_dictionary.concentrator_tilt_e}"
            )
            concentrator_tilt_n = config_file.get(
                f"{config_dictionary.heliostat_kinematic_key}/"
                f"{config_dictionary.kinematic_deviations_key}/"
                f"{config_dictionary.concentrator_tilt_n}"
            )
            concentrator_tilt_u = config_file.get(
                f"{config_dictionary.heliostat_kinematic_key}/"
                f"{config_dictionary.kinematic_deviations_key}/"
                f"{config_dictionary.concentrator_tilt_u}"
            )
            if first_joint_translation_e is None:
                log.warning(
                    f"No individual kinematic {config_dictionary.first_joint_translation_e} for {heliostat_name} set. "
                    f"Using default values!"
                )
            if first_joint_translation_n is None:
                log.warning(
                    f"No individual kinematic {config_dictionary.first_joint_translation_n} for {heliostat_name} set. "
                    f"Using default values!"
                )
            if first_joint_translation_u is None:
                log.warning(
                    f"No individual kinematic {config_dictionary.first_joint_translation_u} for {heliostat_name} set. "
                    f"Using default values!"
                )
            if first_joint_tilt_e is None:
                log.warning(
                    f"No individual kinematic {config_dictionary.first_joint_tilt_e} for {heliostat_name} set. "
                    f"Using default values!"
                )
            if first_joint_tilt_n is None:
                log.warning(
                    f"No individual kinematic {config_dictionary.first_joint_tilt_n} for {heliostat_name} set. "
                    f"Using default values!"
                )
            if first_joint_tilt_u is None:
                log.warning(
                    f"No individual kinematic {config_dictionary.first_joint_tilt_u} for {heliostat_name} set. "
                    f"Using default values!"
                )
            if second_joint_translation_e is None:
                log.warning(
                    f"No individual kinematic {config_dictionary.second_joint_translation_e} for {heliostat_name} set. "
                    f"Using default values!"
                )
            if second_joint_translation_n is None:
                log.warning(
                    f"No individual kinematic {config_dictionary.second_joint_translation_n} for {heliostat_name} set. "
                    f"Using default values!"
                )
            if second_joint_translation_u is None:
                log.warning(
                    f"No individual kinematic {config_dictionary.second_joint_translation_u} for {heliostat_name} set. "
                    f"Using default values!"
                )
            if second_joint_tilt_e is None:
                log.warning(
                    f"No individual kinematic {config_dictionary.second_joint_tilt_e} for {heliostat_name} set. "
                    f"Using default values!"
                )
            if second_joint_tilt_n is None:
                log.warning(
                    f"No individual kinematic {config_dictionary.second_joint_tilt_n} for {heliostat_name} set. "
                    f"Using default values!"
                )
            if second_joint_tilt_u is None:
                log.warning(
                    f"No individual kinematic {config_dictionary.second_joint_tilt_u} for {heliostat_name} set. "
                    f"Using default values!"
                )
            if concentrator_translation_e is None:
                log.warning(
                    f"No individual kinematic {config_dictionary.concentrator_translation_e} for {heliostat_name} set. "
                    f"Using default values!"
                )
            if concentrator_translation_n is None:
                log.warning(
                    f"No individual kinematic {config_dictionary.concentrator_translation_n} for {heliostat_name} set. "
                    f"Using default values!"
                )
            if concentrator_translation_u is None:
                log.warning(
                    f"No individual kinematic {config_dictionary.concentrator_translation_u} for {heliostat_name} set. "
                    f"Using default values!"
                )
            if concentrator_tilt_e is None:
                log.warning(
                    f"No individual kinematic {config_dictionary.concentrator_tilt_e} for {heliostat_name} set. "
                    f"Using default values!"
                )
            if concentrator_tilt_n is None:
                log.warning(
                    f"No individual kinematic {config_dictionary.concentrator_tilt_n} for {heliostat_name} set. "
                    f"Using default values!"
                )
            if concentrator_tilt_u is None:
                log.warning(
                    f"No individual kinematic {config_dictionary.concentrator_tilt_u} for {heliostat_name} set. "
                    f"Using default values!"
                )
            kinematic_deviations = KinematicDeviations(
                first_joint_translation_e=(
                    torch.tensor(first_joint_translation_e[()], dtype=torch.float)
                    if first_joint_translation_e
                    else torch.tensor(0.0)
                ),
                first_joint_translation_n=(
                    torch.tensor(first_joint_translation_n[()], dtype=torch.float)
                    if first_joint_translation_n
                    else torch.tensor(0.0)
                ),
                first_joint_translation_u=(
                    torch.tensor(first_joint_translation_u[()], dtype=torch.float)
                    if first_joint_translation_u
                    else torch.tensor(0.0)
                ),
                first_joint_tilt_e=(
                    torch.tensor(first_joint_tilt_e[()], dtype=torch.float)
                    if first_joint_tilt_e
                    else torch.tensor(0.0)
                ),
                first_joint_tilt_n=(
                    torch.tensor(first_joint_tilt_n[()], dtype=torch.float)
                    if first_joint_tilt_n
                    else torch.tensor(0.0)
                ),
                first_joint_tilt_u=(
                    torch.tensor(first_joint_tilt_u[()], dtype=torch.float)
                    if first_joint_tilt_u
                    else torch.tensor(0.0)
                ),
                second_joint_translation_e=(
                    torch.tensor(second_joint_translation_e[()], dtype=torch.float)
                    if second_joint_translation_e
                    else torch.tensor(0.0)
                ),
                second_joint_translation_n=(
                    torch.tensor(second_joint_translation_n[()], dtype=torch.float)
                    if second_joint_translation_n
                    else torch.tensor(0.0)
                ),
                second_joint_translation_u=(
                    torch.tensor(second_joint_translation_u[()], dtype=torch.float)
                    if second_joint_translation_u
                    else torch.tensor(0.0)
                ),
                second_joint_tilt_e=(
                    torch.tensor(second_joint_tilt_e[()], dtype=torch.float)
                    if second_joint_tilt_e
                    else torch.tensor(0.0)
                ),
                second_joint_tilt_n=(
                    torch.tensor(second_joint_tilt_n[()], dtype=torch.float)
                    if second_joint_tilt_n
                    else torch.tensor(0.0)
                ),
                second_joint_tilt_u=(
                    torch.tensor(second_joint_tilt_u[()], dtype=torch.float)
                    if second_joint_tilt_u
                    else torch.tensor(0.0)
                ),
                concentrator_translation_e=(
                    torch.tensor(concentrator_translation_e[()], dtype=torch.float)
                    if concentrator_translation_e
                    else torch.tensor(0.0)
                ),
                concentrator_translation_n=(
                    torch.tensor(concentrator_translation_n[()], dtype=torch.float)
                    if concentrator_translation_n
                    else torch.tensor(0.0)
                ),
                concentrator_translation_u=(
                    torch.tensor(concentrator_translation_u[()], dtype=torch.float)
                    if concentrator_translation_u
                    else torch.tensor(0.0)
                ),
                concentrator_tilt_e=(
                    torch.tensor(concentrator_tilt_e[()], dtype=torch.float)
                    if concentrator_tilt_e
                    else torch.tensor(0.0)
                ),
                concentrator_tilt_n=(
                    torch.tensor(concentrator_tilt_n[()], dtype=torch.float)
                    if concentrator_tilt_n
                    else torch.tensor(0.0)
                ),
                concentrator_tilt_u=(
                    torch.tensor(concentrator_tilt_u[()], dtype=torch.float)
                    if concentrator_tilt_u
                    else torch.tensor(0.0)
                ),
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
            ), "If the heliostat does not have an individual kinematic, a kinematic prototype must be provided!"
            log.info(
                "Individual kinematic configuration not provided - loading a heliostat with the kinematic prototype."
            )
            kinematic_config = prototype_kinematic

        if config_dictionary.heliostat_actuator_key in config_file.keys():
            actuator_list = []
            for ac in config_file[config_dictionary.heliostat_actuator_key].keys():
                increment = config_file.get(
                    f"{config_dictionary.heliostat_actuator_key}/{ac}/"
                    f"{config_dictionary.actuator_parameters_key}/"
                    f"{config_dictionary.actuator_increment}"
                )
                initial_stroke_length = config_file.get(
                    f"{config_dictionary.heliostat_actuator_key}/{ac}/"
                    f"{config_dictionary.actuator_parameters_key}/"
                    f"{config_dictionary.actuator_initial_stroke_length}"
                )
                offset = config_file.get(
                    f"{config_dictionary.heliostat_actuator_key}/{ac}/"
                    f"{config_dictionary.actuator_parameters_key}/"
                    f"{config_dictionary.actuator_offset}"
                )
                radius = config_file.get(
                    f"{config_dictionary.heliostat_actuator_key}/{ac}/"
                    f"{config_dictionary.actuator_parameters_key}/"
                    f"{config_dictionary.actuator_radius}"
                )
                phi_0 = config_file.get(
                    f"{config_dictionary.heliostat_actuator_key}/{ac}/"
                    f"{config_dictionary.actuator_parameters_key}/"
                    f"{config_dictionary.actuator_phi_0}"
                )
                if increment is None:
                    log.warning(
                        f"No individual {config_dictionary.actuator_increment} set for {ac}. Using default values!"
                    )
                if initial_stroke_length is None:
                    log.warning(
                        f"No individual {config_dictionary.actuator_initial_stroke_length} set for {ac} on "
                        f"{heliostat_name}. "
                        f"Using default values!"
                    )
                if offset is None:
                    log.warning(
                        f"No individual {config_dictionary.actuator_offset} set for {ac} on "
                        f"{heliostat_name}. Using default values!"
                    )
                if radius is None:
                    log.warning(
                        f"No individual {config_dictionary.actuator_radius} set for {ac} on "
                        f"{heliostat_name}. Using default values!"
                    )
                if phi_0 is None:
                    log.warning(
                        f"No individual {config_dictionary.actuator_phi_0} set for {ac} on "
                        f"{heliostat_name}. Using default values!"
                    )
                actuator_parameters = ActuatorParameters(
                    increment=(
                        torch.tensor(increment[()], dtype=torch.float)
                        if increment
                        else torch.tensor(0.0)
                    ),
                    initial_stroke_length=(
                        torch.tensor(initial_stroke_length[()], dtype=torch.float)
                        if initial_stroke_length
                        else torch.tensor(0.0)
                    ),
                    offset=(
                        torch.tensor(offset[()], dtype=torch.float)
                        if offset
                        else torch.tensor(0.0)
                    ),
                    radius=(
                        torch.tensor(radius[()], dtype=torch.float)
                        if radius
                        else torch.tensor(0.0)
                    ),
                    phi_0=(
                        torch.tensor(phi_0[()], dtype=torch.float)
                        if phi_0
                        else torch.tensor(0.0)
                    ),
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
            ), "If the heliostat does not have individual actuators, an actuator prototype must be provided!"
            log.info(
                "Individual actuator configurations not provided - loading a heliostat with the actuator prototype."
            )
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
        surface_points, surface_normals = self.surface.get_surface_points_and_normals()
        (
            self.current_aligned_surface_points,
            self.current_aligned_surface_normals,
        ) = self.kinematic.align_surface(
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

        self.preferred_reflection_direction = reflect(
            incoming_ray_direction=rays,
            reflection_surface_normals=self.current_aligned_surface_normals,
        )
