import logging
from typing import Optional, Union

import h5py
import torch
from typing_extensions import Self

from artist.field.kinematic_rigid_body import RigidBody
from artist.field.surface import Surface
from artist.raytracing.raytracing_utils import reflect
from artist.util import config_dictionary, utils
from artist.util.configuration_classes import (
    ActuatorConfig,
    ActuatorListConfig,
    ActuatorParameters,
    FacetConfig,
    KinematicDeviations,
    KinematicLoadConfig,
    SurfaceConfig,
)

kinematic_type_mapping = {config_dictionary.rigid_body_key: RigidBody}
"""A type mapping dictionary that allows ``ARTIST`` to automatically infer the correct kinematic type."""

log = logging.getLogger(__name__)
"""A logger for the heliostat."""


class Heliostat(torch.nn.Module):
    """
    Implement the behavior of a heliostat.

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
    surface_points : torch.Tensor
        The original, unaligned surface points.
    surface_normals : torch.Tensor
        The original, unaligned surface normals.

    Methods
    -------
    from_hdf5()
        Class method to initialize a heliostat from an HDF5 file.
    set_aligned_surface_with_incident_ray_direction()
        Compute the aligned surface points and aligned surface normals of the heliostat.
    set_aligned_surface_with_motor_positions()
        Compute the aligned surface points and aligned surface normals of the heliostat.
    get_orientation_from_motor_positions()
        Compute the orientation for a heliostat given the desired motor positions.
    set_preferred_reflection_direction()
        Reflect incoming rays according to a normal vector.
    forward()
        Specify the forward pass.
    """

    def __init__(
        self,
        heliostat_id: int,
        position: torch.Tensor,
        aim_point: torch.Tensor,
        surface_config: SurfaceConfig,
        kinematic_config: KinematicLoadConfig,
        actuator_config: ActuatorListConfig,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """
        Implement the behavior of a heliostat.

        A heliostat is used to reflect light onto the receiver. A heliostat has a position within the field and an
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
        kinematic_config : KinematicLoadConfig
            The configuration parameters to use for the heliostat kinematic.
        actuator_config : ActuatorListConfig
            The configuration parameters to use for the list of actuators.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).
        """
        super().__init__()
        device = torch.device(device)
        self.heliostat_id = heliostat_id
        self.position = position
        self.aim_point = aim_point
        self.surface = Surface(surface_config=surface_config)
        try:
            kinematic_object = kinematic_type_mapping[kinematic_config.type]
        except KeyError:
            raise KeyError(
                f"Currently the selected kinematic type: {kinematic_config.type} is not supported."
            )
        self.kinematic = kinematic_object(
            position=position,
            aim_point=aim_point,
            actuator_config=actuator_config,
            initial_orientation=kinematic_config.initial_orientation,
            deviation_parameters=kinematic_config.deviations,
            device=device,
        )
        self.current_aligned_surface_points = torch.empty(0, device=device)
        self.current_aligned_surface_normals = torch.empty(0, device=device)
        self.is_aligned = False
        self.preferred_reflection_direction = torch.empty(0, device=device)

        self.surface_points, self.surface_normals = (
            self.surface.get_surface_points_and_normals(device=device)
        )

    @classmethod
    def from_hdf5(
        cls,
        config_file: h5py.File,
        prototype_surface: Optional[SurfaceConfig] = None,
        prototype_kinematic: Optional[KinematicLoadConfig] = None,
        prototype_actuator_list: Optional[ActuatorListConfig] = None,
        heliostat_name: Optional[str] = None,
        device: Union[torch.device, str] = "cuda",
    ) -> Self:
        """
        Class method to initialize a heliostat from an HDF5 file.

        Parameters
        ----------
        config_file : h5py.File
            The config file containing all the information about the heliostat.
        prototype_surface  : SurfaceConfig, optional
            An optional prototype for the surface configuration.
        prototype_kinematic : KinematicLoadConfig, optional
            An optional prototype for the kinematic configuration.
        prototype_actuator_list : ActuatorListConfig, optional
            An optional prototype for the actuator configuration.
        heliostat_name : str, optional
            The name of the heliostat being loaded - used for logging.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        Heliostat
            A heliostat initialized from an HDF5 file.
        """
        if heliostat_name:
            log.info(f"Loading {heliostat_name} from an HDF5 file.")
        device = torch.device(device)
        heliostat_id = int(config_file[config_dictionary.heliostat_id][()])
        position = torch.tensor(
            config_file[config_dictionary.heliostat_position][()],
            dtype=torch.float,
            device=device,
        )
        aim_point = torch.tensor(
            config_file[config_dictionary.heliostat_aim_point][()],
            dtype=torch.float,
            device=device,
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
                        device=device,
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
                    translation_vector=torch.tensor(
                        config_file[config_dictionary.heliostat_surface_key][
                            config_dictionary.facets_key
                        ][facet][config_dictionary.facets_translation_vector][()],
                        dtype=torch.float,
                        device=device,
                    ),
                    canting_e=torch.tensor(
                        config_file[config_dictionary.heliostat_surface_key][
                            config_dictionary.facets_key
                        ][facet][config_dictionary.facets_canting_e][()],
                        dtype=torch.float,
                        device=device,
                    ),
                    canting_n=torch.tensor(
                        config_file[config_dictionary.heliostat_surface_key][
                            config_dictionary.facets_key
                        ][facet][config_dictionary.facets_canting_n][()],
                        dtype=torch.float,
                        device=device,
                    ),
                )
                for facet in config_file[config_dictionary.heliostat_surface_key][
                    config_dictionary.facets_key
                ].keys()
            ]
            surface_config = SurfaceConfig(facets_list=facets_list)
        else:
            if prototype_surface is None:
                raise ValueError(
                    "If the heliostat does not have individual surface parameters, a surface prototype must be provided!"
                )
            log.info(
                "Individual surface parameters not provided - loading a heliostat with the surface prototype."
            )
            surface_config = prototype_surface

        if config_dictionary.heliostat_kinematic_key in config_file.keys():
            initial_orientation = torch.tensor(
                config_file[config_dictionary.heliostat_kinematic_key][
                    config_dictionary.kinematic_initial_orientation
                ][()],
                dtype=torch.float,
                device=device,
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
                    torch.tensor(
                        first_joint_translation_e[()], dtype=torch.float, device=device
                    )
                    if first_joint_translation_e
                    else torch.tensor(0.0, dtype=torch.float, device=device)
                ),
                first_joint_translation_n=(
                    torch.tensor(
                        first_joint_translation_n[()], dtype=torch.float, device=device
                    )
                    if first_joint_translation_n
                    else torch.tensor(0.0, dtype=torch.float, device=device)
                ),
                first_joint_translation_u=(
                    torch.tensor(
                        first_joint_translation_u[()], dtype=torch.float, device=device
                    )
                    if first_joint_translation_u
                    else torch.tensor(0.0, dtype=torch.float, device=device)
                ),
                first_joint_tilt_e=(
                    torch.tensor(
                        first_joint_tilt_e[()], dtype=torch.float, device=device
                    )
                    if first_joint_tilt_e
                    else torch.tensor(0.0, dtype=torch.float, device=device)
                ),
                first_joint_tilt_n=(
                    torch.tensor(
                        first_joint_tilt_n[()], dtype=torch.float, device=device
                    )
                    if first_joint_tilt_n
                    else torch.tensor(0.0, dtype=torch.float, device=device)
                ),
                first_joint_tilt_u=(
                    torch.tensor(
                        first_joint_tilt_u[()], dtype=torch.float, device=device
                    )
                    if first_joint_tilt_u
                    else torch.tensor(0.0, dtype=torch.float, device=device)
                ),
                second_joint_translation_e=(
                    torch.tensor(
                        second_joint_translation_e[()], dtype=torch.float, device=device
                    )
                    if second_joint_translation_e
                    else torch.tensor(0.0, dtype=torch.float, device=device)
                ),
                second_joint_translation_n=(
                    torch.tensor(
                        second_joint_translation_n[()], dtype=torch.float, device=device
                    )
                    if second_joint_translation_n
                    else torch.tensor(0.0, dtype=torch.float, device=device)
                ),
                second_joint_translation_u=(
                    torch.tensor(
                        second_joint_translation_u[()], dtype=torch.float, device=device
                    )
                    if second_joint_translation_u
                    else torch.tensor(0.0, dtype=torch.float, device=device)
                ),
                second_joint_tilt_e=(
                    torch.tensor(
                        second_joint_tilt_e[()], dtype=torch.float, device=device
                    )
                    if second_joint_tilt_e
                    else torch.tensor(0.0, dtype=torch.float, device=device)
                ),
                second_joint_tilt_n=(
                    torch.tensor(
                        second_joint_tilt_n[()], dtype=torch.float, device=device
                    )
                    if second_joint_tilt_n
                    else torch.tensor(0.0, dtype=torch.float, device=device)
                ),
                second_joint_tilt_u=(
                    torch.tensor(
                        second_joint_tilt_u[()], dtype=torch.float, device=device
                    )
                    if second_joint_tilt_u
                    else torch.tensor(0.0, dtype=torch.float, device=device)
                ),
                concentrator_translation_e=(
                    torch.tensor(
                        concentrator_translation_e[()], dtype=torch.float, device=device
                    )
                    if concentrator_translation_e
                    else torch.tensor(0.0, dtype=torch.float, device=device)
                ),
                concentrator_translation_n=(
                    torch.tensor(
                        concentrator_translation_n[()], dtype=torch.float, device=device
                    )
                    if concentrator_translation_n
                    else torch.tensor(0.0, dtype=torch.float, device=device)
                ),
                concentrator_translation_u=(
                    torch.tensor(
                        concentrator_translation_u[()], dtype=torch.float, device=device
                    )
                    if concentrator_translation_u
                    else torch.tensor(0.0, dtype=torch.float, device=device)
                ),
                concentrator_tilt_e=(
                    torch.tensor(
                        concentrator_tilt_e[()], dtype=torch.float, device=device
                    )
                    if concentrator_tilt_e
                    else torch.tensor(0.0, dtype=torch.float, device=device)
                ),
                concentrator_tilt_n=(
                    torch.tensor(
                        concentrator_tilt_n[()], dtype=torch.float, device=device
                    )
                    if concentrator_tilt_n
                    else torch.tensor(0.0, dtype=torch.float, device=device)
                ),
                concentrator_tilt_u=(
                    torch.tensor(
                        concentrator_tilt_u[()], dtype=torch.float, device=device
                    )
                    if concentrator_tilt_u
                    else torch.tensor(0.0, dtype=torch.float, device=device)
                ),
            )
            kinematic_config = KinematicLoadConfig(
                type=str(
                    config_file[config_dictionary.heliostat_kinematic_key][
                        config_dictionary.kinematic_type
                    ][()].decode("utf-8")
                ),
                initial_orientation=initial_orientation,
                deviations=kinematic_deviations,
            )
        else:
            if prototype_kinematic is None:
                raise ValueError(
                    "If the heliostat does not have an individual kinematic, a kinematic prototype must be provided!"
                )
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
                pivot_radius = config_file.get(
                    f"{config_dictionary.heliostat_actuator_key}/{ac}/"
                    f"{config_dictionary.actuator_parameters_key}/"
                    f"{config_dictionary.actuator_pivot_radius}"
                )
                initial_angle = config_file.get(
                    f"{config_dictionary.heliostat_actuator_key}/{ac}/"
                    f"{config_dictionary.actuator_parameters_key}/"
                    f"{config_dictionary.actuator_initial_angle}"
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
                if pivot_radius is None:
                    log.warning(
                        f"No individual {config_dictionary.actuator_pivot_radius} set for {ac} on "
                        f"{heliostat_name}. Using default values!"
                    )
                if initial_angle is None:
                    log.warning(
                        f"No individual {config_dictionary.actuator_initial_angle} set for {ac} on "
                        f"{heliostat_name}. Using default values!"
                    )
                actuator_parameters = ActuatorParameters(
                    increment=(
                        torch.tensor(increment[()], dtype=torch.float, device=device)
                        if increment
                        else torch.tensor(0.0, dtype=torch.float, device=device)
                    ),
                    initial_stroke_length=(
                        torch.tensor(
                            initial_stroke_length[()], dtype=torch.float, device=device
                        )
                        if initial_stroke_length
                        else torch.tensor(0.0, dtype=torch.float, device=device)
                    ),
                    offset=(
                        torch.tensor(offset[()], dtype=torch.float, device=device)
                        if offset
                        else torch.tensor(0.0, dtype=torch.float, device=device)
                    ),
                    pivot_radius=(
                        torch.tensor(pivot_radius[()], dtype=torch.float, device=device)
                        if pivot_radius
                        else torch.tensor(0.0, dtype=torch.float, device=device)
                    ),
                    initial_angle=(
                        torch.tensor(
                            initial_angle[()], dtype=torch.float, device=device
                        )
                        if initial_angle
                        else torch.tensor(0.0, dtype=torch.float, device=device)
                    ),
                )
                actuator_list.append(
                    ActuatorConfig(
                        key="",
                        type=str(
                            config_file[config_dictionary.heliostat_actuator_key][ac][
                                config_dictionary.actuator_type_key
                            ][()].decode("utf-8")
                        ),
                        clockwise_axis_movement=bool(
                            config_file[config_dictionary.heliostat_actuator_key][ac][
                                config_dictionary.actuator_clockwise_axis_movement
                            ][()]
                        ),
                        parameters=actuator_parameters,
                    )
                )
            actuator_list_config = ActuatorListConfig(actuator_list=actuator_list)
        else:
            if prototype_actuator_list is None:
                raise ValueError(
                    "If the heliostat does not have individual actuators, an actuator prototype must be provided!"
                )
            log.info(
                "Individual actuator configurations not provided - loading a heliostat with the actuator prototype."
            )
            actuator_list_config = prototype_actuator_list

        # Adapt initial angle of actuator one according to kinematic initial orientation.
        # ARTIST always expects heliostats to be initially oriented to the south [0.0, -1.0, 0.0] (in ENU).
        # The first actuator always rotates along the east-axis.
        # Since the actuator coordinate system is relative to the heliostat orientation, the initial angle
        # of actuator one needs to be transformed accordingly.
        initial_angle = actuator_list_config.actuator_list[0].parameters.initial_angle

        transformed_initial_angle = utils.transform_initial_angle(
            initial_angle=initial_angle,
            initial_orientation=kinematic_config.initial_orientation,
            device=device,
        )

        actuator_list_config.actuator_list[
            0
        ].parameters.initial_angle = transformed_initial_angle

        return cls(
            heliostat_id=heliostat_id,
            position=position,
            aim_point=aim_point,
            surface_config=surface_config,
            kinematic_config=kinematic_config,
            actuator_config=actuator_list_config,
            device=device,
        )

    def set_aligned_surface_with_incident_ray_direction(
        self,
        incident_ray_direction: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """
        Compute the aligned surface points and aligned surface normals of the heliostat.

        This method uses the incident ray direction to align the heliostat.

        Parameters
        ----------
        incident_ray_direction : torch.Tensor
            The incident ray direction.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).
        """
        device = torch.device(device)
        (
            self.current_aligned_surface_points,
            self.current_aligned_surface_normals,
        ) = self.kinematic.align_surface_with_incident_ray_direction(
            incident_ray_direction, self.surface_points, self.surface_normals, device
        )
        self.is_aligned = True

    def set_aligned_surface_with_motor_positions(
        self,
        motor_positions: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """
        Compute the aligned surface points and aligned surface normals of the heliostat.

        This method uses the motor positions to align the heliostat.

        Parameters
        ----------
        motor_positions : torch.Tensor
            The motor positions.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).
        """
        device = torch.device(device)
        (
            self.current_aligned_surface_points,
            self.current_aligned_surface_normals,
        ) = self.kinematic.align_surface_with_motor_positions(
            motor_positions, self.surface_points, self.surface_normals, device
        )
        self.is_aligned = True

    def get_orientation_from_motor_positions(
        self,
        motor_positions: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> torch.Tensor:
        """
        Compute the orientation for a heliostat given the desired motor positions.

        Parameters
        ----------
        motor_positions : torch.Tensor
            The desired motor positions.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        torch.Tensor
            The orientation for the given motor position.
        """
        device = torch.device(device)
        return self.kinematic.motor_positions_to_orientation(motor_positions, device)

    def set_preferred_reflection_direction(self, rays: torch.Tensor) -> None:
        """
        Reflect incoming rays according to a normal vector.

        Parameters
        ----------
        rays : torch.Tensor
            The incoming rays (from the sun) to be reflected.

        Raises
        ------
        ValueError
            If the heliostat has not yet been aligned.
        """
        if not self.is_aligned:
            raise ValueError("Heliostat has not yet been aligned.")
        self.preferred_reflection_direction = reflect(
            incoming_ray_direction=rays,
            reflection_surface_normals=self.current_aligned_surface_normals,
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
