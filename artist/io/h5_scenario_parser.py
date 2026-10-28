import functools
import logging
from collections.abc import Callable, Iterator
from typing import NamedTuple

import h5py
import torch

from artist.geometry import rotations
from artist.util import constants, indices
from artist.util.config import FacetConfig, SurfaceConfig
from artist.util.env import get_device

rigid_body_translation_deviations = (
    (indices.first_joint_translation_e, constants.first_joint_translation_e),
    (indices.first_joint_translation_n, constants.first_joint_translation_n),
    (indices.first_joint_translation_u, constants.first_joint_translation_u),
    (indices.second_joint_translation_e, constants.second_joint_translation_e),
    (indices.second_joint_translation_n, constants.second_joint_translation_n),
    (indices.second_joint_translation_u, constants.second_joint_translation_u),
    (indices.concentrator_translation_e, constants.concentrator_translation_e),
    (indices.concentrator_translation_n, constants.concentrator_translation_n),
    (indices.concentrator_translation_u, constants.concentrator_translation_u),
)
"""Index and HDF5 key of each rigid body translation deviation parameter."""

rigid_body_rotation_deviations = (
    (indices.first_joint_tilt_n, constants.first_joint_tilt_n),
    (indices.first_joint_tilt_u, constants.first_joint_tilt_u),
    (indices.second_joint_tilt_e, constants.second_joint_tilt_e),
    (indices.second_joint_tilt_n, constants.second_joint_tilt_n),
)
"""Index and HDF5 key of each rigid body rotation deviation parameter."""

linear_actuator_non_optimizable_parameters = (
    (indices.actuator_increment, constants.actuator_increment),
    (indices.actuator_offset, constants.actuator_offset),
    (indices.actuator_pivot_radius, constants.actuator_pivot_radius),
)
"""Index and HDF5 key of each non-optimizable linear actuator parameter."""

linear_actuator_optimizable_parameters = (
    (indices.actuator_initial_angle, constants.actuator_initial_angle),
    (indices.actuator_initial_stroke_length, constants.actuator_initial_stroke_length),
)
"""Index and HDF5 key of each optimizable linear actuator parameter."""


def get_rank() -> int:
    """
    Get the rank of this process, or 0 if not running distributed.

    Returns
    -------
    int
        The rank of this process.
    """
    return (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )


def read_facets(
    facet_config: h5py.Group,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Read the raw facet data of one surface from an HDF5 group onto the CPU.

    Parameters
    ----------
    facet_config : h5py.Group
        The HDF5 group containing one subgroup per facet.

    Returns
    -------
    torch.Tensor
        The control points of all facets.
        Shape is ``[number_of_facets, number_of_control_points_u_direction, number_of_control_points_v_direction, 3]``.
    torch.Tensor
        The NURBS degrees of all facets.
        Shape is ``[number_of_facets, 2]``.
    torch.Tensor
        The translation vectors of all facets.
        Shape is ``[number_of_facets, 4]``.
    torch.Tensor
        The canting vectors of all facets.
        Shape is ``[number_of_facets, 2, 4]``.
    """
    control_points, degrees, translation_vectors, canting = [], [], [], []
    for facet in facet_config.keys():
        single_facet_config = facet_config[facet]
        control_points.append(
            torch.as_tensor(
                single_facet_config[constants.facet_control_points][()],
                dtype=torch.float,
            )
        )
        facet_degrees = single_facet_config[constants.facet_degrees][()]
        degrees.append(
            torch.tensor(
                [facet_degrees[indices.nurbs_u], facet_degrees[indices.nurbs_v]],
                dtype=torch.int32,
            )
        )
        translation_vectors.append(
            torch.as_tensor(
                single_facet_config[constants.facets_translation_vector][()],
                dtype=torch.float,
            )
        )
        canting.append(
            torch.as_tensor(
                single_facet_config[constants.facets_canting][()], dtype=torch.float
            )
        )
    return (
        torch.stack(control_points),
        torch.stack(degrees),
        torch.stack(translation_vectors),
        torch.stack(canting),
    )


def surface_config(
    prototype: bool,
    scenario_file: h5py.File,
    device: torch.device | None = None,
) -> SurfaceConfig:
    """
    Load a surface configuration from an HDF5 scenario file.

    Parameters
    ----------
    prototype : bool
        Whether a prototype or an individual surface configuration should be loaded.
    scenario_file : h5py.File
        The opened scenario HDF5 file containing the information.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    SurfaceConfig
        The surface configuration.
    """
    device = get_device(device=device)

    if prototype:
        facet_config = scenario_file[constants.prototype_key][
            constants.surface_prototype_key
        ][constants.facets_key]
    else:
        facet_config = scenario_file[constants.heliostat_surface_key][
            constants.facets_key
        ]

    control_points, degrees, translation_vectors, canting = (
        tensor.to(device) for tensor in read_facets(facet_config=facet_config)
    )

    facet_list = [
        FacetConfig(
            facet_key="",
            control_points=control_points[facet_index],
            degrees=degrees[facet_index],
            translation_vector=translation_vectors[facet_index],
            canting=canting[facet_index],
        )
        for facet_index in range(control_points.shape[0])
    ]
    return SurfaceConfig(facet_list=facet_list)


def read_kinematics_deviations(
    kinematics_type: str,
    kinematics_config: h5py.Group,
) -> tuple[torch.Tensor, torch.Tensor, int, list[str]]:
    """
    Read kinematics deviations from an HDF5 group onto the CPU.

    Parameters
    ----------
    kinematics_type : str
        The kinematics type.
    kinematics_config : h5py.Group
        The HDF5 group containing the kinematics configuration.

    Raises
    ------
    ValueError
        If the kinematics type is unknown.

    Returns
    -------
    torch.Tensor
        Translation deviation parameters for the kinematics.
    torch.Tensor
        Rotation deviation parameters for the kinematics.
    int
        The number of actuators needed for this kinematics type.
    list[str]
        The names of all deviation parameters missing in the file, which default to zero.
    """
    if kinematics_type == constants.rigid_body_key:
        translation_deviations, rotation_deviations, missing_parameters = (
            read_rigid_body_deviations(kinematics_config=kinematics_config)
        )
        number_of_actuators = constants.rigid_body_number_of_actuators
    else:
        raise ValueError(
            f"The kinematics type: {kinematics_type} is not yet implemented!"
        )

    return (
        translation_deviations,
        rotation_deviations,
        number_of_actuators,
        missing_parameters,
    )


def kinematics_deviations(
    prototype: bool,
    kinematics_type: str,
    scenario_file: h5py.File,
    log: logging.Logger,
    heliostat_name: str | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Load kinematics deviations from an HDF5 scenario file.

    Parameters
    ----------
    prototype : bool
        Whether a prototype or individual kinematics deviations should be loaded.
    kinematics_type : str
        The kinematics type.
    scenario_file : h5py.File
        The opened scenario HDF5 file containing the information.
    log : logging.Logger
        The logger for the scenario loader.
    heliostat_name : str | None
        The heliostat name, only needed for individual heliostats, not prototypes (default is None).
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Raises
    ------
    ValueError
        If the kinematics type in the scenario file is unknown.

    Returns
    -------
    torch.Tensor
        Translation deviation parameters for the kinematics.
    torch.Tensor
        Rotation deviation parameters for the kinematics.
    int
        The number of actuators needed for this kinematics type.
    """
    device = get_device(device=device)

    if prototype:
        kinematics_config = scenario_file[constants.prototype_key][
            constants.kinematics_prototype_key
        ]
    else:
        kinematics_config = scenario_file[constants.heliostat_kinematics_key]

    translation_deviations, rotation_deviations, number_of_actuators, missing = (
        read_kinematics_deviations(
            kinematics_type=kinematics_type, kinematics_config=kinematics_config
        )
    )
    log_missing_kinematics_deviations(
        log=log, missing_parameters=missing, heliostat_name=heliostat_name
    )

    return (
        translation_deviations.to(device),
        rotation_deviations.to(device),
        number_of_actuators,
    )


def read_rigid_body_deviations(
    kinematics_config: h5py.Group,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Read kinematics deviations for a rigid body kinematics from an HDF5 group onto the CPU.

    Missing parameters default to zero.

    Parameters
    ----------
    kinematics_config : h5py.Group
        The HDF5 group containing the kinematics configuration.

    Returns
    -------
    torch.Tensor
        Translation deviation parameters for the kinematics.
        Shape is ``[9]``.
    torch.Tensor
        Rotation deviation parameters for the kinematics.
        Shape is ``[4]``.
    list[str]
        The names of all deviation parameters missing in the file.
    """
    translation_deviations = torch.zeros(
        constants.rigid_body_number_of_translation_deviation_parameters,
        dtype=torch.float,
    )
    rotation_deviations = torch.zeros(
        constants.rigid_body_number_of_rotation_deviation_parameters,
        dtype=torch.float,
    )
    missing_parameters = []

    for deviations, parameters in (
        (translation_deviations, rigid_body_translation_deviations),
        (rotation_deviations, rigid_body_rotation_deviations),
    ):
        for index, key in parameters:
            parameter = kinematics_config.get(
                f"{constants.kinematics_deviations}/{key}"
            )
            if parameter is None:
                missing_parameters.append(key)
            else:
                deviations[index] = torch.as_tensor(parameter[()], dtype=torch.float)

    return translation_deviations, rotation_deviations, missing_parameters


def log_missing_kinematics_deviations(
    log: logging.Logger,
    missing_parameters: list[str],
    heliostat_name: str | None = None,
) -> None:
    """
    Warn about kinematics deviation parameters that are missing and default to zero.

    Parameters
    ----------
    log : logging.Logger
        The logger for the scenario loader.
    missing_parameters : list[str]
        The names of all missing deviation parameters.
    heliostat_name : str | None
        The heliostat name, only needed for individual heliostats, not prototypes (default is None).
    """
    if get_rank() != 0:
        return
    for parameter in missing_parameters:
        log.warning(
            f"No individual kinematics {parameter} for {heliostat_name} set. "
            f"Using default values!"
        )


def rigid_body_deviations(
    kinematics_config: h5py.File,
    log: logging.Logger,
    heliostat_name: str | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load kinematics deviations for a rigid body kinematics from an HDF5 scenario file.

    Parameters
    ----------
    kinematics_config : h5py.File
        The opened scenario HDF5 file containing the information.
    log : logging.Logger
        The logger for the scenario loader.
    heliostat_name : str | None
        The heliostat name, only needed for individual heliostats, not prototypes (default is None).
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Returns
    -------
    torch.Tensor
        Translation deviation parameters for the kinematics.
        Shape is ``[9]``.
    torch.Tensor
        Rotation deviation parameters for the kinematics.
        Shape is ``[4]``.
    """
    device = get_device(device=device)

    translation_deviations, rotation_deviations, missing_parameters = (
        read_rigid_body_deviations(kinematics_config=kinematics_config)
    )
    log_missing_kinematics_deviations(
        log=log, missing_parameters=missing_parameters, heliostat_name=heliostat_name
    )

    return translation_deviations.to(device), rotation_deviations.to(device)


def read_actuator_parameters(
    actuator_type: str,
    actuator_config: h5py.Group,
    number_of_actuators: int,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[str, str]]]:
    """
    Read actuator parameters from an HDF5 group onto the CPU.

    Parameters
    ----------
    actuator_type : str
        The actuator type.
    actuator_config : h5py.Group
        The HDF5 group containing one subgroup per actuator.
    number_of_actuators : int
        The number of actuators.

    Raises
    ------
    ValueError
        If the actuator type is unknown.

    Returns
    -------
    torch.Tensor
        Non-optimizable actuator parameters for each actuator in the file.
    torch.Tensor
        Optimizable actuator parameters for each actuator in the file.
    list[tuple[str, str]]
        The actuator and parameter names of all actuator parameters missing in the file, which default to zero.
    """
    if actuator_type == constants.linear_actuator_key:
        return read_linear_actuators(
            actuator_config=actuator_config, number_of_actuators=number_of_actuators
        )
    if actuator_type == constants.ideal_actuator_key:
        non_optimizable, optimizable = read_ideal_actuators(
            actuator_config=actuator_config, number_of_actuators=number_of_actuators
        )
        return non_optimizable, optimizable, []
    raise ValueError(f"The actuator type: {actuator_type} is not yet implemented!")


def actuator_parameters(
    prototype: bool,
    scenario_file: h5py.File,
    actuator_type: str,
    number_of_actuators: int,
    log: logging.Logger,
    heliostat_name: str | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load actuator parameters from an HDF5 scenario file.

    Parameters
    ----------
    prototype : bool
        Whether prototype or individual actuator parameters should be loaded.
    scenario_file : h5py.File
        The opened scenario HDF5 file containing the information.
    actuator_type : str
        The actuator type.
    number_of_actuators : int
        The number of actuators.
    log : logging.Logger
        The logger for the scenario loader.
    heliostat_name : str | None
        The heliostat name, only needed for individual heliostats, not prototypes (default is None).
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Raises
    ------
    ValueError
        If the actuator type in the scenario file is unknown.

    Returns
    -------
    torch.Tensor
        Non-optimizable actuator parameters for each actuator in the file.
    torch.Tensor
        Optimizable actuator parameters for each actuator in the file.
    """
    device = get_device(device=device)

    if prototype:
        actuator_config = scenario_file[constants.prototype_key][
            constants.actuators_prototype_key
        ]
    else:
        actuator_config = scenario_file[constants.heliostat_actuator_key]

    non_optimizable, optimizable, missing_parameters = read_actuator_parameters(
        actuator_type=actuator_type,
        actuator_config=actuator_config,
        number_of_actuators=number_of_actuators,
    )
    log_missing_actuator_parameters(
        log=log, missing_parameters=missing_parameters, heliostat_name=heliostat_name
    )

    return non_optimizable.to(device), optimizable.to(device)


@functools.cache
def linear_actuator_initial_angle_offset() -> torch.Tensor:
    """
    Compute the offset added to the initial angle of the first linear actuator.

    For all linear actuators in the rigid body kinematics, the initial angle of actuator number one is adapted
    according to the initial surface orientation:

    - ARTIST always expects heliostats to be initially oriented to the south -> artist_standard_orientation = [0.0, -1.0, 0.0] (in ENU).
    - The first actuator rotates along the east-axis.
    - The "initial angle" of the actuator is a relative angle: it defines the actuator's initial angle relative to the physical geometry of the surface.
    - Surfaces in ARTIST are always provided oriented upwards ([0, 0, 1]), even if the initial orientation in a database deviates.
      -> The surface points and normals are always sampled from a model (converted NURBS from deflectometry or ideal NURBS) that lays
      flat on the ground, i.e., the surface normals are always pointing upwards [0.0, 0.0, 1.0].
    - The final orientation needs to be computed correctly from the surface orientation and the relative actuator initial angle.
    - To ensure final orientations remain consistent, first the rotation from the standard orientation (ARTIST: south) to the actual surface
      orientation is computed (rotation from south (ARTIST) to up (surfaces)). This rotation is projected along the
      rotation axis of the first actuator (east-axis) and added to the initial angle. This compensates for the different orientations of the sampled
      surfaces and the relative turning axis of the first actuator.

    Returns
    -------
    torch.Tensor
        The initial angle offset as a scalar tensor on the CPU.
    """
    cpu = torch.device("cpu")
    surface_orientation = torch.tensor([0.0, 0.0, 1.0, 0.0], device=cpu)
    artist_standard_orientation = torch.tensor([0.0, -1.0, 0.0, 0.0], device=cpu)
    axis, angle = rotations.rotation_angle_and_axis(
        from_orientation=artist_standard_orientation,
        to_orientation=surface_orientation,
        device=cpu,
    )
    return axis[indices.e] * angle


def read_linear_actuators(
    actuator_config: h5py.Group,
    number_of_actuators: int,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[str, str]]]:
    """
    Read actuator parameters for linear actuators from an HDF5 group onto the CPU.

    Missing parameters default to zero.

    Parameters
    ----------
    actuator_config : h5py.Group
        The HDF5 group containing one subgroup per actuator.
    number_of_actuators : int
        The number of actuators used for a specific kinematics.

    Raises
    ------
    ValueError
        If the file contains the wrong number of actuators for a heliostat with a specific kinematics type.

    Returns
    -------
    torch.Tensor
        Non-optimizable actuator parameters for each actuator in the file.
    torch.Tensor
        Optimizable actuator parameters for each actuator in the file.
    list[tuple[str, str]]
        The actuator and parameter names of all actuator parameters missing in the file.
    """
    if len(actuator_config.keys()) != number_of_actuators:
        raise ValueError(
            f"This scenario file contains the wrong amount of actuators for this heliostat and its kinematics type."
            f" Expected {number_of_actuators} actuators, found {len(actuator_config.keys())} actuator(s)."
        )

    actuator_parameters_non_optimizable = torch.zeros((7, number_of_actuators))
    actuator_parameters_optimizable = torch.zeros((2, number_of_actuators))
    missing_parameters = []

    for index, actuator in enumerate(actuator_config.keys()):
        single_actuator_config = actuator_config[actuator]
        read_actuator_motor_settings(
            actuator_parameters_non_optimizable=actuator_parameters_non_optimizable,
            single_actuator_config=single_actuator_config,
            actuator_type_int=constants.linear_actuator_int,
            index=index,
        )
        parameters_config = single_actuator_config.get(
            constants.actuator_parameters_key
        )
        for actuator_parameters, parameters in (
            (
                actuator_parameters_non_optimizable,
                linear_actuator_non_optimizable_parameters,
            ),
            (actuator_parameters_optimizable, linear_actuator_optimizable_parameters),
        ):
            for parameter_index, key in parameters:
                parameter = (
                    parameters_config.get(key)
                    if parameters_config is not None
                    else None
                )
                if parameter is None:
                    missing_parameters.append((actuator, key))
                else:
                    actuator_parameters[parameter_index, index] = torch.as_tensor(
                        parameter[()], dtype=torch.float
                    )

    actuator_parameters_optimizable[
        indices.actuator_initial_angle, indices.actuator_one_index
    ] += linear_actuator_initial_angle_offset()

    return (
        actuator_parameters_non_optimizable,
        actuator_parameters_optimizable,
        missing_parameters,
    )


def log_missing_actuator_parameters(
    log: logging.Logger,
    missing_parameters: list[tuple[str, str]],
    heliostat_name: str | None = None,
) -> None:
    """
    Warn about actuator parameters that are missing and default to zero.

    Parameters
    ----------
    log : logging.Logger
        The logger for the scenario loader.
    missing_parameters : list[tuple[str, str]]
        The actuator and parameter names of all missing actuator parameters.
    heliostat_name : str | None
        The heliostat name, only needed for individual heliostats, not prototypes (default is None).
    """
    if get_rank() != 0:
        return
    for actuator, parameter in missing_parameters:
        log.warning(
            f"No individual {parameter} set for {actuator} on {heliostat_name}. "
            f"Using default values!"
        )


def linear_actuators(
    actuator_config: h5py.File,
    number_of_actuators: int,
    log: logging.Logger,
    heliostat_name: str | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load actuator parameters for linear actuators from an HDF5 scenario file.

    Parameters
    ----------
    actuator_config : h5py.File
        The opened scenario HDF5 file containing the information.
    number_of_actuators : int
        The number of actuators used for a specific kinematics.
    log : logging.Logger
        The logger for the scenario loader.
    heliostat_name : str | None
        The heliostat name, only needed for individual heliostats, not prototypes (default is None).
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Raises
    ------
    ValueError
        If the file contains the wrong number of actuators for a heliostat with a specific kinematics type.

    Returns
    -------
    torch.Tensor
        Non-optimizable actuator parameters for each actuator in the file.
    torch.Tensor
        Optimizable actuator parameters for each actuator in the file.
    """
    device = get_device(device=device)

    non_optimizable, optimizable, missing_parameters = read_linear_actuators(
        actuator_config=actuator_config, number_of_actuators=number_of_actuators
    )
    log_missing_actuator_parameters(
        log=log, missing_parameters=missing_parameters, heliostat_name=heliostat_name
    )

    return non_optimizable.to(device), optimizable.to(device)


def read_actuator_motor_settings(
    actuator_parameters_non_optimizable: torch.Tensor,
    single_actuator_config: h5py.Group,
    actuator_type_int: int,
    index: int,
) -> None:
    """
    Read the type, turning direction and motor limits of one actuator into its parameter column.

    Parameters
    ----------
    actuator_parameters_non_optimizable : torch.Tensor
        The non-optimizable actuator parameters, updated in place.
    single_actuator_config : h5py.Group
        The HDF5 group of one actuator.
    actuator_type_int : int
        The integer encoding of the actuator type.
    index : int
        The index of the actuator.
    """
    clockwise_axis_movement = bool(
        single_actuator_config[constants.actuator_clockwise_axis_movement][()]
    )
    min_max_motor_positions = single_actuator_config[
        constants.actuator_min_max_motor_positions
    ][()]

    actuator_parameters_non_optimizable[indices.actuator_type, index] = (
        actuator_type_int
    )
    actuator_parameters_non_optimizable[indices.actuator_clockwise_movement, index] = (
        0 if not clockwise_axis_movement else 1
    )
    actuator_parameters_non_optimizable[indices.actuator_min_motor_position, index] = (
        float(min_max_motor_positions[indices.data_actuator_min_motor_position])
    )
    actuator_parameters_non_optimizable[indices.actuator_max_motor_position, index] = (
        float(min_max_motor_positions[indices.data_actuator_max_motor_position])
    )


def read_ideal_actuators(
    actuator_config: h5py.Group,
    number_of_actuators: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Read actuator parameters for ideal actuators from an HDF5 group onto the CPU.

    Parameters
    ----------
    actuator_config : h5py.Group
        The HDF5 group containing one subgroup per actuator.
    number_of_actuators : int
        The number of actuators used for a specific kinematics.

    Raises
    ------
    ValueError
        If the file contains the wrong number of actuators for a heliostat with a specific kinematics type.

    Returns
    -------
    torch.Tensor
        Non-optimizable actuator parameters for each actuator in the file.
    torch.Tensor
        Optimizable actuator parameters for each actuator in the file.
    """
    if len(actuator_config.keys()) != number_of_actuators:
        raise ValueError(
            f"This scenario file contains the wrong amount of actuators for this heliostat and its kinematics type."
            f" Expected {number_of_actuators} actuators, found {len(actuator_config.keys())} actuator(s)."
        )

    actuator_parameters_non_optimizable = torch.zeros((4, number_of_actuators))
    actuator_parameters_optimizable = torch.tensor([])

    for index, actuator in enumerate(actuator_config.keys()):
        read_actuator_motor_settings(
            actuator_parameters_non_optimizable=actuator_parameters_non_optimizable,
            single_actuator_config=actuator_config[actuator],
            actuator_type_int=constants.ideal_actuator_int,
            index=index,
        )

    return actuator_parameters_non_optimizable, actuator_parameters_optimizable


def ideal_actuators(
    actuator_config: h5py.File,
    number_of_actuators: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load actuator parameters for ideal actuators from an HDF5 scenario file.

    Parameters
    ----------
    actuator_config : h5py.File
        The opened scenario HDF5 file containing the information.
    number_of_actuators : int
        The number of actuators used for a specific kinematics.
    device : torch.device | None
        The device on which to perform computations or load tensors and models (default is None).
        If None, ``ARTIST`` will automatically select the most appropriate
        device (CUDA or CPU) based on availability and OS.

    Raises
    ------
    ValueError
        If the file contains the wrong number of actuators for a heliostat with a specific kinematics type.

    Returns
    -------
    torch.Tensor
        Non-optimizable actuator parameters for each actuator in the file.
    torch.Tensor
        Optimizable actuator parameters for each actuator in the file.
    """
    device = get_device(device=device)

    non_optimizable, optimizable = read_ideal_actuators(
        actuator_config=actuator_config, number_of_actuators=number_of_actuators
    )
    return non_optimizable.to(device), optimizable.to(device)


class HeliostatRecord(NamedTuple):
    """
    Lazy access to the data of one heliostat in a scenario file, independent of the file layout.

    Components the heliostat has no individual parameters for are None, so that the prototype is used instead.
    All data is read onto the CPU.

    Attributes
    ----------
    name : str
        The heliostat name.
    position : Callable[[], torch.Tensor]
        Read the heliostat position.
    surface : Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None
        Read the control points, degrees, translation vectors, and canting vectors of all facets, see ``read_facets()``.
    kinematics : Callable[[], tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, int, list[str]]] | None
        Read the kinematics type, initial orientation, translation and rotation deviations, number of actuators,
        and the names of missing deviation parameters.
    actuator_types : Callable[[], list[str]] | None
        Read the type of each actuator.
    actuator_parameters : Callable[..., tuple[torch.Tensor, torch.Tensor, list[tuple[str, str]]]] | None
        Read the actuator parameters for the keyword arguments ``actuator_type`` and ``number_of_actuators``,
        see ``read_actuator_parameters()``.
    """

    name: str
    position: Callable[[], torch.Tensor]
    surface: (
        Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
        | None
    )
    kinematics: (
        Callable[
            [], tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, int, list[str]]
        ]
        | None
    )
    actuator_types: Callable[[], list[str]] | None
    actuator_parameters: (
        Callable[..., tuple[torch.Tensor, torch.Tensor, list[tuple[str, str]]]] | None
    )


def read_position(single_heliostat_config: h5py.Group) -> torch.Tensor:
    """
    Read the position of one heliostat from its HDF5 group onto the CPU.

    Parameters
    ----------
    single_heliostat_config : h5py.Group
        The HDF5 group of one heliostat.

    Returns
    -------
    torch.Tensor
        The heliostat position.
        Shape is ``[4]``.
    """
    return torch.as_tensor(
        single_heliostat_config[constants.heliostat_position][()], dtype=torch.float
    )


def read_kinematics(
    kinematics_config: h5py.Group,
) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, int, list[str]]:
    """
    Read the kinematics of one heliostat from its HDF5 group onto the CPU.

    Parameters
    ----------
    kinematics_config : h5py.Group
        The HDF5 group containing the kinematics configuration.

    Returns
    -------
    str
        The kinematics type.
    torch.Tensor
        The initial orientation.
        Shape is ``[4]``.
    torch.Tensor
        Translation deviation parameters for the kinematics.
    torch.Tensor
        Rotation deviation parameters for the kinematics.
    int
        The number of actuators needed for this kinematics type.
    list[str]
        The names of all deviation parameters missing in the file, which default to zero.
    """
    initial_orientation = torch.as_tensor(
        kinematics_config[constants.kinematics_initial_orientation][()],
        dtype=torch.float,
    )
    kinematics_type = kinematics_config[constants.kinematics_type][()].decode("utf-8")
    translation_deviations, rotation_deviations, number_of_actuators, missing = (
        read_kinematics_deviations(
            kinematics_type=kinematics_type, kinematics_config=kinematics_config
        )
    )
    return (
        kinematics_type,
        initial_orientation,
        translation_deviations,
        rotation_deviations,
        number_of_actuators,
        missing,
    )


def read_actuator_types(actuators_config: h5py.Group) -> list[str]:
    """
    Read the type of each actuator of one heliostat from its HDF5 group.

    Parameters
    ----------
    actuators_config : h5py.Group
        The HDF5 group containing one subgroup per actuator.

    Returns
    -------
    list[str]
        The type of each actuator.
    """
    return [
        actuators_config[key][constants.actuator_type_key][()].decode("utf-8")
        for key in actuators_config.keys()
    ]


def legacy_heliostat_records(
    heliostats_config: h5py.Group,
) -> Iterator[HeliostatRecord]:
    """
    Create lazy records for heliostats stored in the legacy layout, with one HDF5 group per heliostat.

    Parameters
    ----------
    heliostats_config : h5py.Group
        The HDF5 group containing one subgroup per heliostat.

    Yields
    ------
    HeliostatRecord
        The record of each heliostat.
    """
    for heliostat_name in heliostats_config.keys():
        single_heliostat_config = heliostats_config[heliostat_name]
        has_surface = constants.heliostat_surface_key in single_heliostat_config.keys()
        has_kinematics = (
            constants.heliostat_kinematics_key in single_heliostat_config.keys()
        )
        has_actuators = (
            constants.heliostat_actuator_key in single_heliostat_config.keys()
        )
        actuators_config = (
            single_heliostat_config[constants.heliostat_actuator_key]
            if has_actuators
            else None
        )
        yield HeliostatRecord(
            name=heliostat_name,
            position=functools.partial(read_position, single_heliostat_config),
            surface=functools.partial(
                read_facets,
                single_heliostat_config[constants.heliostat_surface_key][
                    constants.facets_key
                ],
            )
            if has_surface
            else None,
            kinematics=functools.partial(
                read_kinematics,
                single_heliostat_config[constants.heliostat_kinematics_key],
            )
            if has_kinematics
            else None,
            actuator_types=functools.partial(read_actuator_types, actuators_config)
            if has_actuators
            else None,
            actuator_parameters=functools.partial(
                read_actuator_parameters, actuator_config=actuators_config
            )
            if has_actuators
            else None,
        )
