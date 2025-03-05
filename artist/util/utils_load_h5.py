from ast import Tuple
import logging
from typing import Optional, Union
from artist.field import surface
from artist.util import config_dictionary, utils
from artist.util.configuration_classes import FacetConfig, SurfaceConfig
import h5py
import torch


def surface_config(prototype: bool,
                   scenario_file: h5py.File,
                   device: Union[torch.device, str] = "cuda"
) -> SurfaceConfig:
    """
    Load a surface configuration from an HDF5 scenario file.

    Parameters
    ----------
    prototype : bool
        Loading a prototype or an individual surface configuration.
    scenario_file : h5py.File
        The opened scenario HDF5 file containing the information.
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Returns
    -------
    SurfaceConfig
        The surface configuration.
    """
    device = torch.device(device)
    if prototype:
        facet_config = scenario_file[config_dictionary.prototype_key][
            config_dictionary.surface_prototype_key
        ][config_dictionary.facets_key]
    else:
        facet_config = scenario_file[config_dictionary.heliostat_surface_key][config_dictionary.facets_key]

    facet_list = [
        FacetConfig(
            facet_key="",
            control_points=torch.tensor(
                facet_config[facet][config_dictionary.facet_control_points][()],
                dtype=torch.float,
                device=device,
            ),
            degree_e=int(
                facet_config[facet][config_dictionary.facet_degree_e][()]
            ),
            degree_n=int(
                facet_config[facet][config_dictionary.facet_degree_n][()]
            ),
            number_eval_points_e=int(
                facet_config[facet][config_dictionary.facet_number_eval_e][()]
            ),
            number_eval_points_n=int(
                facet_config[facet][config_dictionary.facet_number_eval_n][()]
            ),
            translation_vector=torch.tensor(
                facet_config[facet][config_dictionary.facets_translation_vector][()],
                dtype=torch.float,
                device=device,
            ),
            canting_e=torch.tensor(
                facet_config[facet][config_dictionary.facets_canting_e][()],
                dtype=torch.float,
                device=device,
            ),
            canting_n=torch.tensor(
                facet_config[facet][config_dictionary.facets_canting_n][()],
                dtype=torch.float,
                device=device,
            ),
        )
        for facet in facet_config.keys()
    ]
    surface_config = SurfaceConfig(facet_list=facet_list)
    return surface_config


def kinematic_deviations(prototype: bool,
                         kinematic_type: str,
                         scenario_file: h5py.File,
                         log: logging.Logger,
                         heliostat_name: Optional[str]=None, 
                         device: Union[torch.device, str] = "cuda"
) -> tuple[torch.Tensor, int]:
    """
    Load kinematic deviations from an HDF5 scenario file.

    Parameters
    ----------
    prototype : bool
        Loading prototype or individual kinematic deviations.
    kinematic_type : str
        The kinematic type.
    scenario_file : h5py.File
        The opened scenario HDF5 file containing the information.
    log : logging.Logger
        The logger for the scenario loader.
    heliostat_name : Optional[str]
        The heliostat name, only needed for individual heliostats, not prototypes (default is None).
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Raises
    ------
    ValueError
        If the kinmatic type in the scenario file is unknown.

    Returns
    -------
    torch.Tensor
        The kinematic deviation parameters.
    int
        The number of actuators needed for this kinematic type.
    """
    device = torch.device(device)

    if prototype:
        kinematic_deviations_config = scenario_file[config_dictionary.prototype_key][
            config_dictionary.kinematic_prototype_key
        ][config_dictionary.kinematic_deviations_key]
    else:
        kinematic_deviations_config = scenario_file[config_dictionary.heliostat_kinematic_key][config_dictionary.kinematic_deviations_key]

    if kinematic_type == config_dictionary.rigid_body_key:
        kinematic_deviations = rigid_body_deviations(
            kinematic_deviations_config=kinematic_deviations_config,
            log=log,
            heliostat_name=heliostat_name,
            device=device
        )
        number_of_actuators = config_dictionary.rigid_body_number_of_actuators
    elif kinematic_type == "new_kinematic":
        kinematic_deviations = new_kinematic_deviations()
        number_of_actuators = 3
    else:
        raise ValueError(
            f"The kinematic type: {kinematic_type} is not yet implemented!"
        )
    
    return kinematic_deviations, number_of_actuators

def rigid_body_deviations(kinematic_deviations_config: h5py.File,
                          log: logging.Logger,
                          heliostat_name: Optional[str]=None, 
                          device: Union[torch.device, str] = "cuda"
) -> torch.Tensor:
    """
    Load kinematic deviations for a rigid body kinematic from an HDF5 scenario file.

    Parameters
    ----------
    kinematic_deviations_config : h5py.File
        The opened scenario HDF5 file containing the information.
    log : logging.Logger
        The logger for the scenario loader.
    heliostat_name : Optional[str]
        The heliostat name, only needed for individual heliostats, not prototypes (default is None).
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Returns
    -------
    torch.Tensor
        18 deviation parameters for the rigid body kinematic.
    """
    device = torch.device(device)

    kinematic_deviations = torch.zeros(config_dictionary.rigid_body_number_of_deviation_parameters, dtype=torch.float, device=device)
    
    first_joint_translation_e = kinematic_deviations_config.get(
        f"{config_dictionary.first_joint_translation_e}"
    )
    first_joint_translation_n = kinematic_deviations_config.get(
        f"{config_dictionary.first_joint_translation_n}"
    )
    first_joint_translation_u = kinematic_deviations_config.get(
        f"{config_dictionary.first_joint_translation_u}"
    )
    first_joint_tilt_e = kinematic_deviations_config.get(
        f"{config_dictionary.first_joint_tilt_e}"
    )
    first_joint_tilt_n = kinematic_deviations_config.get(
        f"{config_dictionary.first_joint_tilt_n}"
    )
    first_joint_tilt_u = kinematic_deviations_config.get(
        f"{config_dictionary.first_joint_tilt_u}"
    )
    second_joint_translation_e = kinematic_deviations_config.get(
        f"{config_dictionary.second_joint_translation_e}"
    )
    second_joint_translation_n = kinematic_deviations_config.get(
        f"{config_dictionary.second_joint_translation_n}"
    )
    second_joint_translation_u = kinematic_deviations_config.get(
        f"{config_dictionary.second_joint_translation_u}"
    )
    second_joint_tilt_e = kinematic_deviations_config.get(
        f"{config_dictionary.second_joint_tilt_e}"
    )
    second_joint_tilt_n = kinematic_deviations_config.get(
        f"{config_dictionary.second_joint_tilt_n}"
    )
    second_joint_tilt_u = kinematic_deviations_config.get(
        f"{config_dictionary.second_joint_tilt_u}"
    )
    concentrator_translation_e = kinematic_deviations_config.get(
        f"{config_dictionary.concentrator_translation_e}"
    )
    concentrator_translation_n = kinematic_deviations_config.get(
        f"{config_dictionary.concentrator_translation_n}"
    )
    concentrator_translation_u = kinematic_deviations_config.get(
        f"{config_dictionary.concentrator_translation_u}"
    )
    concentrator_tilt_e = kinematic_deviations_config.get(
        f"{config_dictionary.concentrator_tilt_e}"
    )
    concentrator_tilt_n = kinematic_deviations_config.get(
        f"{config_dictionary.concentrator_tilt_n}"
    )
    concentrator_tilt_u = kinematic_deviations_config.get(
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
    kinematic_deviations[0] = (
        torch.tensor(
            first_joint_translation_e[()], dtype=torch.float, device=device
        )
        if first_joint_translation_e
        else torch.tensor(0.0, dtype=torch.float, device=device)
    )
    kinematic_deviations[1] = (
        torch.tensor(
            first_joint_translation_n[()], dtype=torch.float, device=device
        )
        if first_joint_translation_n
        else torch.tensor(0.0, dtype=torch.float, device=device)
    )
    kinematic_deviations[2] = (
        torch.tensor(
            first_joint_translation_u[()], dtype=torch.float, device=device
        )
        if first_joint_translation_u
        else torch.tensor(0.0, dtype=torch.float, device=device)
    )
    kinematic_deviations[3] = (
        torch.tensor(
            first_joint_tilt_e[()], dtype=torch.float, device=device
        )
        if first_joint_tilt_e
        else torch.tensor(0.0, dtype=torch.float, device=device)
    )
    kinematic_deviations[4] = (
        torch.tensor(
            first_joint_tilt_n[()], dtype=torch.float, device=device
        )
        if first_joint_tilt_n
        else torch.tensor(0.0, dtype=torch.float, device=device)
    )
    kinematic_deviations[5] = (
        torch.tensor(
            first_joint_tilt_u[()], dtype=torch.float, device=device
        )
        if first_joint_tilt_u
        else torch.tensor(0.0, dtype=torch.float, device=device)
    )
    kinematic_deviations[6] = (
        torch.tensor(
            second_joint_translation_e[()], dtype=torch.float, device=device
        )
        if second_joint_translation_e
        else torch.tensor(0.0, dtype=torch.float, device=device)
    )
    kinematic_deviations[7] = (
        torch.tensor(
            second_joint_translation_n[()], dtype=torch.float, device=device
        )
        if second_joint_translation_n
        else torch.tensor(0.0, dtype=torch.float, device=device)
    )
    kinematic_deviations[8] = (
        torch.tensor(
            second_joint_translation_u[()], dtype=torch.float, device=device
        )
        if second_joint_translation_u
        else torch.tensor(0.0, dtype=torch.float, device=device)
    )
    kinematic_deviations[9] = (
        torch.tensor(
            second_joint_tilt_e[()], dtype=torch.float, device=device
        )
        if second_joint_tilt_e
        else torch.tensor(0.0, dtype=torch.float, device=device)
    )
    kinematic_deviations[10] = (
        torch.tensor(
            second_joint_tilt_n[()], dtype=torch.float, device=device
        )
        if second_joint_tilt_n
        else torch.tensor(0.0, dtype=torch.float, device=device)
    )
    kinematic_deviations[11] = (
        torch.tensor(
            second_joint_tilt_u[()], dtype=torch.float, device=device
        )
        if second_joint_tilt_u
        else torch.tensor(0.0, dtype=torch.float, device=device)
    )
    kinematic_deviations[12] = (
        torch.tensor(
            concentrator_translation_e[()], dtype=torch.float, device=device
        )
        if concentrator_translation_e
        else torch.tensor(0.0, dtype=torch.float, device=device)
    )
    kinematic_deviations[13] = (
        torch.tensor(
            concentrator_translation_n[()], dtype=torch.float, device=device
        )
        if concentrator_translation_n
        else torch.tensor(0.0, dtype=torch.float, device=device)
    )
    kinematic_deviations[14] = (
        torch.tensor(
            concentrator_translation_u[()], dtype=torch.float, device=device
        )
        if concentrator_translation_u
        else torch.tensor(0.0, dtype=torch.float, device=device)
    )
    kinematic_deviations[15] = (
        torch.tensor(
            concentrator_tilt_e[()], dtype=torch.float, device=device
        )
        if concentrator_tilt_e
        else torch.tensor(0.0, dtype=torch.float, device=device)
    )
    kinematic_deviations[16] = (
        torch.tensor(
            concentrator_tilt_n[()], dtype=torch.float, device=device
        )
        if concentrator_tilt_n
        else torch.tensor(0.0, dtype=torch.float, device=device)
    )
    kinematic_deviations[17] = (
        torch.tensor(
            concentrator_tilt_u[()], dtype=torch.float, device=device
        )
        if concentrator_tilt_u
        else torch.tensor(0.0, dtype=torch.float, device=device)
    )

    return kinematic_deviations


def new_kinematic_deviations():
    pass


def actuator_parameters(prototype: bool,
                   scenario_file: h5py.File,
                   actuator_type: str,
                   number_of_actuators: int,
                   initial_orientation: torch.Tensor,
                   log = logging.Logger,
                   heliostat_name: Optional[str]=None,
                   device: Union[torch.device, str] = "cuda"
) -> torch.Tensor:
    """
    Load actuator parameters from an HDF5 scenario file.

    Parameters
    ----------
    prototype : bool
        Loading prototype or individual actuator parameters.
    scenario_file : h5py.File
        The opened scenario HDF5 file containing the information.
    actuator_type : str
        The actuator type.
    number_of_actuators : int
        The number of actuators.
    initial_orientation : torch.Tensor
        The initial orientation of the heliostat.
    log : logging.Logger
        The logger for the scenario loader.
    heliostat_name : Optional[str]
        The heliostat name, only needed for individual heliostats, not prototypes (default is None).
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Raises
    ------
    ValueError
        If the actuator type in the scenario file is unknown.
        
    Returns
    -------
    torch.Tensor
        Actuator parameters for for each actuator in the file.
    """
    device = torch.device(device)

    if prototype:
        actuator_config = scenario_file[config_dictionary.prototype_key][
            config_dictionary.actuators_prototype_key
        ]
    else:
        actuator_config = scenario_file[config_dictionary.heliostat_actuator_key]

    if actuator_type == config_dictionary.linear_actuator_key:
        actuator_parameters = parameters_linear_actuators(
            actuator_config=actuator_config,
            number_of_actuators=number_of_actuators,
            initial_orientation=initial_orientation,
            log=log,
            heliostat_name=heliostat_name,
            device=device
        )
    elif actuator_type == config_dictionary.ideal_actuator_key:
        actuator_parameters = parameters_ideal_actuators(
            actuator_config=actuator_config,
            number_of_actuators=number_of_actuators,
            device=device,
        )
    else:
        raise ValueError(
            f"The actuator type: {actuator_type} is not yet implemented!"
        )
    
    return actuator_parameters

def parameters_linear_actuators(
                   actuator_config: h5py.File,
                   number_of_actuators: int,
                   initial_orientation: torch.Tensor,
                   log = logging.Logger,
                   heliostat_name: Optional[str]=None,
                   device: Union[torch.device, str] = "cuda"
) -> torch.Tensor:
    """
    Load actuator parameters for linear actuators from an HDF5 scenario file.

    Parameters
    ----------
    actuator_config : h5py.File
        The opened scenario HDF5 file containing the information.
    number_of_actuators : int
        The number of actuators used for a specific kinematic.
    initial_orientation : torch.Tensor
        The initial orientation of the heliostat.
    log : logging.Logger
        The logger for the scenario loader.
    heliostat_name : Optional[str]
        The heliostat name, only needed for individual heliostats, not prototypes (default is None).
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Raises
    ------
    ValueError
        If the file contains the wrong amount of actuators for a heliostat with a specific kinematic type.
        
    Returns
    -------
    torch.Tensor
        7 actuator parameters for for each linear actuator in the file.
    """
    device = torch.device(device)

    if len(actuator_config.keys()) != number_of_actuators:
        raise ValueError(
            f"This scenario file contains the wrong amount of actuators for this heliostat and its kinematic type."
            f"Expected {number_of_actuators} actuators, found {len(actuator_config.keys())} actuators."
        )
    
    actuator_parameters = torch.zeros((config_dictionary.linear_actuator_number_of_parameters, number_of_actuators), device=device)

    for index, actuator in enumerate(actuator_config.keys()):
        type=str(
            actuator_config[
                actuator
            ][config_dictionary.actuator_type_key][()].decode("utf-8")
        )
        clockwise_axis_movement=bool(
            actuator_config[
                actuator
            ][config_dictionary.actuator_clockwise_axis_movement][()]
        )
        increment = actuator_config.get(
            f"{actuator}/"
            f"{config_dictionary.actuator_parameters_key}/"
            f"{config_dictionary.actuator_increment}"
        )
        initial_stroke_length = actuator_config.get(
            f"{actuator}/"
            f"{config_dictionary.actuator_parameters_key}/"
            f"{config_dictionary.actuator_initial_stroke_length}"
        )
        offset = actuator_config.get(
            f"{actuator}/"
            f"{config_dictionary.actuator_parameters_key}/"
            f"{config_dictionary.actuator_offset}"
        )
        pivot_radius = actuator_config.get(
            f"{actuator}/"
            f"{config_dictionary.actuator_parameters_key}/"
            f"{config_dictionary.actuator_pivot_radius}"
        )
        initial_angle = actuator_config.get(
            f"{actuator}/"
            f"{config_dictionary.actuator_parameters_key}/"
            f"{config_dictionary.actuator_initial_angle}"
        )
        if increment is None:
            log.warning(
                f"No individual {config_dictionary.actuator_increment} set for {actuator}. Using default values!"
            )
        if initial_stroke_length is None:
            log.warning(
                f"No individual {config_dictionary.actuator_initial_stroke_length} set for {actuator} on "
                f"{heliostat_name}. "
                f"Using default values!"
            )
        if offset is None:
            log.warning(
                f"No individual {config_dictionary.actuator_offset} set for {actuator} on "
                f"{heliostat_name}. Using default values!"
            )
        if pivot_radius is None:
            log.warning(
                f"No individual {config_dictionary.actuator_pivot_radius} set for {actuator} on "
                f"{heliostat_name}. Using default values!"
            )
        if initial_angle is None:
            log.warning(
                f"No individual {config_dictionary.actuator_initial_angle} set for {actuator} on "
                f"{heliostat_name}. Using default values!"
            )
        actuator_parameters[0, index] = 0 if type==config_dictionary.linear_actuator_key else 1
        
        actuator_parameters[1, index] = 0 if not clockwise_axis_movement else 1

        actuator_parameters[2, index] = (
            torch.tensor(increment[()], dtype=torch.float, device=device) 
            if increment 
            else torch.tensor(0.0, dtype=torch.float, device=device)
        )
        actuator_parameters[3, index] = (
            torch.tensor(
                initial_stroke_length[()], dtype=torch.float, device=device
            )
            if initial_stroke_length
            else torch.tensor(0.0, dtype=torch.float, device=device)
        )
        actuator_parameters[4, index] = (
            torch.tensor(offset[()], dtype=torch.float, device=device)
            if offset
            else torch.tensor(0.0, dtype=torch.float, device=device)
        )
        actuator_parameters[5, index] = (
            torch.tensor(pivot_radius[()], dtype=torch.float, device=device)
            if pivot_radius
            else torch.tensor(0.0, dtype=torch.float, device=device)
        )
        actuator_parameters[6, index] = (
            torch.tensor(
                initial_angle[()], dtype=torch.float, device=device
            )
            if initial_angle
            else torch.tensor(0.0, dtype=torch.float, device=device)
        )
    
    # Adapt initial angle of actuator one according to kinematic initial orientation.
    # ARTIST always expects heliostats to be initially oriented to the south [0.0, -1.0, 0.0] (in ENU).
    # The first actuator always rotates along the east-axis.
    # Since the actuator coordinate system is relative to the heliostat orientation, the initial angle
    # of actuator one needs to be transformed accordingly.
    actuator_parameters[6, 0] = utils.transform_initial_angle(
        initial_angle=actuator_parameters[6, 0].unsqueeze(0),
        initial_orientation=initial_orientation,
        device=device,
    )
    
    return actuator_parameters

def parameters_ideal_actuators(
    actuator_config: h5py.File,
    number_of_actuators: int,
    device: Union[torch.device, str] = "cuda"
) -> torch.Tensor:
    """
    Load actuator parameters for ideal actuators from an HDF5 scenario file.

    Parameters
    ----------
    actuator_config : h5py.File
        The opened scenario HDF5 file containing the information.
    number_of_actuators : int
        The number of actuators used for a specific kinematic.
    heliostat_name : Optional[str]
        The heliostat name, only needed for individual heliostats, not prototypes (default is None).
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Raises
    ------
    ValueError
        If the file contains the wrong amount of actuators for a heliostat with a specific kinematic type.
        
    Returns
    -------
    torch.Tensor
        2 actuator parameters for for each ideal actuator in the file.
    """
    
    device = torch.device(device)

    if len(actuator_config.keys()) != number_of_actuators:
        raise ValueError(
            f"This scenario file contains the wrong amount of actuators for this heliostat and its kinematic type."
            f"Expected {number_of_actuators} actuators, found {len(actuator_config.keys())} actuators."
        )
    
    actuator_parameters = torch.zeros((config_dictionary.ideal_actuator_number_of_parameters, number_of_actuators), device=device)

    for index, actuator in enumerate(actuator_config.keys()):
        type=str(
            actuator_config[
                actuator
            ][config_dictionary.actuator_type_key][()].decode("utf-8")
        )
        clockwise_axis_movement=bool(
            actuator_config[
                actuator
            ][config_dictionary.actuator_clockwise_axis_movement][()]
        )
        
        actuator_parameters[0, index] = 0 if type==config_dictionary.linear_actuator_key else 1
        
        actuator_parameters[1, index] = 0 if not clockwise_axis_movement else 1

    return actuator_parameters
