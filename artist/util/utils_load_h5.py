import logging
from typing import Optional, Union
from artist.field import surface
from artist.util import config_dictionary
from artist.util.configuration_classes import FacetConfig, SurfaceConfig
import h5py
import torch


def load_surface_config(prototype: bool,
                        scenario_file: h5py.File,
                        device: Union[torch.device, str] = "cuda"):
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


def load_kinematic_deviations_rigid_body(prototype: bool,
                                         scenario_file: h5py.File,
                                         log: logging.Logger,
                                         heliostat_name: Optional[str]=None, 
                                         device: Union[torch.device, str] = "cuda"):
    device = torch.device(device)
    if prototype:
        kinematic_deviations_config = scenario_file[config_dictionary.prototype_key][
            config_dictionary.kinematic_prototype_key
        ][config_dictionary.kinematic_deviations_key]
    else:
        kinematic_deviations_config = scenario_file[config_dictionary.heliostat_kinematic_key][config_dictionary.kinematic_deviations_key]

    kinematic_deviations = torch.zeros(18, dtype=torch.float, device=device)
    
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


def load_kinematic_deviations_new_kinematic():
    pass


def load_actuators(prototype: bool,
                   scenario_file: h5py.File,
                   log = logging.Logger,
                   heliostat_name: Optional[str]=None,
                   device: Union[torch.device, str] = "cuda"):
    device = torch.device(device)
    if prototype:
        actuator_config = scenario_file[config_dictionary.prototype_key][
            config_dictionary.actuators_prototype_key
        ]
    else:
        actuator_config = scenario_file[config_dictionary.heliostat_actuator_key]

    actuator_parameters = torch.zeros((7, 2), device=device)

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
    
    return actuator_parameters
