import logging
from typing import Union

import h5py
import torch
from typing_extensions import Self

from artist.field.heliostat_field import HeliostatField
from artist.field.receiver_field import ReceiverField
from artist.scene.light_source_array import LightSourceArray
from artist.util import config_dictionary
from artist.util.configuration_classes import (
    ActuatorConfig,
    ActuatorListConfig,
    ActuatorParameters,
    FacetConfig,
    KinematicDeviations,
    KinematicLoadConfig,
    KinematicOffsets,
    SurfaceConfig,
)

log = logging.getLogger(__name__)
"""A logger for the scenario."""

class Scenario:
    """
    Define a scenario loaded by ARTIST.

    Attributes
    ----------
    power_plant_position : torch.Tensor
        The position of the power plant in lat, lon, alt.
    receivers : ReceiverField
        A list of receivers included in the scenario.
    light_sources : LightSourceArray
        A list of light sources included in the scenario.
    heliostats : HeliostatField
        The heliostat field for the scenario.

    Methods
    -------
    load_scenario_from_hdf5()
        Class method to initialize the scenario from an HDF5 file.
    """

    def __init__(
        self,
        power_plant_position: torch.Tensor,
        receivers: ReceiverField,
        light_sources: LightSourceArray,
        heliostat_field: HeliostatField,
    ) -> None:
        """
        Initialize the scenario.

        A scenario defines the physical objects and scene to be used by ``ARTIST``. Therefore, a scenario contains at
        least one receiver, at least one light source and at least one heliostat in a heliostat field. ``ARTIST`` also
        supports scenarios that contain multiple receivers, multiple light sources, and multiple heliostats.

        Parameters
        ----------
        power_plant_position : torch.Tensor,
            The position of the power plant in lat, lon, alt.
        receivers : ReceiverField
            A list of receivers included in the scenario.
        light_sources : LightSourceArray
            A list of light sources included in the scenario.
        heliostat_field : HeliostatField
            A field of heliostats included in the scenario.
        """
        self.power_plant_position = power_plant_position
        self.receivers = receivers
        self.light_sources = light_sources
        self.heliostats = heliostat_field

    @classmethod
    def load_scenario_from_hdf5(
        cls, scenario_file: h5py.File, device: Union[torch.device, str] = "cuda"
    ) -> Self:
        """
        Class method to load the scenario from an HDF5 file.

        Parameters
        ----------
        scenario_file : h5py.File
            The config file containing all the information about the scenario being loaded.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        Scenario
            The ``ARTIST`` scenario loaded from the HDF5 file.
        """
        log.info(
            f"Loading an ``ARTIST`` scenario HDF5 file. This scenario file is version {scenario_file.attrs['version']}."
        )
        device = torch.device(device)
        power_plant_position = torch.tensor(
            scenario_file[config_dictionary.power_plant_key][
                config_dictionary.power_plant_position
            ][()]
        )
        receivers = ReceiverField.from_hdf5(config_file=scenario_file, device=device)
        light_sources = LightSourceArray.from_hdf5(
            config_file=scenario_file, device=device
        )

        facets_list = [
            FacetConfig(
                facet_key="",
                control_points=torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.surface_prototype_key
                    ][config_dictionary.facets_key][facet][
                        config_dictionary.facet_control_points
                    ][()],
                    dtype=torch.float,
                    device=device,
                ),
                degree_e=int(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.surface_prototype_key
                    ][config_dictionary.facets_key][facet][
                        config_dictionary.facet_degree_e
                    ][()]
                ),
                degree_n=int(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.surface_prototype_key
                    ][config_dictionary.facets_key][facet][
                        config_dictionary.facet_degree_n
                    ][()]
                ),
                number_eval_points_e=int(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.surface_prototype_key
                    ][config_dictionary.facets_key][facet][
                        config_dictionary.facet_number_eval_e
                    ][()]
                ),
                number_eval_points_n=int(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.surface_prototype_key
                    ][config_dictionary.facets_key][facet][
                        config_dictionary.facet_number_eval_n
                    ][()]
                ),
                width=float(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.surface_prototype_key
                    ][config_dictionary.facets_key][facet][
                        config_dictionary.facets_width
                    ][()]
                ),
                height=float(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.surface_prototype_key
                    ][config_dictionary.facets_key][facet][
                        config_dictionary.facets_height
                    ][()]
                ),
                translation_vector=torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.surface_prototype_key
                    ][config_dictionary.facets_key][facet][
                        config_dictionary.facets_translation_vector
                    ][()],
                    dtype=torch.float,
                    device=device,
                ),
                canting_e=torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.surface_prototype_key
                    ][config_dictionary.facets_key][facet][
                        config_dictionary.facets_canting_e
                    ][()],
                    dtype=torch.float,
                    device=device,
                ),
                canting_n=torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.surface_prototype_key
                    ][config_dictionary.facets_key][facet][
                        config_dictionary.facets_canting_n
                    ][()],
                    dtype=torch.float,
                    device=device,
                ),
            )
            for facet in scenario_file[config_dictionary.prototype_key][
                config_dictionary.surface_prototype_key
            ][config_dictionary.facets_key].keys()
        ]
        surface_prototype = SurfaceConfig(facets_list=facets_list)

        # Create kinematic prototype.
        kinematic_initial_orientation_offset_e = scenario_file.get(
            f"{config_dictionary.prototype_key}/{config_dictionary.kinematic_prototype_key}/"
            f"{config_dictionary.kinematic_offsets_key}/{config_dictionary.kinematic_initial_orientation_offset_e}"
        )
        kinematic_initial_orientation_offset_n = scenario_file.get(
            f"{config_dictionary.prototype_key}/{config_dictionary.kinematic_prototype_key}/"
            f"{config_dictionary.kinematic_offsets_key}/{config_dictionary.kinematic_initial_orientation_offset_n}"
        )
        kinematic_initial_orientation_offset_u = scenario_file.get(
            f"{config_dictionary.prototype_key}/{config_dictionary.kinematic_prototype_key}/"
            f"{config_dictionary.kinematic_offsets_key}/{config_dictionary.kinematic_initial_orientation_offset_u}"
        )
        if kinematic_initial_orientation_offset_e is None:
            log.warning(
                f"No kinematic prototype {config_dictionary.kinematic_initial_orientation_offset_e} set."
                f"Using default values!"
            )
        if kinematic_initial_orientation_offset_n is None:
            log.warning(
                f"No kinematic prototype {config_dictionary.kinematic_initial_orientation_offset_n} set."
                f"Using default values!"
            )
        if kinematic_initial_orientation_offset_u is None:
            log.warning(
                f"No kinematic prototype {config_dictionary.kinematic_initial_orientation_offset_u} set."
                f"Using default values!"
            )
        kinematic_offsets = KinematicOffsets(
            kinematic_initial_orientation_offset_e=(
                torch.tensor(
                    kinematic_initial_orientation_offset_e[()],
                    dtype=torch.float,
                    device=device,
                )
                if kinematic_initial_orientation_offset_e
                else torch.tensor(0.0, dtype=torch.float, device=device)
            ),
            kinematic_initial_orientation_offset_n=(
                torch.tensor(
                    kinematic_initial_orientation_offset_n[()],
                    dtype=torch.float,
                    device=device,
                )
                if kinematic_initial_orientation_offset_n
                else torch.tensor(0.0, dtype=torch.float, device=device)
            ),
            kinematic_initial_orientation_offset_u=(
                torch.tensor(
                    kinematic_initial_orientation_offset_u[()],
                    dtype=torch.float,
                    device=device,
                )
                if kinematic_initial_orientation_offset_u
                else torch.tensor(0.0, dtype=torch.float, device=device)
            ),
        )

        first_joint_translation_e = scenario_file.get(
            f"{config_dictionary.prototype_key}/"
            f"{config_dictionary.kinematic_prototype_key}/"
            f"{config_dictionary.kinematic_deviations_key}/"
            f"{config_dictionary.first_joint_translation_e}"
        )
        first_joint_translation_n = scenario_file.get(
            f"{config_dictionary.prototype_key}/"
            f"{config_dictionary.kinematic_prototype_key}/"
            f"{config_dictionary.kinematic_deviations_key}/"
            f"{config_dictionary.first_joint_translation_n}"
        )
        first_joint_translation_u = scenario_file.get(
            f"{config_dictionary.prototype_key}/"
            f"{config_dictionary.kinematic_prototype_key}/"
            f"{config_dictionary.kinematic_deviations_key}/"
            f"{config_dictionary.first_joint_translation_u}"
        )
        first_joint_tilt_e = scenario_file.get(
            f"{config_dictionary.prototype_key}/"
            f"{config_dictionary.kinematic_prototype_key}/"
            f"{config_dictionary.kinematic_deviations_key}/"
            f"{config_dictionary.first_joint_tilt_e}"
        )
        first_joint_tilt_n = scenario_file.get(
            f"{config_dictionary.prototype_key}/"
            f"{config_dictionary.kinematic_prototype_key}/"
            f"{config_dictionary.kinematic_deviations_key}/"
            f"{config_dictionary.first_joint_tilt_n}"
        )
        first_joint_tilt_u = scenario_file.get(
            f"{config_dictionary.prototype_key}/"
            f"{config_dictionary.kinematic_prototype_key}/"
            f"{config_dictionary.kinematic_deviations_key}/"
            f"{config_dictionary.first_joint_tilt_u}"
        )
        second_joint_translation_e = scenario_file.get(
            f"{config_dictionary.prototype_key}/"
            f"{config_dictionary.kinematic_prototype_key}/"
            f"{config_dictionary.kinematic_deviations_key}/"
            f"{config_dictionary.second_joint_translation_e}"
        )
        second_joint_translation_n = scenario_file.get(
            f"{config_dictionary.prototype_key}/"
            f"{config_dictionary.kinematic_prototype_key}/"
            f"{config_dictionary.kinematic_deviations_key}/"
            f"{config_dictionary.second_joint_translation_n}"
        )
        second_joint_translation_u = scenario_file.get(
            f"{config_dictionary.prototype_key}/"
            f"{config_dictionary.kinematic_prototype_key}/"
            f"{config_dictionary.kinematic_deviations_key}/"
            f"{config_dictionary.second_joint_translation_u}"
        )
        second_joint_tilt_e = scenario_file.get(
            f"{config_dictionary.prototype_key}/"
            f"{config_dictionary.kinematic_prototype_key}/"
            f"{config_dictionary.kinematic_deviations_key}/"
            f"{config_dictionary.second_joint_tilt_e}"
        )
        second_joint_tilt_n = scenario_file.get(
            f"{config_dictionary.prototype_key}/"
            f"{config_dictionary.kinematic_prototype_key}/"
            f"{config_dictionary.kinematic_deviations_key}/"
            f"{config_dictionary.second_joint_tilt_n}"
        )
        second_joint_tilt_u = scenario_file.get(
            f"{config_dictionary.prototype_key}/"
            f"{config_dictionary.kinematic_prototype_key}/"
            f"{config_dictionary.kinematic_deviations_key}/"
            f"{config_dictionary.second_joint_tilt_u}"
        )
        concentrator_translation_e = scenario_file.get(
            f"{config_dictionary.prototype_key}/"
            f"{config_dictionary.kinematic_prototype_key}/"
            f"{config_dictionary.kinematic_deviations_key}/"
            f"{config_dictionary.concentrator_translation_e}"
        )
        concentrator_translation_n = scenario_file.get(
            f"{config_dictionary.prototype_key}/"
            f"{config_dictionary.kinematic_prototype_key}/"
            f"{config_dictionary.kinematic_deviations_key}/"
            f"{config_dictionary.concentrator_translation_n}"
        )
        concentrator_translation_u = scenario_file.get(
            f"{config_dictionary.prototype_key}/"
            f"{config_dictionary.kinematic_prototype_key}/"
            f"{config_dictionary.kinematic_deviations_key}/"
            f"{config_dictionary.concentrator_translation_u}"
        )
        concentrator_tilt_e = scenario_file.get(
            f"{config_dictionary.prototype_key}/"
            f"{config_dictionary.kinematic_prototype_key}/"
            f"{config_dictionary.kinematic_deviations_key}/"
            f"{config_dictionary.concentrator_tilt_e}"
        )
        concentrator_tilt_n = scenario_file.get(
            f"{config_dictionary.prototype_key}/"
            f"{config_dictionary.kinematic_prototype_key}/"
            f"{config_dictionary.kinematic_deviations_key}/"
            f"{config_dictionary.concentrator_tilt_n}"
        )
        concentrator_tilt_u = scenario_file.get(
            f"{config_dictionary.prototype_key}/"
            f"{config_dictionary.kinematic_prototype_key}/"
            f"{config_dictionary.kinematic_deviations_key}/"
            f"{config_dictionary.concentrator_tilt_u}"
        )
        if first_joint_translation_e is None:
            log.warning(
                f"No prototype kinematic {config_dictionary.first_joint_translation_e} set. Using default values!"
            )
        if first_joint_translation_n is None:
            log.warning(
                f"No prototype kinematic {config_dictionary.first_joint_translation_n} set. Using default values!"
            )
        if first_joint_translation_u is None:
            log.warning(
                f"No prototype kinematic {config_dictionary.first_joint_translation_u} set. Using default values!"
            )
        if first_joint_tilt_e is None:
            log.warning(
                f"No prototype kinematic {config_dictionary.first_joint_tilt_e} set. Using default values!"
            )
        if first_joint_tilt_n is None:
            log.warning(
                f"No prototype kinematic {config_dictionary.first_joint_tilt_n} set. Using default values!"
            )
        if first_joint_tilt_u is None:
            log.warning(
                f"No prototype kinematic {config_dictionary.first_joint_tilt_u} set. Using default values!"
            )
        if second_joint_translation_e is None:
            log.warning(
                f"No prototype kinematic {config_dictionary.second_joint_translation_e} set. Using default values!"
            )
        if second_joint_translation_n is None:
            log.warning(
                f"No prototype kinematic {config_dictionary.second_joint_translation_n} set. Using default values!"
            )
        if second_joint_translation_u is None:
            log.warning(
                f"No prototype kinematic {config_dictionary.second_joint_translation_u} set. Using default values!"
            )
        if second_joint_tilt_e is None:
            log.warning(
                f"No prototype kinematic {config_dictionary.second_joint_tilt_e} set. Using default values!"
            )
        if second_joint_tilt_n is None:
            log.warning(
                f"No prototype kinematic {config_dictionary.second_joint_tilt_n} set. Using default values!"
            )
        if second_joint_tilt_u is None:
            log.warning(
                f"No prototype kinematic {config_dictionary.second_joint_tilt_u} set. Using default values!"
            )
        if concentrator_translation_e is None:
            log.warning(
                f"No prototype kinematic {config_dictionary.concentrator_translation_e} set. Using default values!"
            )
        if concentrator_translation_n is None:
            log.warning(
                f"No prototype kinematic {config_dictionary.concentrator_translation_n} set. Using default values!"
            )
        if concentrator_translation_u is None:
            log.warning(
                f"No prototype kinematic {config_dictionary.concentrator_translation_u} set. Using default values!"
            )
        if concentrator_tilt_e is None:
            log.warning(
                f"No prototype kinematic {config_dictionary.concentrator_tilt_e} set. Using default values!"
            )
        if concentrator_tilt_n is None:
            log.warning(
                f"No prototype kinematic {config_dictionary.concentrator_tilt_n} set. Using default values!"
            )
        if concentrator_tilt_u is None:
            log.warning(
                f"No prototype kinematic {config_dictionary.concentrator_tilt_u} set. Using default values!"
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
                torch.tensor(first_joint_tilt_e[()], dtype=torch.float, device=device)
                if first_joint_tilt_e
                else torch.tensor(0.0, dtype=torch.float, device=device)
            ),
            first_joint_tilt_n=(
                torch.tensor(first_joint_tilt_n[()], dtype=torch.float, device=device)
                if first_joint_tilt_n
                else torch.tensor(0.0, dtype=torch.float, device=device)
            ),
            first_joint_tilt_u=(
                torch.tensor(first_joint_tilt_u[()], dtype=torch.float, device=device)
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
                torch.tensor(second_joint_tilt_e[()], dtype=torch.float, device=device)
                if second_joint_tilt_e
                else torch.tensor(0.0, dtype=torch.float, device=device)
            ),
            second_joint_tilt_n=(
                torch.tensor(second_joint_tilt_n[()], dtype=torch.float, device=device)
                if second_joint_tilt_n
                else torch.tensor(0.0, dtype=torch.float, device=device)
            ),
            second_joint_tilt_u=(
                torch.tensor(second_joint_tilt_u[()], dtype=torch.float, device=device)
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
                torch.tensor(concentrator_tilt_e[()], dtype=torch.float, device=device)
                if concentrator_tilt_e
                else torch.tensor(0.0, dtype=torch.float, device=device)
            ),
            concentrator_tilt_n=(
                torch.tensor(concentrator_tilt_n[()], dtype=torch.float, device=device)
                if concentrator_tilt_n
                else torch.tensor(0.0, dtype=torch.float, device=device)
            ),
            concentrator_tilt_u=(
                torch.tensor(concentrator_tilt_u[()], dtype=torch.float, device=device)
                if concentrator_tilt_u
                else torch.tensor(0.0, dtype=torch.float, device=device)
            ),
        )
        kinematic_prototype = KinematicLoadConfig(
            kinematic_type=str(
                scenario_file[config_dictionary.prototype_key][
                    config_dictionary.kinematic_prototype_key
                ][config_dictionary.kinematic_type][()].decode("utf-8")
            ),
            kinematic_initial_orientation_offsets=kinematic_offsets,
            kinematic_deviations=kinematic_deviations,
        )

        # Create actuator prototype.
        actuator_list = []
        for ac in scenario_file[config_dictionary.prototype_key][
            config_dictionary.actuator_prototype_key
        ].keys():
            increment = scenario_file.get(
                f"{config_dictionary.prototype_key}/"
                f"{config_dictionary.actuator_prototype_key}/{ac}/"
                f"{config_dictionary.actuator_parameters_key}/"
                f"{config_dictionary.actuator_increment}"
            )
            initial_stroke_length = scenario_file.get(
                f"{config_dictionary.prototype_key}/"
                f"{config_dictionary.actuator_prototype_key}/{ac}/"
                f"{config_dictionary.actuator_parameters_key}/"
                f"{config_dictionary.actuator_initial_stroke_length}"
            )
            offset = scenario_file.get(
                f"{config_dictionary.prototype_key}/"
                f"{config_dictionary.actuator_prototype_key}/{ac}/"
                f"{config_dictionary.actuator_parameters_key}/"
                f"{config_dictionary.actuator_offset}"
            )
            radius = scenario_file.get(
                f"{config_dictionary.prototype_key}/"
                f"{config_dictionary.actuator_prototype_key}/{ac}/"
                f"{config_dictionary.actuator_parameters_key}/"
                f"{config_dictionary.actuator_radius}"
            )
            phi_0 = scenario_file.get(
                f"{config_dictionary.prototype_key}/"
                f"{config_dictionary.actuator_prototype_key}/{ac}/"
                f"{config_dictionary.actuator_parameters_key}/"
                f"{config_dictionary.actuator_phi_0}"
            )
            if increment is None:
                log.warning(
                    f"No prototype {config_dictionary.actuator_increment} set for {ac}. Using default values!"
                )
            if initial_stroke_length is None:
                log.warning(
                    f"No prototype {config_dictionary.actuator_initial_stroke_length} set for {ac}. "
                    f"Using default values!"
                )
            if offset is None:
                log.warning(
                    f"No prototype {config_dictionary.actuator_offset} set for {ac}. Using default values!"
                )
            if radius is None:
                log.warning(
                    f"No prototype {config_dictionary.actuator_radius} set for {ac}. Using default values!"
                )
            if phi_0 is None:
                log.warning(
                    f"No prototype {config_dictionary.actuator_phi_0} set for {ac}. Using default values!"
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
                radius=(
                    torch.tensor(radius[()], dtype=torch.float, device=device)
                    if radius
                    else torch.tensor(0.0, dtype=torch.float, device=device)
                ),
                phi_0=(
                    torch.tensor(phi_0[()], dtype=torch.float, device=device)
                    if phi_0
                    else torch.tensor(0.0, dtype=torch.float, device=device)
                ),
            )
            actuator_list.append(
                ActuatorConfig(
                    actuator_key="",
                    actuator_type=str(
                        scenario_file[config_dictionary.prototype_key][
                            config_dictionary.actuator_prototype_key
                        ][ac][config_dictionary.actuator_type_key][()].decode("utf-8")
                    ),
                    actuator_clockwise=bool(
                        scenario_file[config_dictionary.prototype_key][
                            config_dictionary.actuator_prototype_key
                        ][ac][config_dictionary.actuator_clockwise][()]
                    ),
                    actuator_parameters=actuator_parameters,
                )
            )
        actuator_prototype = ActuatorListConfig(actuator_list=actuator_list)

        heliostat_field = HeliostatField.from_hdf5(
            config_file=scenario_file,
            prototype_surface=surface_prototype,
            prototype_kinematic=kinematic_prototype,
            prototype_actuator=actuator_prototype,
            device=device,
        )

        return cls(
            power_plant_position=power_plant_position,
            receivers=receivers,
            light_sources=light_sources,
            heliostat_field=heliostat_field,
        )

    def __repr__(self) -> str:
        """Return a string representation of the scenario."""
        return (
            f"ARTIST Scenario containing:\n\tReceivers: {len(self.receivers.receiver_list)}, \tLight Sources: "
            f"{len(self.light_sources.light_source_list)},\t Heliostats: {len(self.heliostats.heliostat_list)}"
        )
