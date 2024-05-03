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
    KinematicConfig,
    KinematicDeviations,
    KinematicOffsets,
    SurfaceConfig,
)


class Scenario:
    """
    This class represents a scenario loaded by ARTIST.

    Attributes
    ----------
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
        receivers: ReceiverField,
        light_sources: LightSourceArray,
        heliostat_field: HeliostatField,
    ) -> None:
        """
        Initialize the scenario.

        Parameters
        ----------
        receivers : ReceiverField
            A list of receivers included in the scenario.
        light_sources : LightSourceArray
            A list of light sources included in the scenario.
        heliostat_field : HeliostatField
            A field of heliostats included in the scenario.
        """
        self.receivers = receivers
        self.light_sources = light_sources
        self.heliostats = heliostat_field

    @classmethod
    def load_scenario_from_hdf5(cls, scenario_file: h5py.File) -> Self:
        """
        Class method to load the scenario from an HDF5 file.

        Parameters
        ----------
        scenario_file : h5py.File
            The config file containing all the information about the scenario being loaded.

        Returns
        -------
        Scenario
            The ARTIST scenario loaded from the HDF5 file.
        """
        receivers = ReceiverField.from_hdf5(config_file=scenario_file)
        light_sources = LightSourceArray.from_hdf5(config_file=scenario_file)

        facets_list = [
            FacetConfig(
                facet_key="",
                control_points_e=torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.surface_prototype_key
                    ][config_dictionary.facets_key][facet][
                        config_dictionary.facet_control_points_e
                    ][()],
                    dtype=torch.float,
                ),
                control_points_u=torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.surface_prototype_key
                    ][config_dictionary.facets_key][facet][
                        config_dictionary.facet_control_points_u
                    ][()],
                    dtype=torch.float,
                ),
                knots_e=torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.surface_prototype_key
                    ][config_dictionary.facets_key][facet][
                        config_dictionary.facet_knots_e
                    ][()],
                    dtype=torch.float,
                ),
                knots_u=torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.surface_prototype_key
                    ][config_dictionary.facets_key][facet][
                        config_dictionary.facet_knots_u
                    ][()],
                    dtype=torch.float,
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
                position=torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.surface_prototype_key
                    ][config_dictionary.facets_key][facet][
                        config_dictionary.facets_position
                    ][()],
                    dtype=torch.float,
                ),
                canting_e=torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.surface_prototype_key
                    ][config_dictionary.facets_key][facet][
                        config_dictionary.facets_canting_e
                    ][()],
                    dtype=torch.float,
                ),
                canting_u=torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.surface_prototype_key
                    ][config_dictionary.facets_key][facet][
                        config_dictionary.facets_canting_u
                    ][()],
                    dtype=torch.float,
                ),
            )
            for facet in scenario_file[config_dictionary.prototype_key][
                config_dictionary.surface_prototype_key
            ][config_dictionary.facets_key].keys()
        ]
        surface_prototype = SurfaceConfig(facets_list=facets_list)

        # Create kinematic prototype
        kinematic_initial_orientation_offset_e = torch.tensor(0.0)
        kinematic_initial_orientation_offset_n = torch.tensor(0.0)
        kinematic_initial_orientation_offset_u = torch.tensor(0.0)
        if (
            config_dictionary.kinematic_offsets_key
            in scenario_file[config_dictionary.prototype_key][
                config_dictionary.kinematic_prototype_key
            ].keys()
        ):
            if (
                config_dictionary.kinematic_initial_orientation_offset_e
                in scenario_file[config_dictionary.prototype_key][
                    config_dictionary.kinematic_prototype_key
                ][config_dictionary.kinematic_offsets_key].keys()
            ):
                kinematic_initial_orientation_offset_e = torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.kinematic_prototype_key
                    ][config_dictionary.kinematic_offsets_key][
                        config_dictionary.kinematic_initial_orientation_offset_e
                    ][()],
                    dtype=torch.float,
                )
            if (
                config_dictionary.kinematic_initial_orientation_offset_n
                in scenario_file[config_dictionary.prototype_key][
                    config_dictionary.kinematic_prototype_key
                ][config_dictionary.kinematic_offsets_key].keys()
            ):
                kinematic_initial_orientation_offset_n = torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.kinematic_prototype_key
                    ][config_dictionary.kinematic_offsets_key][
                        config_dictionary.kinematic_initial_orientation_offset_n
                    ][()],
                    dtype=torch.float,
                )
            if (
                config_dictionary.kinematic_initial_orientation_offset_u
                in scenario_file[config_dictionary.prototype_key][
                    config_dictionary.kinematic_prototype_key
                ][config_dictionary.kinematic_offsets_key].keys()
            ):
                kinematic_initial_orientation_offset_u = torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.kinematic_prototype_key
                    ][config_dictionary.kinematic_offsets_key][
                        config_dictionary.kinematic_initial_orientation_offset_u
                    ][()],
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
            in scenario_file[config_dictionary.prototype_key][
                config_dictionary.kinematic_prototype_key
            ].keys()
        ):
            if (
                config_dictionary.first_joint_translation_e
                in scenario_file[config_dictionary.prototype_key][
                    config_dictionary.kinematic_prototype_key
                ][config_dictionary.kinematic_deviations_key].keys()
            ):
                first_joint_translation_e = torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.kinematic_prototype_key
                    ][config_dictionary.kinematic_deviations_key][
                        config_dictionary.first_joint_translation_e
                    ][()],
                    dtype=torch.float,
                )
            if (
                config_dictionary.first_joint_translation_n
                in scenario_file[config_dictionary.prototype_key][
                    config_dictionary.kinematic_prototype_key
                ][config_dictionary.kinematic_deviations_key].keys()
            ):
                first_joint_translation_n = torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.kinematic_prototype_key
                    ][config_dictionary.kinematic_deviations_key][
                        config_dictionary.first_joint_translation_n
                    ][()],
                    dtype=torch.float,
                )
            if (
                config_dictionary.first_joint_translation_u
                in scenario_file[config_dictionary.prototype_key][
                    config_dictionary.kinematic_prototype_key
                ][config_dictionary.kinematic_deviations_key].keys()
            ):
                first_joint_translation_u = torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.kinematic_prototype_key
                    ][config_dictionary.kinematic_deviations_key][
                        config_dictionary.first_joint_translation_u
                    ][()],
                    dtype=torch.float,
                )
            if (
                config_dictionary.first_joint_tilt_e
                in scenario_file[config_dictionary.prototype_key][
                    config_dictionary.kinematic_prototype_key
                ][config_dictionary.kinematic_deviations_key].keys()
            ):
                first_joint_tilt_e = torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.kinematic_prototype_key
                    ][config_dictionary.kinematic_deviations_key][
                        config_dictionary.first_joint_tilt_e
                    ][()],
                    dtype=torch.float,
                )
            if (
                config_dictionary.first_joint_tilt_n
                in scenario_file[config_dictionary.prototype_key][
                    config_dictionary.kinematic_prototype_key
                ][config_dictionary.kinematic_deviations_key].keys()
            ):
                first_joint_tilt_n = torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.kinematic_prototype_key
                    ][config_dictionary.kinematic_deviations_key][
                        config_dictionary.first_joint_tilt_n
                    ][()],
                    dtype=torch.float,
                )
            if (
                config_dictionary.first_joint_tilt_u
                in scenario_file[config_dictionary.prototype_key][
                    config_dictionary.kinematic_prototype_key
                ][config_dictionary.kinematic_deviations_key].keys()
            ):
                first_joint_tilt_u = torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.kinematic_prototype_key
                    ][config_dictionary.kinematic_deviations_key][
                        config_dictionary.first_joint_tilt_u
                    ][()],
                    dtype=torch.float,
                )
            if (
                config_dictionary.second_joint_translation_e
                in scenario_file[config_dictionary.prototype_key][
                    config_dictionary.kinematic_prototype_key
                ][config_dictionary.kinematic_deviations_key].keys()
            ):
                second_joint_translation_e = torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.kinematic_prototype_key
                    ][config_dictionary.kinematic_deviations_key][
                        config_dictionary.second_joint_translation_e
                    ][()],
                    dtype=torch.float,
                )
            if (
                config_dictionary.second_joint_translation_n
                in scenario_file[config_dictionary.prototype_key][
                    config_dictionary.kinematic_prototype_key
                ][config_dictionary.kinematic_deviations_key].keys()
            ):
                second_joint_translation_n = torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.kinematic_prototype_key
                    ][config_dictionary.kinematic_deviations_key][
                        config_dictionary.second_joint_translation_n
                    ][()],
                    dtype=torch.float,
                )
            if (
                config_dictionary.second_joint_translation_u
                in scenario_file[config_dictionary.prototype_key][
                    config_dictionary.kinematic_prototype_key
                ][config_dictionary.kinematic_deviations_key].keys()
            ):
                second_joint_translation_u = torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.kinematic_prototype_key
                    ][config_dictionary.kinematic_deviations_key][
                        config_dictionary.second_joint_translation_u
                    ][()],
                    dtype=torch.float,
                )
            if (
                config_dictionary.second_joint_tilt_e
                in scenario_file[config_dictionary.prototype_key][
                    config_dictionary.kinematic_prototype_key
                ][config_dictionary.kinematic_deviations_key].keys()
            ):
                second_joint_tilt_e = torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.kinematic_prototype_key
                    ][config_dictionary.kinematic_deviations_key][
                        config_dictionary.second_joint_tilt_e
                    ][()],
                    dtype=torch.float,
                )
            if (
                config_dictionary.second_joint_tilt_n
                in scenario_file[config_dictionary.prototype_key][
                    config_dictionary.kinematic_prototype_key
                ][config_dictionary.kinematic_deviations_key].keys()
            ):
                second_joint_tilt_n = torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.kinematic_prototype_key
                    ][config_dictionary.kinematic_deviations_key][
                        config_dictionary.second_joint_tilt_n
                    ][()],
                    dtype=torch.float,
                )
            if (
                config_dictionary.second_joint_tilt_u
                in scenario_file[config_dictionary.prototype_key][
                    config_dictionary.kinematic_prototype_key
                ][config_dictionary.kinematic_deviations_key].keys()
            ):
                second_joint_tilt_u = torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.kinematic_prototype_key
                    ][config_dictionary.kinematic_deviations_key][
                        config_dictionary.second_joint_tilt_u
                    ][()],
                    dtype=torch.float,
                )
            if (
                config_dictionary.concentrator_translation_e
                in scenario_file[config_dictionary.prototype_key][
                    config_dictionary.kinematic_prototype_key
                ][config_dictionary.kinematic_deviations_key].keys()
            ):
                concentrator_translation_e = torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.kinematic_prototype_key
                    ][config_dictionary.kinematic_deviations_key][
                        config_dictionary.concentrator_translation_e
                    ][()],
                    dtype=torch.float,
                )
            if (
                config_dictionary.concentrator_translation_n
                in scenario_file[config_dictionary.prototype_key][
                    config_dictionary.kinematic_prototype_key
                ][config_dictionary.kinematic_deviations_key].keys()
            ):
                concentrator_translation_n = torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.kinematic_prototype_key
                    ][config_dictionary.kinematic_deviations_key][
                        config_dictionary.concentrator_translation_n
                    ][()],
                    dtype=torch.float,
                )
            if (
                config_dictionary.concentrator_translation_u
                in scenario_file[config_dictionary.prototype_key][
                    config_dictionary.kinematic_prototype_key
                ][config_dictionary.kinematic_deviations_key].keys()
            ):
                concentrator_translation_u = torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.kinematic_prototype_key
                    ][config_dictionary.kinematic_deviations_key][
                        config_dictionary.concentrator_translation_u
                    ][()],
                    dtype=torch.float,
                )
            if (
                config_dictionary.concentrator_tilt_e
                in scenario_file[config_dictionary.prototype_key][
                    config_dictionary.kinematic_prototype_key
                ][config_dictionary.kinematic_deviations_key].keys()
            ):
                concentrator_tilt_e = torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.kinematic_prototype_key
                    ][config_dictionary.kinematic_deviations_key][
                        config_dictionary.concentrator_tilt_e
                    ][()],
                    dtype=torch.float,
                )
            if (
                config_dictionary.concentrator_tilt_n
                in scenario_file[config_dictionary.prototype_key][
                    config_dictionary.kinematic_prototype_key
                ][config_dictionary.kinematic_deviations_key].keys()
            ):
                concentrator_tilt_n = torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.kinematic_prototype_key
                    ][config_dictionary.kinematic_deviations_key][
                        config_dictionary.concentrator_tilt_n
                    ][()],
                    dtype=torch.float,
                )
            if (
                config_dictionary.concentrator_tilt_u
                in scenario_file[config_dictionary.prototype_key][
                    config_dictionary.kinematic_prototype_key
                ][config_dictionary.kinematic_deviations_key].keys()
            ):
                concentrator_tilt_u = torch.tensor(
                    scenario_file[config_dictionary.prototype_key][
                        config_dictionary.kinematic_prototype_key
                    ][config_dictionary.kinematic_deviations_key][
                        config_dictionary.concentrator_tilt_u
                    ][()],
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
        kinematic_prototype = KinematicConfig(
            kinematic_type=str(
                scenario_file[config_dictionary.prototype_key][
                    config_dictionary.kinematic_prototype_key
                ][config_dictionary.kinematic_type][()].decode("utf-8")
            ),
            kinematic_initial_orientation_offsets=kinematic_offsets,
            kinematic_deviations=kinematic_deviations,
        )

        # Create actuator prototype
        actuator_list = []
        for ac in scenario_file[config_dictionary.prototype_key][
            config_dictionary.actuator_prototype_key
        ].keys():
            increment = torch.tensor(0.0)
            initial_stroke_length = torch.tensor(0.0)
            offset = torch.tensor(0.0)
            radius = torch.tensor(0.0)
            phi_0 = torch.tensor(0.0)
            if (
                config_dictionary.actuator_parameters_key
                in scenario_file[config_dictionary.prototype_key][
                    config_dictionary.actuator_prototype_key
                ][ac].keys()
            ):
                if (
                    config_dictionary.actuator_increment
                    in scenario_file[config_dictionary.prototype_key][
                        config_dictionary.actuator_prototype_key
                    ][ac][config_dictionary.actuator_parameters_key].keys()
                ):
                    increment = torch.tensor(
                        scenario_file[config_dictionary.prototype_key][
                            config_dictionary.actuator_prototype_key
                        ][ac][config_dictionary.actuator_parameters_key][
                            config_dictionary.actuator_increment
                        ][()],
                        dtype=torch.float,
                    )
                if (
                    config_dictionary.actuator_initial_stroke_length
                    in scenario_file[config_dictionary.prototype_key][
                        config_dictionary.actuator_prototype_key
                    ][ac][config_dictionary.actuator_parameters_key].keys()
                ):
                    initial_stroke_length = torch.tensor(
                        scenario_file[config_dictionary.prototype_key][
                            config_dictionary.actuator_prototype_key
                        ][ac][config_dictionary.actuator_parameters_key][
                            config_dictionary.actuator_initial_stroke_length
                        ][()],
                        dtype=torch.float,
                    )
                if (
                    config_dictionary.actuator_offset
                    in scenario_file[config_dictionary.prototype_key][
                        config_dictionary.actuator_prototype_key
                    ][ac][config_dictionary.actuator_parameters_key].keys()
                ):
                    offset = torch.tensor(
                        scenario_file[config_dictionary.prototype_key][
                            config_dictionary.actuator_prototype_key
                        ][ac][config_dictionary.actuator_parameters_key][
                            config_dictionary.actuator_offset
                        ][()],
                        dtype=torch.float,
                    )
                if (
                    config_dictionary.actuator_radius
                    in scenario_file[config_dictionary.prototype_key][
                        config_dictionary.actuator_prototype_key
                    ][ac][config_dictionary.actuator_parameters_key].keys()
                ):
                    radius = torch.tensor(
                        scenario_file[config_dictionary.prototype_key][
                            config_dictionary.actuator_prototype_key
                        ][ac][config_dictionary.actuator_parameters_key][
                            config_dictionary.actuator_radius
                        ][()],
                        dtype=torch.float,
                    )
                if (
                    config_dictionary.actuator_phi_0
                    in scenario_file[config_dictionary.prototype_key][
                        config_dictionary.actuator_prototype_key
                    ][ac][config_dictionary.actuator_parameters_key].keys()
                ):
                    phi_0 = torch.tensor(
                        scenario_file[config_dictionary.prototype_key][
                            config_dictionary.actuator_prototype_key
                        ][ac][config_dictionary.actuator_parameters_key][
                            config_dictionary.actuator_phi_0
                        ][()],
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
        )

        return cls(
            receivers=receivers,
            light_sources=light_sources,
            heliostat_field=heliostat_field,
        )
