import logging
from typing import Union

from artist.scenario import Scenario
import h5py
import torch
from typing_extensions import Self

from artist.field.heliostat_field import HeliostatField
from artist.field.tower_target_area_array import TargetAreaArray
from artist.scene.light_source_array import LightSourceArray
from artist.util import config_dictionary
from artist.util.configuration_classes import (
    ActuatorConfig,
    ActuatorListConfig,
    ActuatorParameters,
    FacetConfig,
    KinematicDeviations,
    KinematicLoadConfig,
    SurfaceConfig,
)

log = logging.getLogger(__name__)
"""A logger for the scenario."""


class NewScenario:
    """
    Define a scenario loaded by ARTIST.

    Attributes
    ----------
    power_plant_position : torch.Tensor
        The position of the power plant as latitude, longitude, altitude.
    target_areas : TargetAreaArray
        A list of tower target areas included in the scenario.
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
        scenario: Scenario,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """
        Initialize the scenario.

        A scenario defines the physical objects and scene to be used by ``ARTIST``. Therefore, a scenario contains at
        least one target area that is a receiver, at least one light source and at least one heliostat in a heliostat field.
        ``ARTIST`` also supports scenarios that contain multiple target areas, multiple light sources, and multiple heliostats.

        Parameters
        ----------
        power_plant_position : torch.Tensor,
            The position of the power plant as latitude, longitude, altitude.
        target_areas : TargetAreaArray
            A list of tower target areas included in the scenario.
        light_sources : LightSourceArray
            A list of light sources included in the scenario.
        heliostat_field : HeliostatField
            A field of heliostats included in the scenario.
        """
        self.power_plant_position = scenario.power_plant_position
        self.target_areas = scenario.target_areas
        self.light_sources = scenario.light_sources
        
        number_of_heliostats = 2200 #len(scenario.heliostats.heliostat_list)
        number_of_surface_points_per_heliostat = scenario.heliostats.heliostat_list[0].surface_points.shape[0] * scenario.heliostats.heliostat_list[0].surface_points.shape[1]
        dimension_of_single_point = scenario.heliostats.heliostat_list[0].surface_points.shape[2]

        self.all_preferred_reflection_directions = torch.zeros((number_of_heliostats, dimension_of_single_point))
        self.all_current_aligned_surface_points = torch.zeros((number_of_heliostats, number_of_surface_points_per_heliostat, dimension_of_single_point), device=device)
        self.all_current_aligned_surface_normals = torch.zeros((number_of_heliostats, number_of_surface_points_per_heliostat, dimension_of_single_point), device=device)
        self.is_aligned = True

        for i in range(number_of_heliostats):
            self.all_current_aligned_surface_points[i] = scenario.heliostats.heliostat_list[i%4].current_aligned_surface_points.reshape(-1, dimension_of_single_point)
            self.all_current_aligned_surface_normals[i] = scenario.heliostats.heliostat_list[i%4].current_aligned_surface_normals.reshape(-1, dimension_of_single_point)

