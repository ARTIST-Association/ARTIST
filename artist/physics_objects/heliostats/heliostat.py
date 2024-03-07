from typing import Tuple

import h5py
from yacs.config import CfgNode

import torch

from artist.io.datapoint import HeliostatDataPoint
from artist.physics_objects.heliostats.concentrator.concentrator import (
    ConcentratorModule,
)
from artist.physics_objects.heliostats.alignment.alignment import AlignmentModule
from artist.physics_objects.module import AModule
from artist.util import config_dictionary 

class HeliostatModule(AModule):
    """
    Implementation of the Heliostat as a module.

    Attributes
    ----------
    position : torch.Tensor
        The position of the heliostat in the field.
    aim_point : torch.Tensor
        The aim point on the receiver.
    incident_ray_direction : torch.Tensor
        The direction of the rays.
    concentrator : ConcentratorModule
        The surface of the heliostat.
    alignment : AlignmentModule
        The alignment module of the heliostat.

    Methods
    -------
    get_aligned_surface()
        Compute the aligned surface points and aligned surface normals of the heliostat.

    See also
    --------
    :class:AModule : Reference to the parent class.
    """

    # def __init__(
    #     self,
    #     position: torch.Tensor,
    #     incident_ray_direction: torch.Tensor,
    #     heliostat_name: str,
    # ) -> None:
    #     """
    #     Initialize the heliostat.

    #     Parameters
    #     ----------
    #     position : torch.Tensor
    #         The position of the Heliostat.
    #     incident_ray_direction : torch.Tensor
    #         The direction of the incident ray as seen from the heliostat.
    #     heliostat_name : str
    #         The name of the heliostat being initialized.
    #     """
    #     super().__init__()

    #     self.position = position
    #     self.incident_ray_direction = incident_ray_direction
    #     self.concentrator = ConcentratorModule(
    #         heliostat_name=heliostat_name, config_file=config_file
    #     )
    #     self.alignment = AlignmentModule(
    #         heliostat_name=heliostat_name, config_file=config_file
    #     )
    
    # @classmethod
    # def instantiate_from_file(cls, config_file: h5py.File, incident_ray_direction, heliostat_name):
    #     position = torch.tensor(
    #         config_file[config_dictionary.heliostat_prefix][config_dictionary.heliostats_list][heliostat_name][config_dictionary.heliostat_position][
    #             ()
    #         ],
    #         dtype=torch.float,
    #     )
    #     return cls(position, incident_ray_direction, heliostat_name)

    def __init__(
        self,
        heliostat_name: str,
        incident_ray_direction: torch.Tensor,
        config_file: h5py.File = None,
    ) -> None:
        """
        Initialize the heliostat.

        Parameters
        ----------
        heliostat_name : str
            The name of the heliostat being initialized.
        incident_ray_direction : torch.Tensor
            The direction of the incident ray as seen from the heliostat.
        config_file : h5py.File
            An open hdf5 file containing the scenario configuration.
        """
        super().__init__()
        self.position = torch.tensor(
            config_file[config_dictionary.heliostat_prefix][config_dictionary.heliostats_list][heliostat_name][config_dictionary.heliostat_position][
                ()
            ],
            dtype=torch.float,
        )
        self.incident_ray_direction = incident_ray_direction
        self.concentrator = ConcentratorModule(
            heliostat_name=heliostat_name, config_file=config_file
        )
        self.alignment = AlignmentModule(
            heliostat_name=heliostat_name, config_file=config_file
        )

    def get_aligned_surface(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the aligned surface points and aligned surface normals of the heliostat.

        Parameters
        ----------
        aim_point : torch.Tensor
            The desired aim point.
        incident_ray_direction : torch.Tensor
            The direction of the rays.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The aligned surface points and aligned surface normals.
        """
        surface_points, surface_normals = (
            self.concentrator.facets.surface_points,
            self.concentrator.facets.surface_normals,
        )
        aligned_surface_points, aligned_surface_normals = self.alignment.align_surface(
            self.incident_ray_direction, surface_points, surface_normals
        )
        return aligned_surface_points, aligned_surface_normals
