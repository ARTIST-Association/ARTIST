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

        print("Hi")

        self.aim_point = torch.tensor(
            config_file["heliostats"]["heliostats_list"][heliostat_name]["aim_point"][
                ()
            ]
        )
        self.position = torch.tensor(
            config_file["heliostats"]["heliostats_list"][heliostat_name]["position"][()]
        )
        self.incident_ray_direction = incident_ray_direction
        self.concentrator = ConcentratorModule(
            heliostat_name=heliostat_name, config_file=config_file
        )
        # self.alignment = AlignmentModule(heliostat_config)

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
        surface_points, surface_normals = self.concentrator.get_surface()
        aligned_surface_points, aligned_surface_normals = self.alignment.align_surface(
            self.aim_point, self.incident_ray_direction, surface_points, surface_normals
        )
        return aligned_surface_points, aligned_surface_normals
