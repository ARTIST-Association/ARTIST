from typing import Tuple

import torch

from artist.io.datapoint import HeliostatDataPoint
from artist.physics_objects.heliostats.alignment.alignment import AlignmentModule
from artist.physics_objects.heliostats.surface.concentrator import ConcentratorModule
from artist.physics_objects.module import AModule


class HeliostatModule(AModule):
    """
    Implementation of the Heliostat as a module.

    Attributes
    ----------
    concentrator : ConcentratorModule
        The surface of the heliostat.
    alignment : AlignmentModule
        The alignment module of the heliostat.

    Methods
    -------
    get_aligned_surface()
        Compute the aligned surface points and aligned surface normals of the heliostat.

    See Also
    --------
    :class:AModule : Reference to the parent class.
    """

    def __init__(
        self, concentrator: ConcentratorModule, alignment: AlignmentModule
    ) -> None:
        """
        Initialize the heliostat.

        Parameters
        ----------
        concentrator : ConcentratorModule
            The surface of the heliostat.
        alignment : AlignmentModule
            The alignment module of the heliostat.
        """
        super().__init__()
        self.concentrator = concentrator
        self.alignment = alignment

    def get_aligned_surface(
        self, datapoint: HeliostatDataPoint
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the aligned surface points and aligned surface normals of the heliostat.

        Parameters
        ----------
        datapoint : HeliostatDataPoint
            Datapoint containing information about the heliostat and the environment.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The aligned surface points and aligned surface normals.
        """
        surface_points, surface_normals = self.concentrator.get_surface()
        aligned_surface_points, aligned_surface_normals = self.alignment.align_surface(
            datapoint, surface_points, surface_normals
        )
        return aligned_surface_points, aligned_surface_normals
