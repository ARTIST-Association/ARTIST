"""
This file contains the functionality to create a heliostat surface from a loaded pointcloud.
"""

import struct
from typing import Any, Callable, Dict, List, Optional, Tuple

import h5py
import torch
from yacs.config import CfgNode

from artist.physics_objects.heliostats.concentrator.facets.facets import AFacetModule
from artist.util import config_dictionary 

class PointCloudFacetModule(AFacetModule):
    """
    Implementation of the heliostat surface loaded from a point cloud.

    Attributes
    ----------
    surface_points : torch.Tensor
        The surface points vectors.
    surface_normals : torch.Tensor
        The surface normal vectors.

    See also
    --------
    :class:AFacetModule : Reference to the parent class.
    """

    def __init__(
        self,
        config_file: h5py.File,
    ) -> None:
        """
        Initialize the surface from a point cloud.

        Parameters
        ----------
        heliostat_name : str
            The name of the heliostat being initialized.
        config_file : h5py.File
            An open hdf5 file containing the scenario configuration.
        """
        super().__init__()
        if not config_file[config_dictionary.heliostat_prefix][config_dictionary.general_surface_normals]:
            raise NotImplementedError(
                "Currently ARTIST is only implemented for general surface normals."
            )
        if not config_file[config_dictionary.heliostat_prefix][config_dictionary.general_surface_points]:
            raise NotImplementedError(
                "Currently ARTIST is only implemented for general surface points."
            )
        else:
            self.surface_points = torch.tensor(
                config_file[config_dictionary.heliostat_prefix][config_dictionary.general_surface_points][()],
                dtype=torch.float,
            )
            self.surface_normals = torch.tensor(
                config_file[config_dictionary.heliostat_prefix][config_dictionary.general_surface_normals][()],
                dtype=torch.float,
            )
