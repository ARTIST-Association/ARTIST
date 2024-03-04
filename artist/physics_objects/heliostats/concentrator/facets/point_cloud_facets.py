"""
This file contains the functionality to create a heliostat surface from a loaded pointcloud.
"""

import struct
from typing import Any, Callable, Dict, List, Optional, Tuple

import h5py
import torch
from yacs.config import CfgNode

from artist.physics_objects.heliostats.concentrator.facets.facets import AFacetModule
from artist.physics_objects.heliostats.concentrator import bpro_loader

HeliostatParams = Tuple[
    torch.Tensor,  # surface position on field
    torch.Tensor,  # facet positions
    torch.Tensor,  # facet spans N
    torch.Tensor,  # facet spans E
    torch.Tensor,  # discrete points
    torch.Tensor,  # normals
    float,  # height
    float,  # width
]


def real_surface(
    config: CfgNode,
) -> HeliostatParams:
    """
    Compute a surface loaded from deflectometric data.

    Parameters
    ----------
    config : CfgNode
        The config file containing information about the real surface.

    Returns
    -------
    HeliostatParams
        Tuple of all heliostat parameters.

    """
    concentratorHeader_struct = struct.Struct(config.CONCENTRATORHEADER_STRUCT_FMT)
    facetHeader_struct = struct.Struct(config.FACETHEADER_STRUCT_FMT)
    ray_struct = struct.Struct(config.RAY_STRUCT_FMT)

    (
        surface_position,
        facet_positions,
        facet_spans_n,
        facet_spans_e,
        ideal_surface_points,
        normal_vectors,
        _,
        width,
        height,
    ) = bpro_loader.load_bpro(
        config.FILENAME,
        concentratorHeader_struct,
        facetHeader_struct,
        ray_struct,
        config.VERBOSE,
    )

    surface_position = set_default_surface_position(config, surface_position)
    points_on_facet = normal_vectors.shape[2]
    if points_on_facet < config.TAKE_N_VECTORS:
        raise ValueError(
            f"TAKE_N_VECTORS was {config.TAKE_N_VECTORS} cannot be larger than number of points on facet {points_on_facet}"
        )
    step_size = points_on_facet // config.TAKE_N_VECTORS

    # thinning: keep only every step-size-th surface point and normal
    # shape goes from (#facets, 3, #points) -> (#facets, 3, #points/step_size)
    ideal_surface_points = ideal_surface_points[:, :, ::step_size]
    normal_vectors = normal_vectors[:, :, ::step_size]

    if config.VERBOSE:
        print("Done")

    return (
        surface_position,
        facet_positions,
        facet_spans_n,
        facet_spans_e,
        ideal_surface_points,
        normal_vectors,
        height,
        width,
    )


def set_default_surface_position(
    config: CfgNode, position: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Retrieve the position of the heliostat in the field as tensor either from cfg or bpro.

    Parameters
    ----------
    config : CfgNode
        The config file containing the information about the heliostat.
    position : torch.Tensor
        The position of the suspension point of the surface.

    Returns
    -------
    torch.tensor
        The position of the heliostat in the field.
    """
    if config.POSITION_ON_FIELD is None:
        return torch.tensor(config.POSITION_ON_FIELD).reshape(-1, 1)
    return position


class PointCloudFacetModule(AFacetModule):
    """
    Implementation of the heliostat surface loaded from a point cloud.

    Attributes
    ----------
    config : CfgNode
        The config file containing information about the surface.
    surface_position : torch.Tensor
        The position of the surface suspension point.
    ideal_surface_points : torch.Tensor
        The ideal surface points.
    surface_normals : torch.Tensor
        The surface normal vectors.
    heliostat_height : float
        The height of the surface.
    heliostat_width : float
        The width of the surface.

    Methods
    -------
    load()
        Load a surface from deflectometry data.
    select_surface_builder()
        Select which kind of surface is to be loaded.

    See also
    --------
    :class:AFacetModule : Reference to the parent class.
    """

    def __init__(
        self,
        heliostat_name: str,
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

        print("THIS IS A FAR AS WE ARE")
        print("KEEP WORKING FROM HERE")
        # self.config = config
        # self.surface_position = None
        # self.ideal_surface_points = None
        # self.surface_normals = None
        # self.heliostat_height = 0.0
        # self.heliostat_width = 0.0
        #
        # self.load()

    def load(self) -> None:
        """
        Load a surface from deflectometry data.
        """
        builder_function, heliostat_config = self.get_surface_builder(self.config)

        (
            self.surface_position,
            _,
            _,
            _,
            self.ideal_surface_points,
            self.surface_normals,
            self.heliostat_height,
            self.heliostat_width,
        ) = builder_function(heliostat_config)

    def get_surface_builder(self, config: CfgNode) -> Tuple[
        Callable[[CfgNode, torch.device], HeliostatParams],
        CfgNode,
    ]:
        """
        Select which kind of surface is to be loaded.

        At the moment only real surfaces can be loaded.

        Parameters
        ----------
        config : CfgNode
            Contains the information about the kind of surface to be loaded.

        Returns
        -------
        Tuple[Callable[[CfgNode, torch.device], HeliostatParams], CfgNode,]
            The loaded surface, the heliostat parameters, and deflectometry data.
        """
        return real_surface, config.DEFLECT_DATA
