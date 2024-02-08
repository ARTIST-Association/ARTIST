"""
This file contains the functionality to create a heliostat surface from a loaded pointcloud.
"""
import struct
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from yacs.config import CfgNode

from artist.physics_objects.heliostats.surface.facets.facets import AFacetModule
from artist.physics_objects.heliostats.surface.nurbs import bpro_loader


HeliostatParams = Tuple[
    torch.Tensor,  # surface position on field
    torch.Tensor,  # facet positions
    torch.Tensor,  # facet spans N
    torch.Tensor,  # facet spans E
    torch.Tensor,  # discrete points
    torch.Tensor,  # ideal discrete points
    torch.Tensor,  # normals
    torch.Tensor,  # ideal normals
    float,  # height
    float,  # width
    Optional[int],  # rows
    Optional[int],  # cols
    Optional[Dict[str, Any]],  # params
]


def real_surface(
    real_configs: CfgNode,
    device: torch.device,
) -> HeliostatParams:
    """
    Compute a surface loaded from deflectometric data.

    Parameters
    ----------
    real_config : CfgNode
        The config file containing information about the real surface.
    device : torch.device
        Specifies the device type responsible to load tensors into memory.

    Returns
    -------
    HeliostatParams
        Tuple of all heliostat parameters.

    """
    cfg = real_configs
    dtype = torch.get_default_dtype()

    concentratorHeader_struct = struct.Struct(cfg.CONCENTRATORHEADER_STRUCT_FMT)
    facetHeader_struct = struct.Struct(cfg.FACETHEADER_STRUCT_FMT)
    ray_struct = struct.Struct(cfg.RAY_STRUCT_FMT)

    (
        surface_position,
        facet_positions,
        facet_spans_n,
        facet_spans_e,
        ideal_positions,
        directions,
        ideal_normal_vecs,
        width,
        height,
    ) = bpro_loader.load_bpro(
        cfg.FILENAME,
        concentratorHeader_struct,
        facetHeader_struct,
        ray_struct,
        cfg.VERBOSE,
    )

    surface_position: torch.Tensor = (
        torch.tensor(surface_position, dtype=dtype, device=device)
        if cfg.POSITION_ON_FIELD is None
        else get_position(cfg, dtype, device)
    )

    heliostat_normal_vecs = []
    heliostat_ideal_vecs = []
    heliostat = []
    heliostat_ideal = []

    step_size = sum(map(len, directions)) // cfg.TAKE_N_VECTORS
    for f in range(len(directions)):
        heliostat_normal_vecs.append(
            torch.tensor(
                directions[f][::step_size],
                dtype=dtype,
                device=device,
            )
        )
        heliostat_ideal_vecs.append(
            torch.tensor(
                ideal_normal_vecs[f][::step_size],
                dtype=dtype,
                device=device,
            )
        )
        heliostat.append(
            torch.tensor(
                ideal_positions[f][::step_size],
                dtype=dtype,
                device=device,
            )
        )

        heliostat_ideal.append(
            torch.tensor(
                ideal_positions[f][::step_size],
                dtype=dtype,
                device=device,
            )
        )

    heliostat_normal_vecs: torch.Tensor = torch.cat(heliostat_normal_vecs, dim=0)
    heliostat_ideal_vecs: torch.Tensor = torch.cat(heliostat_ideal_vecs, dim=0)
    heliostat: torch.Tensor = torch.cat(heliostat, dim=0)

    heliostat_ideal: torch.Tensor = torch.cat(heliostat_ideal, dim=0)
    if cfg.VERBOSE:
        print("Done")

    return (
        surface_position,
        torch.tensor(facet_positions, dtype=dtype, device=device),
        torch.tensor(facet_spans_n, dtype=dtype, device=device),
        torch.tensor(facet_spans_e, dtype=dtype, device=device),
        heliostat,
        heliostat_ideal,
        heliostat_normal_vecs,
        heliostat_ideal_vecs,
        height,
        width,
        None,
        None,
        None,
    )


def get_position(
    cfg: CfgNode,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Retrieve the position of the heliostat in the field as tensor.

    Parameters
    ----------
    cfg : CfgNode
        The config file containing the information about the heliostat.
    dtype : torch.dtype
        The type and size of the tensor.
    device : torch.device
        Specifies the device type responsible to load tensors into memory.

    Returns
    -------
    torch.tensor
        The position of the heliostat in the field.
    """
    position_on_field: List[float] = cfg.POSITION_ON_FIELD
    return torch.tensor(position_on_field, dtype=dtype, device=device)


def _indices_between(
    points: torch.Tensor,
    from_: torch.Tensor,
    to: torch.Tensor,
) -> torch.Tensor:
    indices = ((from_ <= points) & (points < to)).all(dim=-1)
    return indices


def facet_point_indices(
    points: torch.Tensor,
    position: torch.Tensor,
    span_n: torch.Tensor,
    span_e: torch.Tensor,
) -> torch.Tensor:
    from_xyz = position + span_e - span_n
    to_xyz = position - span_e + span_n
    # We ignore the z-axis here.
    return _indices_between(
        points[:, :-1],
        from_xyz[:-1],
        to_xyz[:-1],
    )


class PointCloudFacetModule(AFacetModule):
    """
    Implementation of the heliostat surface loaded from a pointcloud.

    Attributes
    ----------
    cfg : CfgNode
        The config file containing information about the surface.
    device : torch.device
        Specifies the device type responsible to load tensors into memory.

    Methods
    -------
    load()
        Load a surface from deflectometry data.
    select_surface_builder()
        Select which kind of surface is to be loaded.
    _get_aim_point()
        Retrieve the aim point.
    _get_disturbance_angles()
        Retrieve the disturbance angles.
    set_up_facets()
        Split up the surface points and surface normals into their respective facets.
    discrete_points_and_normals()
        Return the surface points and surface normals.
    facetted_discrete_points_and_normals()
        Return the facetted surface points and facetted surface normals.
    make_facets_list()
        Create a list of facets.

    See also
    --------
    :class:AFacetModule : Reference to the parent class.
    """

    def __init__(
        self,
        config: CfgNode,
        aimpoint: torch.Tensor,
        sun_direction: torch.Tensor,
        device: torch.device,
    ) -> None:
        """
        Initialize the surface from a pointcloud.

        Parameters
        ----------
        config : CfgNode
            The config file containing information about the loaded surface.
        aimpoint : torch.Tensor
            The aimpoint of the heliostat.
        sun_direction : torch.Tensor
            The direction of the light.
        device : torch.Tensor
            Specifies the device type responsible to load tensors into memory.
        """
        super().__init__()

        self.cfg = config
        self.device = device

        self.load(aimpoint, sun_direction)

    def load(
        self,
        aim_point: Optional[torch.Tensor],
        sun_direction: Optional[torch.Tensor],
    ) -> None:
        """
        Load a surface from deflectometry data.

        Parameters
        ----------
        aim_point : Optional[torch.Tensor]
            The aimpoint.
        sun_direction : Optional[torch.Tensor]
            The sun vector.
        """
        builder_fn, h_cfg = self.select_surface_builder(self.cfg)

        self.aim_point = self._get_aim_point(
            h_cfg,
            aim_point,
        )
        # Radians
        self.disturbance_angles = self._get_disturbance_angles(h_cfg)

        (
            surface_position,
            facet_positions,
            facet_spans_n,
            facet_spans_e,
            surface_points,
            surface_ideal,
            surface_normals,
            surface_ideal_vecs,
            height,
            width,
            rows,
            cols,
            params,
        ) = builder_fn(h_cfg, self.device)

        self.position_on_field = surface_position
        self.set_up_facets(
            surface_points,
            surface_normals,
        )
        self.params = params
        self.height = height
        self.width = width
        self.rows = rows
        self.cols = cols

        self.surface_points: torch.Tensor
        self.surface_normals: torch.Tensor

    def select_surface_builder(
        self, cfg: CfgNode
    ) -> Tuple[Callable[[CfgNode, torch.device], HeliostatParams], CfgNode,]:
        """
        Select which kind of surface is to be loaded.

        At the moment only real surfaces can be loaded.

        Parameters
        ----------
        cfg : CfgNode
            Contains the information about the shape/kind of surface to be loaded.

        Returns
        -------
        Tuple[Callable[[CfgNode, torch.device], HeliostatParams], CfgNode,]
            The loaded surface, the heliostat parameters, and deflectometry data.
        """
        shape = cfg.SHAPE.lower()
        if shape == "real":
            return real_surface, cfg.DEFLECT_DATA
        raise ValueError("unknown surface shape")

    def _get_aim_point(
        self,
        cfg: CfgNode,
        aim_point: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Retrieve the aim point.

        Parameters
        ----------
        cfg : CfgNode
            The config file containing indormation about the aim point.
        aim_point : Optional[torch.Tensor]
            Optionally pass an aim point.

        Returns
        -------
        torch.Tensor
            The aim point.
        
        Raises
        ------
        ValueError
            When no aim point is provided via optional argument or config file.        
        """
        cfg_aim_point: Optional[List[float]] = cfg.AIM_POINT
        if cfg_aim_point is not None:
            aim_point = torch.tensor(
                cfg_aim_point,
                dtype=torch.get_default_dtype(),
                device=self.device,
            )
        elif aim_point is not None:
            aim_point = aim_point
        else:
            raise ValueError("no aim point was supplied")
        return aim_point

    def _get_disturbance_angles(self, h_cfg: CfgNode) -> List[torch.Tensor]:
        """
        Retrieve the disturbance angles.

        Parameters
        ----------
        h_cfg : CfgNode
            The config file containing information about the disturbance angles.

        Returns
        -------
        List[torch.Tensor]
            The disturbance angles.
        """
        angles: List[float] = h_cfg.DISTURBANCE_ROT_ANGLES
        return [
            torch.deg2rad(
                torch.tensor(
                    angle,
                    dtype=self.aim_point.dtype,
                    device=self.device,
                )
            )
            for angle in angles
        ]

    def set_up_facets(
        self,
        discrete_points: torch.Tensor,
        normals: torch.Tensor,
    ) -> None:
        """
        Split up the surface points and surface normals into their respective facets.

        Parameters
        ----------
        discrete_points : torch.Tensor
            The surface points.
        normals : torch.Tensor
            The surface normals.
        """
        self.facetted_discrete_points = torch.tensor_split(discrete_points, 4)
        self.facetted_normals = torch.tensor_split(normals, 4)

    def discrete_points_and_normals(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the surface points and surface normals.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The surface points and surface normals.
        """
        return self.surface_points, self.surface_normals

    def facetted_discrete_points_and_normals(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the facetted surface points and facetted surface normals.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The facetted surface points and facetted surface normals.
        """
        return self.facetted_discrete_points, self.facetted_normals

    def make_facets_list(self) -> list[list[torch.Tensor], list[torch.Tensor]]:
        """
        Create a list of facets.

        Returns
        List[List[torch.Tensor], List[torch.Tensor]]
            The list of all facets with their points and normals.
        """
        facets = []
        (
            facetted_discrete_points,
            facetted_normals,
        ) = self.facetted_discrete_points_and_normals()

        for points, normals in zip(facetted_discrete_points, facetted_normals):
            facets.append([points, normals])
        return facets
