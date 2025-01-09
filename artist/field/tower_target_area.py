import logging
from typing import Optional, Union

import h5py
import torch.nn
from typing_extensions import Self

from artist.util import config_dictionary

log = logging.getLogger(__name__)
"""A logger for the tower target area."""

# TODO
# Right now both the calibration targets and the receiver are modeled as a plane.
# At some point the receiver might be modeled more precicely. If that happens
# we might have to create different classes for calibration targets and receivers,
# as parameters like the normal_vector or plane_e only make sense for planar areas not convex_cylinders.

class TargetArea(torch.nn.Module):
    """
    Implement a target area.

    Attributes
    ----------
    geometry : str
        The geometry of the target area, e.g., planar.
    center : torch.Tensor
        The center of the target area.
    normal_vector : torch.Tensor
        The normal to the plane of the target area.
    plane_e : float
        The east dimension of the target plane.
    plane_u : torch.Tensor
        The up dimension of the target plane.
    curvature_e : float
        The curvature of the target area, in the east direction.
    curvature_u : float
        The curvature of the target area, in the up direction.

    Methods
    -------
    from_hdf5()
        Class method that initializes a tower target area from an HDF5 file.
    forward()
        Specify the forward pass.
    """

    def __init__(
        self,
        name: str,
        geometry: str,
        center: torch.Tensor,
        normal_vector: torch.Tensor,
        plane_e: float,
        plane_u: float,
        curvature_e: Optional[float] = None,
        curvature_u: Optional[float] = None,
    ) -> None:
        """
        Initialize the tower target area.
        
        Target areas are positioned on the solar tower and are either calibration targets or receivers.
        The calibration target areas are used in the alignment optimization to find optimal kinematic parameters. 
        The receiver areas are used for the actual power plant operation. During the usual operation, the heliostats 
        are aimed at a receiver area. This heats up the receiver. Behind the receiver, the clean energy extraction 
        processes begin. Parameters of a target area include the center coordinate, the normal vector as well 
        as the plane width and height. Optionally, the target area can be provided with curvature parameters, 
        indicating the curvature of the target area.

        Parameters
        ----------
        name : str
            The name of the target area.
        geometry : str
            The geometry of the target area, e.g., planar.
        center : torch.Tensor
            The center of the target area.
        normal_vector : torch.Tensor
            The normal to the plane of the target area.
        plane_e : float
            The east dimension of the target plane.
        plane_u : torch.Tensor
            The up dimension of the target plane.
        curvature_e : float, optional
            The curvature of the target area, in the east direction.
        curvature_u : float, optional
            The curvature of the target area, in the up direction.
        """
        super().__init__()
        self.name = name
        self.geometry = geometry
        self.center = center
        self.normal_vector = normal_vector
        self.plane_e = plane_e
        self.plane_u = plane_u
        self.curvature_e = curvature_e
        self.curvature_u = curvature_u

    @classmethod
    def from_hdf5(
        cls,
        config_file: h5py.File,
        target_area_name: str,
        device: Union[torch.device, str] = "cuda",
    ) -> Self:
        """
        Class method that initializes a tower target area from an HDF5 file.

        Parameters
        ----------
        config_file : h5py.File
            The HDF5 file containing the information about the target area.
        target_area_name : str, optional
            The name of the target area - used for logging.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        TargetArea
            A tower target area initialized from an HDF5 file.
        """
        if target_area_name:
            log.info(f"Loading {target_area_name} from an HDF5 file.")
        device = torch.device(device)
        geometry = config_file[config_dictionary.target_area_geometry][()].decode("utf-8")
        center = torch.tensor(
            config_file[config_dictionary.target_area_position_center][()],
            dtype=torch.float,
            device=device,
        )
        normal_vector = torch.tensor(
            config_file[config_dictionary.target_area_normal_vector][()],
            dtype=torch.float,
            device=device,
        )
        plane_e = float(config_file[config_dictionary.target_area_plane_e][()])
        plane_u = float(config_file[config_dictionary.target_area_plane_u][()])

        curvature_e = None
        curvature_u = None

        if config_dictionary.target_area_curvature_e in config_file.keys():
            curvature_e = float(config_file[config_dictionary.target_area_curvature_e][()])
        else:
            log.warning("No curvature in the east direction set for the receiver!")
        if config_dictionary.target_area_curvature_u in config_file.keys():
            curvature_u = float(config_file[config_dictionary.target_area_curvature_u][()])
        else:
            log.warning("No curvature in the up direction set for the receiver!")

        return cls(
            name=target_area_name,
            geometry=geometry,
            center=center,
            normal_vector=normal_vector,
            plane_e=plane_e,
            plane_u=plane_u,
            curvature_e=curvature_e,
            curvature_u=curvature_u,
        )

    def forward(self) -> None:
        """
        Specify the forward pass.

        Raises
        ------
        NotImplementedError
            Whenever called.
        """
        raise NotImplementedError("Not Implemented!")
