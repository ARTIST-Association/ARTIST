from logging import log
import logging
from typing import Optional, Union
from typing_extensions import Self
from artist.util import config_dictionary
import h5py
import torch

log = logging.getLogger(__name__)
"""A logger for the tower area."""

class TowerArea(torch.nn.Module):
    """
    Implement a tower area.

    Attributes
    ----------
    area_type : str
        The type of the tower area, e.g., planar.
    center : torch.Tensor
        The center of the tower area.
    normal_vector : torch.Tensor
        The normal to the plane of the tower area.
    plane_e : float
        The size of the tower area in east direction.
    plane_u : torch.Tensor
        The size of the tower area in up direction.
    curvature_e : float
        The curvature of the tower area, in the east direction.
    curvature_u : float
        The curvature of the tower area, in the up direction.

    Methods
    -------
    from_hdf5()
        Class method that initializes a tower area from an HDF5 file.
    """

    def __init__(
        self,
        name: str,
        area_type: str,
        center: torch.Tensor,
        normal_vector: torch.Tensor,
        plane_e: float,
        plane_u: float,
        curvature_e: Optional[float] = None,
        curvature_u: Optional[float] = None,
    ) -> None:
        """
        Initialize the tower area.

        Tower areas are positioned on the solar tower and are either calibration targets or receivers that 
        absorbs the concentrated sunlight. The calibration target areas are used in the alignment optimization to
        find optimal kinematic parameters. The receiver areas are used for the actual power plant operation. During
        usual operation, the heliostats are aimed at a receiver area. This heats up the receiver. Behind the receiver,
        the clean energy extraction processes begin. Different kinds of receivers exist, and they are specified in the 
        tower area type, in this case the receiver type. Further parameters of a tower area are the center coordinates,
        the normal vector as well as the plane width and height. Optionally, the tower area can be provided with 
        curvature parameters, indicating the curvature of the tower area.

        Parameters
        ----------
        name : str
            The name of the tower area.
        area_type : str
            The type of the tower area, e.g., planar.
        center : torch.Tensor
            The center of the tower area.
        normal_vector : torch.Tensor
            The normal to the plane of the tower area.
        plane_e : float
            The size of the tower area in east direction.
        plane_u : torch.Tensor
            The size of the tower area in up direction.
        curvature_e : float, optional
            The curvature of the tower area, in the east direction.
        curvature_u : float, optional
            The curvature of the tower area, in the up direction.
        """
        super().__init__()
        self.name = name
        self.area_type = area_type
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
        area_name: Optional[str] = None,
        device: Union[torch.device, str] = "cuda",
    ) -> Self:
        """
        Class method that initializes a tower area from an HDF5 file.

        Parameters
        ----------
        config_file : h5py.File
            The HDF5 file containing the information about the tower area.
        area_name : str, optional
            The name of the tower area - used for logging
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        tower area
            A tower area initialized from an HDF5 file.
        """
        if area_name:
            log.info(f"Loading {area_name} from an HDF5 file.")
        device = torch.device(device)
        area_type = config_file[config_dictionary.tower_area_type][()].decode("utf-8")
        center = torch.tensor(
            config_file[config_dictionary.tower_area_center][()],
            dtype=torch.float,
            device=device,
        )
        normal_vector = torch.tensor(
            config_file[config_dictionary.tower_area_normal_vector][()],
            dtype=torch.float,
            device=device,
        )
        plane_e = float(config_file[config_dictionary.tower_area_plane_e][()])
        plane_u = float(config_file[config_dictionary.tower_area_plane_u][()])

        curvature_e = None
        curvature_u = None

        if config_dictionary.tower_area_curvature_e in config_file.keys():
            curvature_e = float(config_file[config_dictionary.tower_area_curvature_e][()])
        else:
            log.warning("No curvature in the east direction set for the tower area!")
        if config_dictionary.tower_area_curvature_u in config_file.keys():
            curvature_u = float(config_file[config_dictionary.tower_area_curvature_u][()])
        else:
            log.warning("No curvature in the up direction set for the tower area!")

        return cls(
            name=area_name,
            area_type=area_type,
            center=center,
            normal_vector=normal_vector,
            plane_e=plane_e,
            plane_u=plane_u,
            curvature_e=curvature_e,
            curvature_u=curvature_u,
        )
