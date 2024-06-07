import logging
from typing import Optional

import h5py
import torch.nn
from typing_extensions import Self

from artist.util import config_dictionary

log = logging.getLogger(__name__)
"""A logger for the receiver."""


class Receiver(torch.nn.Module):
    """
    Implement a receiver.

    Attributes
    ----------
    receiver_type : str
        The type of the receiver, e.g., planar.
    position_center : torch.Tensor
        The center of the receiver.
    normal_vector : torch.Tensor
        The normal to the plane of the receiver.
    plane_e : float
        The east plane of the receiver.
    plane_u : torch.Tensor
        The up plane of the receiver.
    resolution_e : int
        The horizontal resolution in the east direction of the receiver.
    resolution_u : int
        The vertical resolution in the up direction of the receiver.

    Methods
    -------
    from_hdf5()
        Class method that initializes a receiver from an HDF5 file.
    """

    def __init__(
        self,
        receiver_type: str,
        position_center: torch.Tensor,
        normal_vector: torch.Tensor,
        plane_e: float,
        plane_u: float,
        resolution_e: int,
        resolution_u: int,
        curvature_e: Optional[float] = None,
        curvature_u: Optional[float] = None,
    ) -> None:
        """
        Initialize the receiver.

        The receiver is positioned on the solar tower and absorbs the concentrated sunlight. This heats
        up the receiver. Behind the receiver, the clean energy extraction processes begin. Different kinds
        of receivers exist, and they are specified in the receiver type. Further parameters are the position
        center which indicates the center of the receiver, the normal vector that is a normal to the receiver
        plane, the plane width and height as well as the receiver resolution in both directions. Optionally,
        the receiver can be provided with curvature parameters, indicating the curvature of the receiver.

        Parameters
        ----------
        receiver_type : str
            The type of the receiver, e.g., planar.
        position_center : torch.Tensor
            The center of the receiver.
        normal_vector : torch.Tensor
            The normal to the plane of the receiver.
        plane_e : float
            The width of the receiver.
        plane_u : torch.Tensor
            The height of the receiver.
        resolution_e : int
            The horizontal resolution in the east direction of the receiver.
        resolution_u : int
            The vertical resolution in the up direction of the receiver.
        curvature_e : float, optional
            The curvature of the receiver, in the east direction.
        curvature_u : float, optional
            The curvature of the receiver, in the up direction.
        """
        super().__init__()
        self.receiver_type = receiver_type
        self.position_center = position_center
        self.normal_vector = normal_vector
        self.plane_e = plane_e
        self.plane_u = plane_u
        self.resolution_e = resolution_e
        self.resolution_u = resolution_u
        self.curvature_e = curvature_e
        self.curvature_u = curvature_u

    @classmethod
    def from_hdf5(
        cls, config_file: h5py.File, receiver_name: Optional[str] = None
    ) -> Self:
        """
        Class method that initializes a receiver from an HDF5 file.

        Parameters
        ----------
        config_file : h5py.File
            The HDF5 file containing the information about the receiver.
        receiver_name : str, optional
            The name of the receiver - used for logging

        Returns
        -------
        Receiver
            A receiver initialized from an HDF5 file.
        """
        if receiver_name:
            log.info(f"Loading {receiver_name} from an HDF5 file.")
        receiver_type = config_file[config_dictionary.receiver_type][()].decode("utf-8")
        position_center = torch.tensor(
            config_file[config_dictionary.receiver_position_center][()],
            dtype=torch.float,
        )
        normal_vector = torch.tensor(
            config_file[config_dictionary.receiver_normal_vector][()],
            dtype=torch.float,
        )
        plane_e = float(config_file[config_dictionary.receiver_plane_e][()])
        plane_u = float(config_file[config_dictionary.receiver_plane_u][()])
        resolution_e = int(config_file[config_dictionary.receiver_resolution_e][()])
        resolution_u = int(config_file[config_dictionary.receiver_resolution_u][()])

        curvature_e = None
        curvature_u = None

        if config_dictionary.receiver_curvature_e in config_file.keys():
            curvature_e = float(config_file[config_dictionary.receiver_curvature_e][()])
        else:
            log.warning("No curvature in the east direction set for the receiver!")
        if config_dictionary.receiver_curvature_u in config_file.keys():
            curvature_u = float(config_file[config_dictionary.receiver_curvature_u][()])
        else:
            log.warning("No curvature in the up direction set for the receiver!")

        return cls(
            receiver_type=receiver_type,
            position_center=position_center,
            normal_vector=normal_vector,
            plane_e=plane_e,
            plane_u=plane_u,
            resolution_e=resolution_e,
            resolution_u=resolution_u,
            curvature_e=curvature_e,
            curvature_u=curvature_u,
        )
