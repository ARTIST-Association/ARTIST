import h5py
import torch.nn
from typing_extensions import Self

from artist.util import config_dictionary


class Receiver(torch.nn.Module):
    """
    This class implements a receiver.

    Attributes
    ----------
    center : torch.Tensor
        The center of the receiver.
    plane_normal : torch.Tensor
        The normal to the plane of the receiver.
    plane_x : float
        The x plane of the receiver.
    plane_y : torch.Tensor
        The y plane of the receiver.
    resolution_x : int
        The resolution of the x plane of the receiver.
    resolution_y : int
        The resolution of the y plane of the receiver.
    """

    def __init__(
        self,
        center: torch.Tensor,
        plane_normal: torch.Tensor,
        plane_x: float,
        plane_y: float,
        resolution_x: int,
        resolution_y: int,
    ) -> None:
        """
        Initialize the receiver.

        Parameters
        ----------
        center : torch.Tensor
            The center of the receiver.
        plane_normal : torch.Tensor
            The normal to the plane of the receiver.
        plane_x : float
            The x plane of the receiver.
        plane_y : torch.Tensor
            The y plane of the receiver.
        resolution_x : int
            The resolution of the x plane of the receiver.
        resolution_y : int
            The resolution of the y plane of the receiver
        """
        super().__init__()

        self.center = center
        self.plane_normal = plane_normal
        self.plane_x = plane_x
        self.plane_y = plane_y
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y

    @classmethod
    def from_hdf5(cls, config_file: h5py.File) -> Self:
        """
        Class method that initializes a receiver from an hdf5 file.

        Parameters
        ----------
        config_file : h5py.File
            The hdf5 file containing the information about the receiver.

        Returns
        -------
        Receiver
            A receiver initialized from an hdf5 file.
        """
        center = torch.tensor(
            config_file[config_dictionary.receiver_prefix][
                config_dictionary.receiver_center
            ][()],
            dtype=torch.float,
        )
        plane_normal = torch.tensor(
            config_file[config_dictionary.receiver_prefix][
                config_dictionary.receiver_plane_normal
            ][()],
            dtype=torch.float,
        )
        plane_x = float(
            config_file[config_dictionary.receiver_prefix][
                config_dictionary.receiver_plane_x
            ][()]
        )
        plane_y = float(
            config_file[config_dictionary.receiver_prefix][
                config_dictionary.receiver_plane_y
            ][()]
        )
        resolution_x = int(
            config_file[config_dictionary.receiver_prefix][
                config_dictionary.receiver_resolution_x
            ][()]
        )
        resolution_y = int(
            config_file[config_dictionary.receiver_prefix][
                config_dictionary.receiver_resolution_y
            ][()]
        )

        return cls(
            center=center,
            plane_normal=plane_normal,
            plane_x=plane_x,
            plane_y=plane_y,
            resolution_x=resolution_x,
            resolution_y=resolution_y,
        )
