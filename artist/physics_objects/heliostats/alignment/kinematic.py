import typing
import torch
from artist.physics_objects.module import AModule


class AKinematicModule(AModule):
    """
    Abstract base class for all kinematic modules.

    See Also
    --------
    :class: AModule : Reference to the parent class.
    """

    def __init__(self, position: torch.Tensor):
        super(AKinematicModule, self).__init__()
        self._position = position

    # data_point_tensor = tensor([ax1, ax2, sp_e, sp_n, sp_u, ap_e, ap_n, ap_u])
    def computeOrientation(
            self, data_point_tensor: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the orientation matrix to align the heliostat.

        Parameters
        ----------
        data_point_tensor
            contains the information about the heliostat, light source, receiver.

        Raises
        ------
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError("Must Be Overridden!")
    # data_point_tensor = tensor([ax1, ax2, sp_e, sp_n, sp_u, ap_e, ap_n, ap_u])
    def computeReflection(self, data_point_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute the aimpoints starting from the orientation of the heliostat.

        Parameters
        ----------
        data_point_tensor
            contains the information about the heliostat, light source, receiver.

        Returns
        -------
        torch.Tensor
            The derived aimpoints.
        """
        normal_vectors, normal_origins = self.computeOrientation(
            data_point_tensor=data_point_tensor
        )
        aim_vectors = (
                2 * (data_point_tensor[:, 2:5] @ normal_vectors) - data_point_tensor[:, 2:5]
        )
        aimpoints = (
                self._position
                + aim_vectors * (data_point_tensor[:, 5:] - self._position).norm()
        )
        return aimpoints

    def forward(self, data_points: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward kinematics.

        Parameters
        ----------
        data_point_tensor
            contains the information about the heliostat, light source, receiver.

        Returns
        -------
        torch.Tensor
            The derived aimpoints.

        """
        return self.computeReflection(data_points)

    def to_dict(self):
        return self.state_dict()
