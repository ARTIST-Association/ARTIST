"""
Alignment module for the heliostat.
"""
from typing import Dict, Tuple

import torch

from artist.physics_objects.module import AModule
from artist.util import artist_type_mapping_dict


class AlignmentModule(AModule):
    """
    This class implements the alignment module for the heliostat.

    Attributes
    ----------
    kinematic_model : Union[RigidBodyModule, ...]
        The kinematic model used.

    Methods
    -------
    align_surface()
        Align given surface points and surface normals according to a calculated orientation.

    See Also
    --------
    :class: AModule : The parent class.
    """

    def __init__(
        self,
        alignment_type: str,
        actuator_type: str,
        position: torch.tensor,
        aim_point: torch.tensor,
        kinematic_deviation_parameters: Dict[str, torch.Tensor],
        kinematic_initial_orientation_offset: float,
    ) -> None:
        """
        Initialize the alignment module.

        Parameters
        ----------
        alignment_type : str
            The method by which the helisotat is aligned, currently only rigid-body is possible.
        actuator_type : str
            The type of the actuators of the heliostat.   
        position : torch.Tensor
            Position of the heliostat for which the alignment model is created.
        aim_point : torch.Tensor
            The aimpoint.
        kinematic_deviation_parameters : Dict[str, torch.Tensor]
            The 18 deviation parameters of the kinematic module.
        kinematic_initial_orientation_offset : float
            The initial orientation-rotation angle of the heliostat.
        """
        super().__init__()
        try:
            self.kinematic_model = artist_type_mapping_dict.alignment_type_mapping[alignment_type](
                actuator_type=actuator_type,
                position=position,
                aim_point=aim_point,
                deviation_parameters=kinematic_deviation_parameters,
                initial_orientation_offset=kinematic_initial_orientation_offset,
            )
        except:
            raise KeyError(f"Currently the selected alignment type: {alignment_type} is not supported.")

    def align_surface(
        self,
        incident_ray_direction: torch.Tensor,
        surface_points: torch.Tensor,
        surface_normals: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align given surface points and surface normals according to a calculated orientation.

        Parameters
        ----------
        incident_ray_direction : torch.Tensor
            The direction of the rays.
        surface_points : torch.Tensor
            Points on the surface of the heliostat that reflect the light.
        surface_normals : torch.Tensor
            Normals to the surface points.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple containing the aligned surface points and normals.
        """
        orientation = self.kinematic_model.align(incident_ray_direction).squeeze()
        aligned_surface_points = (orientation @ surface_points.T).T
        aligned_surface_normals = (orientation @ surface_normals.T).T

        return (aligned_surface_points, aligned_surface_normals)
