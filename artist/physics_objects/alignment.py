"""Alignment module for the heliostat."""

from typing import Any, Dict, Tuple

import torch


class AlignmentModule(torch.nn.Module):
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
    """

    def __init__(
        self,
        alignment_type: Any,
        actuator_type: Any,
        position: torch.Tensor,
        aim_point: torch.Tensor,
        kinematic_deviation_parameters: Dict[str, torch.Tensor],
        kinematic_initial_orientation_offsets: Dict[str, torch.Tensor],
        actuator_parameters: Dict[str, torch.Tensor],
    ) -> None:
        """
        Initialize the alignment module.

        Parameters
        ----------
        alignment_type : Any
            The method by which the heliostat is aligned, currently only rigid-body is possible.
        actuator_type : Any
            The type of the actuators of the heliostat.
        position : torch.Tensor
            Position of the heliostat for which the alignment model is created.
        aim_point : torch.Tensor
            The aimpoint.
        kinematic_deviation_parameters : Dict[str, torch.Tensor]
            The 18 deviation parameters of the kinematic module.
        kinematic_initial_orientation_offsets : Dict[str, torch.Tensor]
            The initial orientation-rotation angles of the heliostat.
        actuator_parameters : Dict[str, torch.Tensor]
            The parameters describing the imperfect actuator.
        """
        super().__init__()

        self.kinematic_model = alignment_type(
            actuator_type=actuator_type,
            position=position,
            aim_point=aim_point,
            deviation_parameters=kinematic_deviation_parameters,
            initial_orientation_offsets=kinematic_initial_orientation_offsets,
            actuator_parameters=actuator_parameters,
        )

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
        torch.Tensor
            The aligned surface points.
        torch.Tensor
            The aligned surface normals.
        """
        orientation = self.kinematic_model.align(incident_ray_direction).squeeze()
        aligned_surface_points = (orientation @ surface_points.T).T
        aligned_surface_normals = (orientation @ surface_normals.T).T

        return (aligned_surface_points, aligned_surface_normals)
