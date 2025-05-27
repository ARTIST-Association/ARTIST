"""Heliostat group in ARTIST."""

from typing import Optional, Union

import torch

from artist.field.kinematic import Kinematic


class HeliostatGroup(torch.nn.Module):
    """
    Abstract base class for all heliostat groups.

    The abstract heliostat group implements a template for the construction of inheriting heliostat groups, each
    with a specific kinematic type and specific actuator type. All heliostat groups together form the overall heliostat
    field. The abstract base class defines an align function that all heliostat groups need to overwrite
    in order to align the heliostats within this group.

    Methods
    -------
    align_surfaces_with_incident_ray_directions()
        Align surface points and surface normals with incident ray directions.
    get_orientations_from_motor_positions()
        Compute the orientations of heliostats given some motor positions.
    forward()
        Specify the forward pass.
    """

    def __init__(
        self,
        names: list[str],
        positions: torch.Tensor,
        aim_points: torch.Tensor,
        surface_points: torch.Tensor,
        surface_normals: torch.Tensor,
        initial_orientations: torch.Tensor,
        kinematic_deviation_parameters: torch.Tensor,
        actuator_parameters: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """
        Initialize the heliostat group.

        Parameters
        ----------
        names : list[str]
            The string names of each heliostat in the group in order.
        positions : torch.Tensor
            The positions of all heliostats in the group.
        aim_points : torch.Tensor
            The aim points of all heliostats in the group.
        surface_points : torch.Tensor
            The surface points of all heliostats in the group.
        surface_normals : torch.Tensor
            The surface normals of all heliostats in the group.
        initial_orientations : torch.Tensor
            The initial orientations of all heliostats in the group.
        kinematic_deviation_parameters : torch.Tensor
            The kinematic deviation parameters of all heliostats in the group.
        actuator_parameters : torch.Tensor
            The actuator parameters of all actuators in the group.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).
        """
        super().__init__()
        device = torch.device(device)

        self.number_of_heliostats = len(names)
        self.names = names
        self.positions = positions
        self.surface_points = surface_points
        self.surface_normals = surface_normals
        self.initial_orientations = initial_orientations
        self.kinematic_deviation_parameters = kinematic_deviation_parameters
        self.actuator_parameters = actuator_parameters

        self.kinematic = Kinematic()

        self.aligned_heliostats = 0
        self.number_of_active_heliostats = 0
        self.active_heliostats_mask = None
        self.active_surface_points = None
        self.active_surface_normals = None
        self.preferred_reflection_directions = None

    def align_surfaces_with_incident_ray_directions(
        self,
        incident_ray_directions: torch.Tensor,
        active_heliostats_mask: Optional[torch.Tensor] = None,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """
        Align all surface points and surface normals of all heliostats in the group.

        This method uses the incident ray direction to align the heliostats.

        Parameters
        ----------
        incident_ray_direction : torch.Tensor
            The incident ray direction.
        active_heliostats_mask : Optional[torch.Tensor]
            A mask for the active heliostats that will be aligned (default is None).
            If no mask is provided, all heliostats will be activated.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Raises
        ------
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError("Must be overridden!")

    def get_orientations_from_motor_positions(
        self,
        motor_positions: torch.Tensor,
        active_heliostats_mask: Optional[torch.Tensor] = None,
        device: Union[torch.device, str] = "cuda",
    ) -> torch.Tensor:
        """
        Compute the orientations of heliostats given some motor positions.

        Parameters
        ----------
        motor_positions : torch.Tensor
            The motor positions.
        active_heliostats_mask : Optional[torch.Tensor]
            A mask for the active heliostats that will be aligned (default is None).
            If no mask is provided, all heliostats will be activated.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Raises
        ------
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError("Must be overridden!")
    
    def activate_heliostats(self, active_heliostats_mask: torch.Tensor) -> None:
        """
        Activate certain heliostats for alignment, raytracing or calibration.

        Select and repeat indices of all active heliostat and kinematic parameters once according
        to the mask. Doing this once instead of slicing everytime when accessing one
        of those parameter tensors saves memory.
        
        Parameters
        ----------
        active_heliostats_mask : torch.Tensor
            A mask where 0 indicates a deactivated heliostat and 1 an activated one.
            An integer greater than 1 indicates that this heliostat is regarded multiple times.
        """
        self.number_of_active_heliostats = active_heliostats_mask.sum().item()
        self.active_heliostats_mask = active_heliostats_mask
        self.active_surface_points = self.surface_points.repeat_interleave(active_heliostats_mask, dim=0)
        self.active_surface_normals = self.surface_normals.repeat_interleave(active_heliostats_mask, dim=0)
        self.kinematic.number_of_active_heliostats = active_heliostats_mask.sum().item()
        self.kinematic.active_heliostat_positions = self.kinematic.heliostat_positions.repeat_interleave(active_heliostats_mask, dim=0)
        self.kinematic.active_initial_orientations = self.kinematic.initial_orientations.repeat_interleave(active_heliostats_mask, dim=0)
        self.kinematic.active_deviation_parameters = self.kinematic.deviation_parameters.repeat_interleave(active_heliostats_mask, dim=0)
        self.kinematic.actuators.active_actuator_parameters = self.kinematic.actuators.actuator_parameters.repeat_interleave(active_heliostats_mask, dim=0)


    def forward(self) -> None:
        """
        Specify the forward pass.

        Raises
        ------
        NotImplementedError
            Whenever called.
        """
        raise NotImplementedError("Must be overridden!")
