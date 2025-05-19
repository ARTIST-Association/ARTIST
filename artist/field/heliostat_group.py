"""Heliostat group in ARTIST."""

from typing import Union

import torch


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
        Align all surface points and surface normals of all heliostats in the group.
    get_orientations_from_motor_positions()
        Compute the orientations of all heliostats given some motor positions.
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

    def align_surfaces_with_incident_ray_directions(
        self,
        incident_ray_directions: torch.Tensor,
        active_heliostats_indices: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """
        Align all surface points and surface normals of all heliostats in the group.

        This method uses the incident ray direction to align the heliostats.

        Parameters
        ----------
        incident_ray_direction : torch.Tensor
            The incident ray direction.
        active_heliostats_indices : torch.Tensor
            The indices of the active heliostats to be aligned.
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
        active_heliostats_indices: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> torch.Tensor:
        """
        Compute the orientations of all heliostats given some motor positions.

        Parameters
        ----------
        motor_positions : torch.Tensor
            The motor positions.
        active_heliostats_indices : torch.Tensor
            The indices of the active heliostats to be aligned.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Raises
        ------
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError("Must be overridden!")

    def forward(self) -> None:
        """
        Specify the forward pass.

        Raises
        ------
        NotImplementedError
            Whenever called.
        """
        raise NotImplementedError("Must be overridden!")
