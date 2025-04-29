import logging
from typing import Union

import torch

from artist.field.heliostat_group import HeliostatGroup
from artist.field.kinematic_rigid_body import RigidBody

log = logging.getLogger(__name__)
"""A logger for the heliostat groups with a rigid body kinematic."""

class HeliostatGroupRigidBody(HeliostatGroup):
    """
    The groups of heliostats using a rigid body kinematic.

    The rigid body kinematic works with either linear or ideal actuators. Heliostats with
    differing actuator types belong to different groups even if the kinematic type is the same.
    The `HeliostatGroupRigidBody` can be used for the initialization of both groups.

    Attributes
    ----------
    number_of_heliostats : int
        The number of heliostats in the group.
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
    aligned_heliostats : torch.Tensor
        Information about alignment of heliostats.
        Unaligned heliostats marked with 0, aligned heliostats marked with 1.
    preferred_reflection_directions : torch.Tensor
        The preferred reflection directions of all heliostats in the group.
    current_aligned_surface_points : torch.Tensor
        The aligned surface points of all heliostats in the group.
    current_aligned_surface_normals : torch.Tensor
        The aligned surface normals of all heliostats in the group.
    rigid_body_kinematic : RigidBody
        The kinematic of all heliostats in the group.

    Methods
    -------
    align_surfaces_with_incident_ray_direction()
        Align all surface points and surface normals of all heliostats in the group.
    get_orientations_from_motor_positions()
        Compute the orientations of all heliostats given some motor positions.
    align_surfaces_with_motor_positions()
        Align all surface points and surface normals of all heliostats in the group.
    forward()
        Specify the forward pass.

    See Also
    --------
    :class:`HeliostatGroup` : Reference to the parent class.
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
        Initialize a heliostat group with a rigid body kinematic and linear or ideal actuator type.

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
        super(HeliostatGroupRigidBody, self).__init__()
        device = torch.device(device)

        self.number_of_heliostats = len(names)    
        self.names = names
        self.positions = positions
        self.aim_points = aim_points
        self.surface_points = surface_points
        self.surface_normals = surface_normals
        self.initial_orientations = initial_orientations
        self.kinematic_deviation_parameters = kinematic_deviation_parameters
        self.actuator_parameters = actuator_parameters

        self.aligned_heliostats = torch.zeros(self.number_of_heliostats, device=device)
        self.preferred_reflection_directions = torch.zeros_like(
            self.surface_normals, device=device
        )
        self.current_aligned_surface_points = torch.zeros_like(
            self.surface_points, device=device
        )
        self.current_aligned_surface_normals = torch.zeros_like(
            self.surface_normals, device=device
        )

        self.rigid_body_kinematic = RigidBody(
            number_of_heliostats=self.number_of_heliostats,
            heliostat_positions=self.positions,
            aim_points=self.aim_points,
            actuator_parameters=self.actuator_parameters,
            initial_orientations=self.initial_orientations,
            deviation_parameters=self.kinematic_deviation_parameters,
            device=device,
        )

    def align_surfaces_with_incident_ray_direction(
        self,
        incident_ray_direction: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """
        Align all surface points and surface normals of all heliostats in the group.

        This method uses the incident ray direction to align the heliostats.

        Parameters
        ----------
        incident_ray_direction : torch.Tensor
            The incident ray direction.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).
        """
        device = torch.device(device)

        (
            self.current_aligned_surface_points,
            self.current_aligned_surface_normals,
        ) = self.rigid_body_kinematic.align_surfaces_with_incident_ray_direction(
            incident_ray_direction=incident_ray_direction,
            surface_points=self.surface_points,
            surface_normals=self.surface_normals,
            device=device,
        )

        # Note that all heliostats have been aligned.
        self.aligned_heliostats = torch.ones_like(self.aligned_heliostats)


    def get_orientations_from_motor_positions(
        self,
        motor_positions: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> torch.Tensor:
        """
        Compute the orientations of all heliostats given some motor positions.

        Parameters
        ----------
        motor_positions : torch.Tensor
            The motor positions.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        torch.Tensor
            The orientations of the heliostats for the given motor positions.
        """
        device = torch.device(device)
        
        return self.rigid_body_kinematic.motor_positions_to_orientations(
            motor_positions, device
        )

    
    def align_surfaces_with_motor_positions(
        self,
        motor_positions: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """
        Align all surface points and surface normals of all heliostats in the group.

        This method uses the motor positions to align the heliostats.

        Parameters
        ----------
        motor_positions : torch.Tensor
            The motor positions.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).
        """
        device = torch.device(device)

        (
            self.current_aligned_surface_points,
            self.current_aligned_surface_normals,
        ) = self.rigid_body_kinematic.align_surfaces_with_motor_positions(
            motor_positions=motor_positions,
            surface_points=self.surface_points,
            surface_normals=self.surface_normals,
            device=device,
        )

        # Note that all heliostats have been aligned.
        self.aligned_heliostats = torch.ones_like(self.aligned_heliostats)

    def forward(self) -> None:
        """
        Specify the forward pass.

        Raises
        ------
        NotImplementedError
            Whenever called.
        """
        raise NotImplementedError("Not Implemented!")
