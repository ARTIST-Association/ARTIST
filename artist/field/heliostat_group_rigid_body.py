import logging

import torch

from artist.field.heliostat_group import HeliostatGroup
from artist.field.kinematic_rigid_body import RigidBody
from artist.util.environment_setup import get_device

log = logging.getLogger(__name__)
"""A logger for the heliostat groups with a rigid body kinematic."""


class HeliostatGroupRigidBody(HeliostatGroup):
    """
    The groups of heliostats using a rigid body kinematic.

    The rigid body kinematic works with either linear or ideal actuators. Heliostats with
    differing actuator types belong to different groups even if the kinematic type is the same.
    The `HeliostatGroupRigidBody` can be used for the initialization of both groups.
    Individual heliostats in the same group are not saved as separate entities, instead
    separate tensors for each heliostat property exist. Each property tensor or list contains
    information about this property for all heliostats within this group.

    Attributes
    ----------
    number_of_heliostats : int
        The number of heliostats in the group.
    number_of_facets_per_heliostat
        The number of facets per heliostat in the group.
    names : list[str]
        The string names of each heliostat in the group in order.
    positions : torch.Tensor
        The positions of all heliostats in the group.
    surface_points : torch.Tensor
        The surface points of all heliostats in the group.
    surface_normals : torch.Tensor
        The surface normals of all heliostats in the group.
    initial_orientations : torch.Tensor
        The initial orientations of all heliostats in the group.
    nurbs_control_points : torch.Tensor
        The control points for NURBS surfaces for all heliostats in the group.
    nurbs_degrees : torch.Tensor
        The degrees for NURBS surfaces for all heliostats in the group.
    kinematic : RigidBody
        The kinematic (rigid body kinematic) of all heliostats in the group.
    number_of_active_heliostats : int
        The number of active heliostats.
    active_heliostats_mask : torch.Tensor
        A mask defining which heliostats are activated.
    active_surface_points : torch.Tensor
        The surface points of all active heliostats in the group, these can be aligned.
    active_surface_normals : torch.Tensor
        The surface normals of all active heliostats in the group, these can be aligned.
    active_nurbs_control_points : torch.Tensor
        The NURBS control points of all active heliostats in the group, these can be learned.
    preferred_reflection_directions : torch.Tensor
        The preferred reflection directions of all heliostats in the group.

    Methods
    -------
    align_surfaces_with_incident_ray_directions()
        Align surface points and surface normals with incident ray directions.

    See Also
    --------
    :class:`HeliostatGroup` : Reference to the parent class.
    """

    def __init__(
        self,
        names: list[str],
        positions: torch.Tensor,
        surface_points: torch.Tensor,
        surface_normals: torch.Tensor,
        initial_orientations: torch.Tensor,
        nurbs_control_points: torch.Tensor,
        nurbs_degrees: torch.Tensor,
        kinematic_deviation_parameters: torch.Tensor,
        actuator_parameters: torch.Tensor,
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize a heliostat group with a rigid body kinematic and linear or ideal actuator type.

        Parameters
        ----------
        names : list[str]
            The string names of each heliostat in the group in order.
        positions : torch.Tensor
            The positions of all heliostats in the group.
        surface_points : torch.Tensor
            The surface points of all heliostats in the group.
        surface_normals : torch.Tensor
            The surface normals of all heliostats in the group.
        initial_orientations : torch.Tensor
            The initial orientations of all heliostats in the group.
        nurbs_control_points : torch.Tensor
            The control points for NURBS surfaces for all heliostats in the group.
        nurbs_degrees : torch.Tensor
            The degrees for NURBS surfaces for all heliostats in the group.
        kinematic_deviation_parameters : torch.Tensor
            The kinematic deviation parameters of all heliostats in the group.
        actuator_parameters : torch.Tensor
            The actuator parameters of all actuators in the group.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        super().__init__(
            names=names,
            positions=positions,
            surface_points=surface_points,
            surface_normals=surface_normals,
            initial_orientations=initial_orientations,
            nurbs_control_points=nurbs_control_points,
            nurbs_degrees=nurbs_degrees,
            device=device,
        )

        self.kinematic = RigidBody(
            number_of_heliostats=self.number_of_heliostats,
            heliostat_positions=self.positions,
            actuator_parameters=actuator_parameters,
            initial_orientations=self.initial_orientations,
            deviation_parameters=kinematic_deviation_parameters,
            device=device,
        )

    def align_surfaces_with_incident_ray_directions(
        self,
        aim_points: torch.Tensor,
        incident_ray_directions: torch.Tensor,
        active_heliostats_mask: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> None:
        """
        Align surface points and surface normals with incident ray directions.

        This method uses incident ray directions to align the heliostats. It is possible to
        have different incident ray directions for different helisotats, for example during
        calibration tasks. Heliostats can be selected and deselected if only a subset should
        be aligned with the active heliostat indices parameter.

        Parameters
        ----------
        aim_points : torch.Tensor
            The aim points for all active heliostats.
        incident_ray_directions : torch.Tensor
            The incident ray directions.
        active_heliostats_mask : torch.Tensor
            A mask where 0 indicates a deactivated heliostat and 1 an activated one.
            An integer greater than 1 indicates that this heliostat is regarded multiple times.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Raises
        ------
        ValueError
            If not all heliostats trying to be aligned have been activated.
        """
        device = get_device(device=device)

        assert torch.equal(self.active_heliostats_mask, active_heliostats_mask), (
            "Some heliostats were not activated and cannot be aligned."
        )

        orientations = self.kinematic.incident_ray_directions_to_orientations(
            incident_ray_directions=incident_ray_directions,
            aim_points=aim_points,
            device=device,
        )

        self.active_surface_points = (
            self.active_surface_points @ orientations.transpose(1, 2)
        )
        self.active_surface_normals = (
            self.active_surface_normals @ orientations.transpose(1, 2)
        )
