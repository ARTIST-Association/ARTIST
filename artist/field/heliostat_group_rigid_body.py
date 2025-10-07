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
    number_of_facets_per_heliostat : int
        The number of facets per heliostat in the group.
    names : list[str]
        The string names of each heliostat in the group in order.
    positions : torch.Tensor
        The positions of all heliostats in the group.
        Tensor of shape [number_of_heliostats, 4].
    surface_points : torch.Tensor
        The surface points of all heliostats in the group.
        Tensor of shape [number_of_heliostats, number_of_combined_surface_points_all_facets, 4].
    surface_normals : torch.Tensor
        The surface normals of all heliostats in the group.
        Tensor of shape [number_of_heliostats, number_of_combined_surface_normals_all_facets, 4].
    initial_orientations : torch.Tensor
        The initial orientations of all heliostats in the group.
        Tensor of shape [number_of_heliostats, 4].
    nurbs_control_points : torch.Tensor
        The control points for NURBS surfaces for all heliostats in the group.
        Tensor of shape [number_of_heliostats, number_of_facets_per_heliostat, number_of_control_points_u_direction, number_of_control_points_v_direction 3].
    nurbs_degrees : torch.Tensor
        The spline degrees for NURBS surfaces in u and then in v direction, for all heliostats in the group.
        Tensor of shape [2].
    kinematic : RigidBody
        The kinematic (rigid body kinematic) of all heliostats in the group.
    number_of_active_heliostats : int
        The number of active heliostats.
    active_heliostats_mask : torch.Tensor
        A mask defining which heliostats are activated.
        Tensor of shape [number_of_heliostats].
    active_surface_points : torch.Tensor
        The surface points of all active heliostats in the group, these can be aligned.
        Tensor of shape [number_of_active_heliostats, number_of_combined_surface_points_all_facets, 4].
    active_surface_normals : torch.Tensor
        The surface normals of all active heliostats in the group, these can be aligned.
        Tensor of shape [number_of_active_heliostats, number_of_combined_surface_normals_all_facets, 4].
    active_nurbs_control_points : torch.Tensor
        The NURBS control points of all active heliostats in the group, these can be learned.
        Tensor of shape [number_of_active_heliostats, number_of_facets_per_heliostat, number_of_control_points_u_direction, number_of_control_points_v_direction 3].
    preferred_reflection_directions : torch.Tensor
        The preferred reflection directions of all heliostats in the group.
        Tensor of shape [number_of_active_heliostats, number_of_combined_surface_normals_all_facets, 4].

    Methods
    -------
    align_surfaces_with_incident_ray_directions()
        Align surface points and surface normals with incident ray directions.
    align_surfaces_with_motor_positions()
        Align surface points and surface normals with motor positions.
    activate_heliostats()
        Activate certain heliostats for alignment, raytracing or calibration.

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
            Tensor of shape [number_of_heliostats, 4].
        surface_points : torch.Tensor
            The surface points of all heliostats in the group.
            Tensor of shape [number_of_heliostats, number_of_combined_surface_points_all_facets, 4].
        surface_normals : torch.Tensor
            The surface normals of all heliostats in the group.
            Tensor of shape [number_of_heliostats, number_of_combined_surface_normals_all_facets, 4].
        initial_orientations : torch.Tensor
            The initial orientations of all heliostats in the group.
            Tensor of shape [number_of_heliostats, 4].
        nurbs_control_points : torch.Tensor
            The control points for NURBS surfaces for all heliostats in the group.
            Tensor of shape [number_of_heliostats, number_of_facets_per_heliostat, number_of_control_points_u_direction, number_of_control_points_v_direction 3].
        nurbs_degrees : torch.Tensor
            The spline degrees for NURBS surfaces in u and then in v direction, for all heliostats in the group.
            Tensor of shape [2].
        kinematic_deviation_parameters : torch.Tensor
            The kinematic deviation parameters of all heliostats in the group.
        actuator_parameters : torch.Tensor
            The actuator parameters of all actuators in the group.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
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
        active_heliostats_mask: torch.Tensor,
        device: torch.device | None = None,
    ) -> None:
        """
        Align surface points and surface normals with incident ray directions.

        This method uses incident ray directions to align the heliostats. It is possible to
        have different incident ray directions for different heliostats, for example during
        calibration tasks. Only active heliostats can be aligned.

        Parameters
        ----------
        aim_points : torch.Tensor
            The aim points for all active heliostats.
            Tensor of shape [number_of_active_heliostats, 4].
        incident_ray_directions : torch.Tensor
            The incident ray directions.
            Tensor of shape [number_of_active_heliostats, 4].
        active_heliostats_mask : torch.Tensor
            A mask where 0 indicates a deactivated heliostat and 1 an activated one.
            An integer greater than 1 indicates that this heliostat is regarded multiple times.
            Tensor of shape [number_of_heliostats].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
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

    def align_surfaces_with_motor_positions(
        self,
        motor_positions: torch.Tensor,
        active_heliostats_mask: torch.Tensor,
        device: torch.device | None = None,
    ) -> None:
        """
        Align surface points and surface normals with motor positions.

        Only active heliostats can be aligned.

        Parameters
        ----------
        motor_positions : torch.Tensor
            The motor positions for all active heliostats.
            Tensor of shape [number_of_active_heliostats, 2].
        active_heliostats_mask : torch.Tensor
            A mask where 0 indicates a deactivated heliostat and 1 an activated one.
            An integer greater than 1 indicates that this heliostat is regarded multiple times.
            Tensor of shape [number_of_heliostats].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
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

        orientations = self.kinematic.motor_positions_to_orientations(
            motor_positions=motor_positions,
            device=device,
        )

        self.active_surface_points = (
            self.active_surface_points @ orientations.transpose(1, 2)
        )
        self.active_surface_normals = (
            self.active_surface_normals @ orientations.transpose(1, 2)
        )
