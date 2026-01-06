"""Heliostat group in ``ARTIST``."""

import torch

from artist.field.kinematic import Kinematic
from artist.util import index_mapping
from artist.util.environment_setup import get_device


class HeliostatGroup:
    """
    Abstract base class for all heliostat groups.

    The abstract heliostat group implements a template for the construction of inheriting heliostat groups, each
    with a specific kinematic type and specific actuator type. All heliostat groups together form the overall heliostat
    field. The abstract base class defines an align function that all heliostat groups need to overwrite
    in order to align the heliostats within this group. The heliostat groups will be initialized with no active
    heliostats. The heliostats have to be selected and activated before alignment, raytracing or optimization can begin.
    The size of the first dimension of all ``active_...``-attributes varies depending on how many heliostats have been
    activated.

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
    kinematic : Kinematic
        The kinematic of all heliostats in the group.
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
    """

    def __init__(
        self,
        names: list[str],
        positions: torch.Tensor,
        surface_points: torch.Tensor,
        surface_normals: torch.Tensor,
        canting: torch.Tensor,
        facet_translations: torch.Tensor,
        initial_orientations: torch.Tensor,
        nurbs_control_points: torch.Tensor,
        nurbs_degrees: torch.Tensor,
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize the heliostat group.

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
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        device = get_device(device=device)

        self.number_of_heliostats = len(names)
        self.number_of_facets_per_heliostat = nurbs_control_points.shape[
            index_mapping.facet_dimension
        ]
        self.names = names
        self.positions = positions
        self.surface_points = surface_points
        self.surface_normals = surface_normals
        self.canting = canting
        self.facet_translations = facet_translations
        self.initial_orientations = initial_orientations

        self.nurbs_control_points = nurbs_control_points
        self.nurbs_degrees = nurbs_degrees

        self.kinematic = Kinematic()

        self.number_of_active_heliostats = 0
        self.active_heliostats_mask = torch.zeros(
            self.number_of_heliostats, device=device
        )
        self.active_surface_points = torch.empty_like(
            self.surface_points, device=device
        )
        self.active_surface_normals = torch.empty_like(
            self.surface_normals, device=device
        )
        self.active_canting = torch.empty_like(
            self.canting, device=device
        )
        self.active_facet_translations = torch.empty_like(
            self.facet_translations, device=device
        )
        self.active_nurbs_control_points = torch.empty_like(
            self.nurbs_control_points, device=device
        )
        self.preferred_reflection_directions = torch.empty(
            (self.number_of_heliostats, 4), device=device
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
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError("Must be overridden!")

    def align_surfaces_with_motor_positions(
        self,
        motor_positions: torch.Tensor,
        active_heliostats_mask: torch.Tensor,
        device: torch.device | None = None,
    ) -> None:
        """
        Align surface points and surface normals with motor positions.

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
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError("Must be overridden!")

    def activate_heliostats(
        self,
        active_heliostats_mask: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> None:
        """
        Activate certain heliostats for alignment, raytracing or optimization.

        Select and repeat indices of all active heliostat and kinematic parameters once according
        to the mask. Doing this once instead of slicing every time when accessing one
        of those parameter tensors saves memory.

        Parameters
        ----------
        active_heliostats_mask : torch.Tensor | None
            A mask where 0 indicates a deactivated heliostat and 1 an activated one (default is None).
            An integer greater than 1 indicates that this heliostat is regarded multiple times.
            If no mask is provided, all heliostats in the scenario will be activated once.
            Tensor of shape [number_of_heliostats].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        device = get_device(device=device)

        if active_heliostats_mask is None:
            active_heliostats_mask = torch.ones(
                self.number_of_heliostats, dtype=torch.int32, device=device
            )

        self.number_of_active_heliostats = active_heliostats_mask.sum().item()
        self.active_heliostats_mask = active_heliostats_mask
        self.active_surface_points = self.surface_points.repeat_interleave(
            active_heliostats_mask, dim=0
        )
        self.active_surface_normals = self.surface_normals.repeat_interleave(
            active_heliostats_mask, dim=0
        )
        self.active_canting = self.canting.repeat_interleave(
            active_heliostats_mask, dim=0
        )
        self.active_facet_translations = self.facet_translations.repeat_interleave(
            active_heliostats_mask, dim=0
        )
        self.active_nurbs_control_points = self.nurbs_control_points.repeat_interleave(
            active_heliostats_mask, dim=0
        )
        self.kinematic.number_of_active_heliostats = active_heliostats_mask.sum().item()
        self.kinematic.active_heliostat_positions = (
            self.kinematic.heliostat_positions.repeat_interleave(
                active_heliostats_mask, dim=0
            )
        )
        self.kinematic.active_initial_orientations = (
            self.kinematic.initial_orientations.repeat_interleave(
                active_heliostats_mask, dim=0
            )
        )
        self.kinematic.active_translation_deviation_parameters = (
            self.kinematic.translation_deviation_parameters.repeat_interleave(
                active_heliostats_mask, dim=0
            )
        )
        self.kinematic.active_rotation_deviation_parameters = (
            self.kinematic.rotation_deviation_parameters.repeat_interleave(
                active_heliostats_mask, dim=0
            )
        )
        self.kinematic.active_motor_positions = (
            self.kinematic.motor_positions.repeat_interleave(
                active_heliostats_mask, dim=0
            )
        )
        self.kinematic.actuators.active_non_optimizable_parameters = (
            self.kinematic.actuators.non_optimizable_parameters.repeat_interleave(
                active_heliostats_mask, dim=0
            )
        )
        if self.kinematic.actuators.active_optimizable_parameters.numel() > 0:
            self.kinematic.actuators.active_optimizable_parameters = (
                self.kinematic.actuators.optimizable_parameters.repeat_interleave(
                    active_heliostats_mask, dim=0
                )
            )
        else:
            self.kinematic.actuators.active_optimizable_parameters = torch.tensor(
                [], requires_grad=True
            )
