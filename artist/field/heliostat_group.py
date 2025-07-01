"""Heliostat group in ARTIST."""

from abc import ABC

import torch

from artist.field.kinematic import Kinematic
from artist.util.environment_setup import get_device


class HeliostatGroup(ABC):
    """
    Abstract base class for all heliostat groups.

    The abstract heliostat group implements a template for the construction of inheriting heliostat groups, each
    with a specific kinematic type and specific actuator type. All heliostat groups together form the overall heliostat
    field. The abstract base class defines an align function that all heliostat groups need to overwrite
    in order to align the heliostats within this group.

    Attributes
    ----------
    number_of_heliostats : int
        The number of heliostats in the group.
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
    kinematic_deviation_parameters : torch.Tensor
        The kinematic deviation parameters of all heliostats in the group.
    actuator_parameters : torch.Tensor
        The actuator parameters of all actuators in the group.
    kinematic : Kinematic
        The kinematic of all heliostats in the group.
    number_of_active_heliostats : int
        The number of active heliostats.
    active_heliostats_mask : torch.Tensor
        A mask defining which heliostats are activated.
    active_surface_points : torch.Tensor
        The surface points of all active heliostats in the group, these can be aligned.
    active_surface_normals : torch.Tensor
        The surface normals of all active heliostats in the group, these can be aligned.
    preferred_reflection_directions : torch.Tensor
        The preferred reflection directions of all heliostats in the group.

    Methods
    -------
    align_surfaces_with_incident_ray_directions()
        Align surface points and surface normals with incident ray directions.
    activate_heliostats()
        Activate certain heliostats for alignment, raytracing or calibration.
    """

    def __init__(
        self,
        names: list[str],
        positions: torch.Tensor,
        surface_points: torch.Tensor,
        surface_normals: torch.Tensor,
        initial_orientations: torch.Tensor,
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
        surface_points : torch.Tensor
            The surface points of all heliostats in the group.
        surface_normals : torch.Tensor
            The surface normals of all heliostats in the group.
        initial_orientations : torch.Tensor
            The initial orientations of all heliostats in the group.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        super().__init__()

        device = get_device(device=device)

        self.number_of_heliostats = len(names)
        self.names = names
        self.positions = positions
        self.surface_points = surface_points
        self.surface_normals = surface_normals
        self.initial_orientations = initial_orientations

        self.kinematic = Kinematic()

        self.number_of_active_heliostats = 0
        self.active_heliostats_mask = torch.empty(
            self.number_of_heliostats, device=device
        )
        self.active_surface_points = torch.empty_like(
            self.surface_points, device=device
        )
        self.active_surface_normals = torch.empty_like(
            self.surface_normals, device=device
        )
        self.preferred_reflection_directions = torch.empty(
            (self.number_of_heliostats, 4), device=device
        )

    def align_surfaces_with_incident_ray_directions(
        self,
        aim_points: torch.Tensor,
        incident_ray_directions: torch.Tensor,
        active_heliostats_mask: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> None:
        """
        Align all surface points and surface normals of all heliostats in the group.

        This method uses the incident ray direction to align the heliostats.

        Parameters
        ----------
        aim_points : torch.Tensor
            The aim points for all active heliostats.
        incident_ray_directions : torch.Tensor
            The incident ray directions.
        active_heliostats_mask : torch.Tensor | None
            A mask where 0 indicates a deactivated heliostat and 1 an activated one (default is None).
            An integer greater than 1 indicates that this heliostat is regarded multiple times.
            If no mask is provided, all heliostats in the scenario will be activated once.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
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
        Activate certain heliostats for alignment, raytracing or calibration.

        Select and repeat indices of all active heliostat and kinematic parameters once according
        to the mask. Doing this once instead of slicing everytime when accessing one
        of those parameter tensors saves memory.

        Parameters
        ----------
        active_heliostats_mask : torch.Tensor | None
            A mask where 0 indicates a deactivated heliostat and 1 an activated one (default is None).
            An integer greater than 1 indicates that this heliostat is regarded multiple times.
            If no mask is provided, all heliostats in the scenario will be activated once.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
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
        self.kinematic.active_deviation_parameters = (
            self.kinematic.deviation_parameters.repeat_interleave(
                active_heliostats_mask, dim=0
            )
        )
        self.kinematic.actuators.active_actuator_parameters = (
            self.kinematic.actuators.actuator_parameters.repeat_interleave(
                active_heliostats_mask, dim=0
            )
        )
