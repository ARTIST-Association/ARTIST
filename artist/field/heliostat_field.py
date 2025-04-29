import logging
from typing import Union

import h5py
import torch.nn
from typing_extensions import Self

from artist.field.kinematic_rigid_body import RigidBody
from artist.field.surface import Surface
from artist.util import config_dictionary, utils_load_h5
from artist.util.configuration_classes import (
    SurfaceConfig,
)

log = logging.getLogger(__name__)
"""A logger for the heliostat field."""


class HeliostatField(torch.nn.Module):
    """
    The heliostat field.

    A heliostat field consists of one or many heliostats that have a unique position in the field. The
    heliostats in the field are aligned individually to reflect the incoming light in a way that
    ensures maximum efficiency for the whole power plant.


    Attributes
    ----------
    number_of_heliostats : int
        The number of heliostats in the field.
    all_heliostat_names : list[str]
        The string names of each heliostat in the field in the correct order.
    all_heliostat_positions : torch.Tensor
        The heliostat positions of all heliostats in the field.
    all_aim_points : torch.Tensor
        The aim points of all heliostats in the field.
    all_surface_points : torch.Tensor
        The surface points of all heliostats in the field.
    all_surface_normals : torch.Tensor
        The surface normals of all heliostats in the field.
    all_initial_orientations : torch.Tensor
        The initial orientations of all heliostats in the field.
    all_kinematic_deviation_parameters : torch.Tensor
        The kinematic deviation parameters of all heliostats in the field.
    all_actuator_parameters : torch.Tensor
        The actuator parameters of all actuators in the field.
    all_aligned_heliostats : torch.Tensor
        Information about alignment of heliostats.
        Unaligned heliostats marked with 0, aligned heliostats marked with 1.
    all_preferred_reflection_directions : torch.Tensor
        The preferred reflection directions of all heliostats in the field.
    all_current_aligned_surface_points : torch.Tensor
        The aligned surface points of all heliostats in the field.
    all_current_aligned_surface_normals : torch.Tensor
        The aligned surface normals of all heliostats in the field.

    Methods
    -------
    from_hdf5()
        Load a heliostat field from an HDF5 file.
    align_surfaces_with_incident_ray_direction()
        Align all surface points and surface normals of all heliostats in the field.
    get_orientations_from_motor_positions()
        Compute the orientations of all heliostats given some motor positions.
    align_surfaces_with_motor_positions()
        Align all surface points and surface normals of all heliostats in the field.
    forward()
        Specify the forward pass.
    """

    def __init__(
        self,
        number_of_heliostats: int,
        all_heliostat_names: list[str],
        all_heliostat_positions: torch.Tensor,
        all_aim_points: torch.Tensor,
        all_surface_points: torch.Tensor,
        all_surface_normals: torch.Tensor,
        all_initial_orientations: torch.Tensor,
        all_kinematic_deviation_parameters: torch.Tensor,
        all_actuator_parameters: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """
        Initialize the heliostat field.

        Individual heliostats are not saved as separate entities, instead separate tensors
        for each heliostat property exist. Each property tensor contains information about this
        property for all heliostats in the field.

        Parameters
        ----------
        number_of_heliostats : int
            The number of heliostats in the field.
        all_heliostat_names : list[str]
            The string names of each heliostat in the field in the correct order.
        all_heliostat_positions : torch.Tensor
            The heliostat positions of all heliostats in the field.
        all_aim_points : torch.Tensor
            The aim points of all heliostats in the field.
        all_surface_points : torch.Tensor
            The surface points of all heliostats in the field.
        all_surface_normals : torch.Tensor
            The surface normals of all heliostats in the field.
        all_initial_orientations : torch.Tensor
            The initial orientations of all heliostats in the field.
        all_kinematic_deviation_parameters : torch.Tensor
            The kinematic deviation parameters of all heliostats in the field.
        all_actuator_parameters : torch.Tensor
            The actuator parameters of all actuators in the field.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).
        """
        super(HeliostatField, self).__init__()
        self.number_of_heliostats = number_of_heliostats
        self.all_heliostat_names = all_heliostat_names
        self.all_heliostat_positions = all_heliostat_positions
        self.all_aim_points = all_aim_points
        self.all_surface_points = all_surface_points
        self.all_surface_normals = all_surface_normals
        self.all_initial_orientations = all_initial_orientations
        self.all_kinematic_deviation_parameters = all_kinematic_deviation_parameters
        self.all_actuator_parameters = all_actuator_parameters

        self.all_aligned_heliostats = torch.zeros(number_of_heliostats, device=device)
        self.all_preferred_reflection_directions = torch.zeros_like(
            all_surface_normals, device=device
        )
        self.all_current_aligned_surface_points = torch.zeros_like(
            all_surface_points, device=device
        )
        self.all_current_aligned_surface_normals = torch.zeros_like(
            all_surface_normals, device=device
        )

    @classmethod
    def from_hdf5(
        cls,
        config_file: h5py.File,
        number_of_heliostats: int,
        number_of_surface_points_per_heliostat: int,
        prototype_surface: SurfaceConfig,
        prototype_initial_orientation: torch.Tensor,
        prototype_kinematic_deviations: torch.Tensor,
        prototype_actuators: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> Self:
        """
        Load a heliostat field from an HDF5 file.

        Parameters
        ----------
        config_file : h5py.File
            The HDF5 file containing the configuration to be loaded.
        number_of_heliostats : int
            The number of heliostats.
        number_of_surface_points_per_heliostat : int
            The number of surface points per heliostat.
        prototype_surface : SurfaceConfig
            The prototype for the surface configuration to be used if a heliostat has no individual surface.
        prototype_initial_orientation : torch.Tensor
            The prototype for the initial orientation to be used if a heliostat has no individual initial orientation.
        prototype_kinematic_deviations : torch.Tensor
            The prototype for the kinematic deviations to be used if a heliostat has no individual kinematic deviations.
        prototype_actuators : torch.Tensor
            The prototype for the actuators to be used if a heliostat has no individual actuators.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Raises
        ------
        ValueError
            If neither prototypes nor individual heliostat parameters are provided.

        Returns
        -------
        HeliostatField
            The heliostat field loaded from the HDF5 file.
        """
        device = torch.device(device)

        log.info("Loading a heliostat field from an HDF5 file.")

        all_heliostat_names = []
        all_heliostat_positions = torch.zeros((number_of_heliostats, 4), device=device)
        all_aim_points = torch.zeros((number_of_heliostats, 4), device=device)
        all_surface_points = torch.zeros(
            (number_of_heliostats, number_of_surface_points_per_heliostat, 4),
            device=device,
        )
        all_surface_normals = torch.zeros(
            (number_of_heliostats, number_of_surface_points_per_heliostat, 4),
            device=device,
        )
        all_initial_orientations = torch.zeros((number_of_heliostats, 4), device=device)

        all_kinematic_deviation_parameters = torch.zeros(
            (
                number_of_heliostats,
                config_dictionary.rigid_body_number_of_deviation_parameters,
            ),
            device=device,
        )
        all_actuator_parameters = torch.zeros(
            (
                number_of_heliostats,
                config_dictionary.number_of_linear_actuator_parameters,
                2,
            ),
            device=device,
        )

        for index, heliostat_name in enumerate(
            config_file[config_dictionary.heliostat_key].keys()
        ):
            all_heliostat_names.append(heliostat_name)

            single_heliostat_config = config_file[config_dictionary.heliostat_key][
                heliostat_name
            ]

            if (
                config_dictionary.heliostat_surface_key
                in single_heliostat_config.keys()
            ):
                surface_config = utils_load_h5.surface_config(
                    prototype=False,
                    scenario_file=single_heliostat_config,
                    device=device,
                )
            else:
                if prototype_surface is None:
                    raise ValueError(
                        "If the heliostat does not have individual surface parameters, a surface prototype must be provided!"
                    )
                log.info(
                    "Individual surface parameters not provided - loading a heliostat with the surface prototype."
                )
                surface_config = prototype_surface

            if (
                config_dictionary.heliostat_kinematic_key
                in single_heliostat_config.keys()
            ):
                initial_orientation = torch.tensor(
                    single_heliostat_config[config_dictionary.heliostat_kinematic_key][
                        config_dictionary.kinematic_initial_orientation
                    ][()],
                    dtype=torch.float,
                    device=device,
                )
                kinematic_type = single_heliostat_config[
                    config_dictionary.heliostat_kinematic_key
                ][config_dictionary.kinematic_type][()].decode("utf-8")

                kinematic_deviations, number_of_actuators = (
                    utils_load_h5.kinematic_deviations(
                        prototype=False,
                        kinematic_type=kinematic_type,
                        scenario_file=single_heliostat_config,
                        log=log,
                        heliostat_name=heliostat_name,
                        device=device,
                    )
                )
            else:
                if prototype_kinematic_deviations is None:
                    raise ValueError(
                        "If the heliostat does not have an individual kinematic, a kinematic prototype must be provided!"
                    )
                log.info(
                    "Individual kinematic configuration not provided - loading a heliostat with the kinematic prototype."
                )
                initial_orientation = prototype_initial_orientation
                kinematic_deviations = prototype_kinematic_deviations

            if (
                config_dictionary.heliostat_actuator_key
                in single_heliostat_config.keys()
            ):
                actuator_keys = list(
                    single_heliostat_config[
                        config_dictionary.heliostat_actuator_key
                    ].keys()
                )

                actuator_type = single_heliostat_config[
                    config_dictionary.heliostat_actuator_key
                ][actuator_keys[0]][config_dictionary.actuator_type_key][()].decode(
                    "utf-8"
                )

                actuator_parameters = utils_load_h5.actuator_parameters(
                    prototype=False,
                    scenario_file=single_heliostat_config,
                    actuator_type=actuator_type,
                    number_of_actuators=number_of_actuators,
                    initial_orientation=initial_orientation,
                    log=log,
                    heliostat_name=heliostat_name,
                    device=device,
                )
            else:
                if prototype_actuators is None:
                    raise ValueError(
                        "If the heliostat does not have individual actuators, an actuator prototype must be provided!"
                    )
                log.info(
                    "Individual actuator configurations not provided - loading a heliostat with the actuator prototype."
                )
                actuator_parameters = prototype_actuators

            all_heliostat_positions[index] = torch.tensor(
                single_heliostat_config[config_dictionary.heliostat_position][()],
                dtype=torch.float,
                device=device,
            )
            all_aim_points[index] = torch.tensor(
                single_heliostat_config[config_dictionary.heliostat_aim_point][()],
                dtype=torch.float,
                device=device,
            )
            surface = Surface(surface_config)
            all_surface_points[index], all_surface_normals[index] = (
                tensor.reshape(-1, 4)
                for tensor in surface.get_surface_points_and_normals(device=device)
            )
            all_initial_orientations[index] = initial_orientation
            all_kinematic_deviation_parameters[index] = kinematic_deviations
            all_actuator_parameters[index] = actuator_parameters

        return cls(
            number_of_heliostats=number_of_heliostats,
            all_heliostat_names=all_heliostat_names,
            all_heliostat_positions=all_heliostat_positions,
            all_aim_points=all_aim_points,
            all_surface_points=all_surface_points,
            all_surface_normals=all_surface_normals,
            all_initial_orientations=all_initial_orientations,
            all_kinematic_deviation_parameters=all_kinematic_deviation_parameters,
            all_actuator_parameters=all_actuator_parameters,
            device=device,
        )

    def align_surfaces_with_incident_ray_direction(
        self,
        incident_ray_direction: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """
        Align all surface points and surface normals of all heliostats in the field.

        This method uses the incident ray direction to align the heliostats.

        Parameters
        ----------
        incident_ray_direction : torch.Tensor
            The incident ray direction.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).
        """
        device = torch.device(device)

        rigid_body_kinematic = RigidBody(
            number_of_heliostats=self.number_of_heliostats,
            heliostat_positions=self.all_heliostat_positions,
            aim_points=self.all_aim_points,
            actuator_parameters=self.all_actuator_parameters,
            initial_orientations=self.all_initial_orientations,
            deviation_parameters=self.all_kinematic_deviation_parameters,
            device=device,
        )
        (
            self.all_current_aligned_surface_points,
            self.all_current_aligned_surface_normals,
        ) = rigid_body_kinematic.align_surfaces_with_incident_ray_direction(
            incident_ray_direction=incident_ray_direction,
            surface_points=self.all_surface_points,
            surface_normals=self.all_surface_normals,
            device=device,
        )
        # Note that heliostats have been aligned
        self.all_aligned_heliostats = torch.ones_like(self.all_aligned_heliostats)

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

        rigid_body_kinematic = RigidBody(
            number_of_heliostats=self.number_of_heliostats,
            heliostat_positions=self.all_heliostat_positions,
            aim_points=self.all_aim_points,
            actuator_parameters=self.all_actuator_parameters,
            initial_orientations=self.all_initial_orientations,
            deviation_parameters=self.all_kinematic_deviation_parameters,
            device=device,
        )
        return rigid_body_kinematic.motor_positions_to_orientations(
            motor_positions, device
        )

    def align_surfaces_with_motor_positions(
        self,
        motor_positions: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """
        Align all surface points and surface normals of all heliostats in the field.

        This method uses the motor positions to align the heliostats.

        Parameters
        ----------
        motor_positions : torch.Tensor
            The motor positions.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).
        """
        device = torch.device(device)

        rigid_body_kinematic = RigidBody(
            number_of_heliostats=self.number_of_heliostats,
            heliostat_positions=self.all_heliostat_positions,
            aim_points=self.all_aim_points,
            actuator_parameters=self.all_actuator_parameters,
            initial_orientations=self.all_initial_orientations,
            deviation_parameters=self.all_kinematic_deviation_parameters,
            device=device,
        )
        (
            self.all_current_aligned_surface_points,
            self.all_current_aligned_surface_normals,
        ) = rigid_body_kinematic.align_surfaces_with_motor_positions(
            motor_positions=motor_positions,
            surface_points=self.all_surface_points,
            surface_normals=self.all_surface_normals,
            device=device,
        )
        # Note that heliostats have been aligned
        self.all_aligned_heliostats = torch.ones_like(self.all_aligned_heliostats)

    def forward(self) -> None:
        """
        Specify the forward pass.

        Raises
        ------
        NotImplementedError
            Whenever called.
        """
        raise NotImplementedError("Not Implemented!")
