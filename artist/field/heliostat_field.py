import logging
from typing import Union

from artist.field import actuator
from artist.field.kinematic_rigid_body import RigidBody
from artist.field.surface import Surface
import h5py
from torch import device
import torch.nn
from typing_extensions import Self

from artist.field.heliostat import Heliostat
from artist.util import config_dictionary, utils_load_h5, utils
from artist.util.configuration_classes import (
    ActuatorListConfig,
    FacetConfig,
    KinematicLoadConfig,
    SurfaceConfig,
)

log = logging.getLogger(__name__)
"""A logger for the heliostat field."""


class HeliostatField(torch.nn.Module):
    """
    Wrap the heliostat list as a ``torch.nn.Module`` to allow gradient calculation.

    Attributes
    ----------
    heliostat_list : list[Heliostat]
        A list of heliostats included in the scenario.

    Methods
    -------
    from_hdf5()
        Load the list of heliostats from an HDF5 file.
    forward()
        Specify the forward pass.
    """

    def __init__(self, 
                 number_of_heliostats,
                 all_heliostat_names,
                 all_heliostat_positions,
                 all_aim_points,
                 all_surface_points,
                 all_surface_normals,
                 all_initial_orientations,
                 all_kinematic_deviation_parameters,
                 all_actuator_parameters,
                 device: Union[torch.device, str] = "cuda"):
        """
        Initialize the heliostat field.

        A heliostat field consists of many heliostats that have a unique position in the field. The
        heliostats in the field are aligned individually to reflect the incoming light in a way that
        ensures maximum efficiency for the whole power plant.

        Parameters
        ----------
        heliostat_list : list[Heliostat]
            The list of heliostats included in the scenario.
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

        self.all_aligned_heliostats = torch.zeros(all_surface_points.shape[0], device=device)
        self.all_preferred_reflection_directions = torch.zeros((all_surface_points.shape[0], 4), device=device)
        self.all_current_aligned_surface_points = torch.zeros_like(all_surface_points, device=device)
        self.all_current_aligned_surface_normals = torch.zeros_like(all_surface_normals, device=device)
       

    @classmethod
    def from_hdf5(
        cls,
        config_file: h5py.File,
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
        prototype_surface : SurfaceConfig
            The prototype for the surface configuration to be used if the heliostat has no individual surface.
        prototype_initial_orientation : torch.Tensor
            The prototype for the initial orientation to be used if the heliostat has no individual initial orientation.
        prototype_kinematic_deviations : torch.Tensor
            The prototype for the kinematic deviations to be used if the heliostat has no individual kinematic deviations.
        prototype_actuators : torch.Tensor
            The prototype for the actuators to be used if the heliostat has no individual actuators.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        HeliostatField
            The heliostat field loaded from the HDF5 file.
        """
        log.info("Loading a heliostat field from an HDF5 file.")
        device = torch.device(device)

        number_of_heliostats = len(config_file[config_dictionary.heliostat_key])

        # TODO 10000 ersetzen
        number_of_surface_points_per_heliostat = 10000
        all_heliostat_names = []
        all_heliostat_positions = torch.zeros((number_of_heliostats, 4), device=device)
        all_aim_points = torch.zeros((number_of_heliostats, 4), device=device)
        all_surface_points = torch.zeros((number_of_heliostats, number_of_surface_points_per_heliostat, 4), device=device)
        all_surface_normals = torch.zeros((number_of_heliostats, number_of_surface_points_per_heliostat, 4), device=device)
        all_initial_orientations = torch.zeros((number_of_heliostats, 4), device=device)
        
        # TODO unterschiedliche Parameter Längen
        all_kinematic_deviation_parameters = torch.zeros((number_of_heliostats, 18), device=device)
        all_actuator_parameters = torch.zeros((number_of_heliostats, 7, 2), device=device)

        for index, heliostat_name in enumerate(config_file[config_dictionary.heliostat_key].keys()):
            all_heliostat_names.append(heliostat_name)
            
            single_heliostat_config = config_file[config_dictionary.heliostat_key][heliostat_name]

            all_heliostat_positions[index] = torch.tensor(single_heliostat_config[config_dictionary.heliostat_position][()], dtype=torch.float, device=device)
            all_aim_points[index] = torch.tensor(single_heliostat_config[config_dictionary.heliostat_aim_point][()], dtype=torch.float, device=device)

            if config_dictionary.heliostat_surface_key in single_heliostat_config.keys():
                surface_config = utils_load_h5.surface_config(
                    prototype=False,
                    scenario_file=single_heliostat_config,
                    device=device)
            else:
                if prototype_surface is None:
                    raise ValueError(
                        "If the heliostat does not have individual surface parameters, a surface prototype must be provided!"
                    )
                log.info(
                    "Individual surface parameters not provided - loading a heliostat with the surface prototype."
                )
                surface_config = prototype_surface
            surface = Surface(surface_config)
            all_surface_points[index], all_surface_normals[index] = (tensor.reshape(-1, 4) for tensor in surface.get_surface_points_and_normals(device=device))

            if config_dictionary.heliostat_kinematic_key in single_heliostat_config.keys():
                initial_orientation = torch.tensor(single_heliostat_config[config_dictionary.heliostat_kinematic_key][config_dictionary.kinematic_initial_orientation][()], dtype=torch.float, device=device)
                kinematic_type = single_heliostat_config[config_dictionary.heliostat_kinematic_key][
                    config_dictionary.kinematic_type
                ][()].decode("utf-8")

                kinematic_deviations, number_of_actuators = utils_load_h5.kinematic_deviations(
                    prototype=False,
                    kinematic_type=kinematic_type,
                    scenario_file=single_heliostat_config,
                    log=log,
                    heliostat_name=heliostat_name,
                    device=device
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
            
            all_initial_orientations[index] = initial_orientation
            all_kinematic_deviation_parameters[index] = kinematic_deviations

            if config_dictionary.heliostat_actuator_key in single_heliostat_config.keys():
                actuator_keys = list(single_heliostat_config[config_dictionary.heliostat_actuator_key].keys())

                actuator_type = single_heliostat_config[config_dictionary.heliostat_actuator_key][actuator_keys[0]][config_dictionary.actuator_type_key][()].decode("utf-8")

                actuator_parameters = utils_load_h5.actuator_parameters(
                    prototype=False,
                    scenario_file=single_heliostat_config,
                    actuator_type=actuator_type,
                    number_of_actuators=number_of_actuators,
                    log=log,
                    heliostat_name=heliostat_name, 
                    device=device)
            else:
                if prototype_actuators is None:
                    raise ValueError(
                        "If the heliostat does not have individual actuators, an actuator prototype must be provided!"
                    )
                log.info(
                    "Individual actuator configurations not provided - loading a heliostat with the actuator prototype."
                )
                actuator_parameters = prototype_actuators

            # Adapt initial angle of actuator one according to kinematic initial orientation.
            # ARTIST always expects heliostats to be initially oriented to the south [0.0, -1.0, 0.0] (in ENU).
            # The first actuator always rotates along the east-axis.
            # Since the actuator coordinate system is relative to the heliostat orientation, the initial angle
            # of actuator one needs to be transformed accordingly.
            actuator_parameters[6, 0] = utils.transform_initial_angle(
                initial_angle=actuator_parameters[6, 0].unsqueeze(0),
                initial_orientation=initial_orientation,
                device=device,
            )

            all_actuator_parameters[index] = actuator_parameters

        return cls(number_of_heliostats=number_of_heliostats,
                   all_heliostat_names=all_heliostat_names,
                   all_heliostat_positions=all_heliostat_positions,
                   all_aim_points=all_aim_points,
                   all_surface_points=all_surface_points,
                   all_surface_normals=all_surface_normals,
                   all_initial_orientations=all_initial_orientations,
                   all_kinematic_deviation_parameters=all_kinematic_deviation_parameters,
                   all_actuator_parameters=all_actuator_parameters,
                   device=device)

    
    def align_surfaces_with_incident_ray_direction(
        self,
        incident_ray_direction: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """
        Compute the aligned surface points and aligned surface normals of the heliostat.

        This method uses the incident ray direction to align the heliostat.

        Parameters
        ----------
        incident_ray_direction : torch.Tensor
            The incident ray direction.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).
        """
        rigid_body_kinematic = RigidBody(
            heliostat_positions=self.all_heliostat_positions,
            aim_points=self.all_aim_points,
            actuator_parameters=self.all_actuator_parameters,
            initial_orientations=self.all_initial_orientations,
            deviation_parameters=self.all_kinematic_deviation_parameters,
            device=device
        )
        device = torch.device(device)
        (
            self.all_current_aligned_surface_points,
            self.all_current_aligned_surface_normals,
        ) = rigid_body_kinematic.align_surface_with_incident_ray_direction(
            incident_ray_direction, self.all_surface_points, self.all_surface_normals, device
        )
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
