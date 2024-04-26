import logging
import sys
from collections.abc import MutableMapping
from typing import Any, Dict, List, Optional

import colorlog
import h5py
import torch

from artist.util import config_dictionary


class ReceiverConfig:
    """
    Contains the receiver configuration parameters.

    Attributes
    ----------
    receiver_key : str
        The ID string used to identify the receiver in the HDF5 file.
    receiver_type : str
        The type of receiver, e.g. planar.
    position_center : torch.Tensor
        The position of the center of the receiver.
    normal_vector : torch.Tensor
        The normal vector to the receiver plane.
    plane_e : float
        The size of the receiver in the east direction.
    plane_u : float
        The size of the receiver in the up direction.
    resolution_e : int
        The resolution of the receiver in the east direction.
    resolution_u : int
        The resolution of the receiver in the up direction.
    curvature_e: Optional[float]
        The curvature of the receiver in the east direction.
    curvature_u: Optional[float]
        The curvature of the receiver in the up direction.

    Methods
    -------
    create_receiver_dict()
       Create a dictionary containing the configuration parameters for the receiver.
    """

    def __init__(
        self,
        receiver_key: str,
        receiver_type: str,
        position_center: torch.Tensor,
        normal_vector: torch.Tensor,
        plane_e: float,
        plane_u: float,
        resolution_e: int,
        resolution_u: int,
        curvature_e: Optional[float] = None,
        curvature_u: Optional[float] = None,
    ) -> None:
        """
        Initialize the receiver configuration.

        Parameters
        ----------
        receiver_key : str
            The key used to identify the receiver in the HDF5 file.
        receiver_type : str
            The type of receiver, e.g. planar.
        position_center : torch.Tensor
            The position of the center of the receiver.
        normal_vector : torch.Tensor
            The normal vector to the receiver plane.
        plane_e : float
            The size of the receiver in the east direction.
        plane_u : float
            The size of the receiver in the up direction.
        resolution_e : int
            The resolution of the receiver in the east direction.
        resolution_u : int
            The resolution of the receiver in the up direction.
        curvature_e: Optional[float]
            The curvature of the receiver in the east direction.
        curvature_u: Optional[float]
            The curvature of the receiver in the up direction.
        """
        self.receiver_key = receiver_key
        self.receiver_type = receiver_type
        self.position_center = position_center
        self.normal_vector = normal_vector
        self.plane_e = plane_e
        self.plane_u = plane_u
        self.resolution_e = resolution_e
        self.resolution_u = resolution_u
        self.curvature_e = curvature_e
        self.curvature_u = curvature_u

    def create_receiver_dict(self) -> Dict[str, Any]:
        """
        Create a dictionary containing the configuration parameters for the receiver.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the configuration parameters for the receiver.
        """
        receiver_dict = {
            config_dictionary.receiver_type: self.receiver_type,
            config_dictionary.receiver_position_center: self.position_center,
            config_dictionary.receiver_normal_vector: self.normal_vector,
            config_dictionary.receiver_plane_e: self.plane_e,
            config_dictionary.receiver_plane_u: self.plane_u,
            config_dictionary.receiver_resolution_e: self.resolution_e,
            config_dictionary.receiver_resolution_u: self.resolution_u,
        }
        if self.curvature_e is not None:
            receiver_dict.update(
                {config_dictionary.receiver_curvature_e: self.curvature_e}
            )
        if self.curvature_u is not None:
            receiver_dict.update(
                {config_dictionary.receiver_curvature_u: self.curvature_u}
            )

        return receiver_dict


class ReceiverListConfig:
    """
    Contains the receiver list configuration parameters.

    Attributes
    ----------
    receiver_list : List[ReceiverConfig]
        A list of receiver configurations to be included in the scenario.

    Methods
    -------
    create_receiver_list_dict()
       Create a dictionary containing the configuration parameters for the list of receivers.
    """

    def __init__(self, receiver_list: List[ReceiverConfig]) -> None:
        """
        Initialize the receiver list configuration.

        Parameters
        ----------
        receiver_list : List[ReceiverConfig]
            The list of receiver configurations included in the scenario.
        """
        self.receiver_list = receiver_list

    def create_receiver_list_dict(self) -> Dict[str, Any]:
        """
        Create a dictionary containing the configuration parameters for the list of receivers.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the configuration parameters for the list of receivers.
        """
        return {
            receiver.receiver_key: receiver.create_receiver_dict()
            for receiver in self.receiver_list
        }


class LightSourceConfig:
    """
    Contains the light source configuration parameters.

    Attributes
    ----------
    light_source_key : str
        The ID string used to identify the light source in the HDF5 file.
    light_source_type:
        The type of light source used, e.g. a sun.
    number_of_rays : int
        The number of rays generated by the light source.
    distribution_type : str
        The distribution type used to model the light source.
    mean : Optional[float]
        The mean used for modeling the sun.
    covariance : Optional[float]
        The covariance used for modeling the sun.

    Methods
    -------
    create_light_source_dict()
        Create a dictionary containing the configuration parameters for the light source.
    """

    def __init__(
        self,
        light_source_key: str,
        light_source_type: str,
        number_of_rays: int,
        distribution_type: str,
        mean: Optional[float],
        covariance: Optional[float],
    ) -> None:
        """
        Initialize the light source configuration.

        Parameters
        ----------
        light_source_key : str
            The key used to identify the light source in the HDF5 file.
        light_source_type:
            The type of light source used, e.g. a sun.
        number_of_rays : int
            The number of rays generated by the light source.
        distribution_type : str
            The distribution type used to model the light source.
        mean : Optional[float]
            The mean used for modeling the sun.
        covariance : Optional[float]
            The covariance used for modeling the sun.
        """
        self.light_source_key = light_source_key
        self.light_source_type = light_source_type
        self.number_of_rays = number_of_rays
        self.distribution_type = distribution_type
        assert (
            self.distribution_type
            == config_dictionary.light_source_distribution_is_normal
        ), "Unknown light source distribution type."

        if (
            self.distribution_type
            == config_dictionary.light_source_distribution_is_normal
        ):
            self.mean = mean
            self.covariance = covariance

    def create_light_source_dict(self) -> Dict[str, Any]:
        """
        Create a dictionary containing the configuration parameters for the light source.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the configuration parameters for the light source.
        """
        # Check if the distribution type is implemented.
        if (
            self.distribution_type
            == config_dictionary.light_source_distribution_is_normal
        ):
            light_source_distribution_parameters_dict = {
                config_dictionary.light_source_distribution_type: str(
                    self.distribution_type
                ),
                config_dictionary.light_source_mean: self.mean,
                config_dictionary.light_source_covariance: self.covariance,
            }
        else:
            raise NotImplementedError("Unknown light source distribution type.")

        # Return the desired dictionary.
        return {
            config_dictionary.light_source_distribution_type: self.light_source_type,
            config_dictionary.light_source_number_of_rays: self.number_of_rays,
            config_dictionary.light_source_distribution_parameters: light_source_distribution_parameters_dict,
        }


class LightSourceListConfig:
    """
    Contains the light source list configuration parameters.

    Attributes
    ----------
    light_source_list : List[LightSourceConfig]
        The list of light source configs to be included in the scenario.

    Methods
    -------
    create_light_list_dict()
       Create a dictionary containing the configuration parameters for the light source list.
    """

    def __init__(self, light_source_list: List[LightSourceConfig]):
        """
        Initialize the light source list configuration.

        Parameters
        ----------
        light_source_list : List[LightSourceConfig]
            The list of light source configs to be included in the scenario.
        """
        self.light_source_list = light_source_list

    def create_light_source_list_dict(self):
        """
        Create a dictionary containing the configuration parameters for the light source list.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the configuration parameters for the light source list.
        """
        return {
            ls.light_source_key: ls.create_light_source_dict()
            for ls in self.light_source_list
        }


class FacetConfig:
    """
    Contains the facet configuration parameters.

    Attributes
    ----------
    facet_key : str
        The ID of the facet used to identify the facet in the HDF5 file.
    control_points_e : torch.Tensor
        The NURBS control points in the east direction.
    control_points_u : torch.Tensor
        The NURBS control points in the up direction.
    knots_e : torch.Tensor
        The NURBS knots in the east direction.
    knots_u : torch.Tensor
        The NURBS knots in the up direction.
    width : float
        The width of the facet.
    height : float
        The height of the facet.
    position : torch.Tensor
        The position of the facet.
    canting_e : torch.Tensor
        The canting vector in the east direction.
    canting_u : torch.Tensor
        The canting vector in the up direction.

    Methods
    -------
    create_facet_dict()
       Create a dictionary containing the configuration parameters for a facet.
    """

    def __init__(
        self,
        facet_key: str,
        control_points_e: torch.Tensor,
        control_points_u: torch.Tensor,
        knots_e: torch.Tensor,
        knots_u: torch.Tensor,
        width: float,
        height: float,
        position: torch.Tensor,
        canting_e: torch.Tensor,
        canting_u: torch.Tensor,
    ) -> None:
        """
        Initialize the facet configuration.

        Parameters
        ----------
        facet_key : str
            The key used to identify the facet in the HDF5 file.
        control_points_e : torch.Tensor
            The NURBS control points in the east direction.
        control_points_u : torch.Tensor
            The NURBS control points in the up direction.
        knots_e : torch.Tensor
            The NURBS knots in the east direction.
        knots_u : torch.Tensor
            The NURBS knots in the up direction.
        width : float
            The width of the facet.
        height : float
            The height of the facet.
        position : torch.Tensor
            The position of the facet.
        canting_e : torch.Tensor
            The canting vector in the east direction.
        canting_u : torch.Tensor
            The canting vector in the up direction.
        """
        self.facet_key = facet_key
        self.control_points_e = control_points_e
        self.control_points_u = control_points_u
        self.knots_e = knots_e
        self.knots_u = knots_u
        self.width = width
        self.height = height
        self.position = position
        self.canting_e = canting_e
        self.canting_u = canting_u

    def create_facet_dict(self) -> Dict[str, Any]:
        """
        Create a dictionary containing the configuration parameters for a facet..

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the configuration parameters for the facet.
        """
        return {
            config_dictionary.facet_control_points_e: self.control_points_e,
            config_dictionary.facet_control_points_u: self.control_points_u,
            config_dictionary.facet_knots_e: self.knots_e,
            config_dictionary.facet_knots_u: self.knots_u,
            config_dictionary.facets_width: self.width,
            config_dictionary.facets_height: self.height,
            config_dictionary.facets_position: self.position,
        }


class SurfaceConfig:
    """
    Contains the surface configuration parameters.

    Attributes
    ----------
    facets_list : List[FacetsConfiguration]
        The list of facets to be used for the surface of the heliostat.

    Methods
    -------
    create_surface_dict()
       Create a dictionary containing the configuration parameters for the surface.
    """

    def __init__(self, facets_list: List[FacetConfig]) -> None:
        """
        Initialize the surface configuration.

        Parameters
        ----------
        facets_list : List[FacetsConfig]
            The list of facets to be used for the surface of the heliostat.
        """
        self.facets_list = facets_list

    def create_surface_dict(self) -> Dict[str, Any]:
        """
        Create a dictionary containing the configuration parameters for the surface.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the configuration parameters for the surface.
        """
        facets_dict = {
            facet.facet_key: facet.create_facet_dict() for facet in self.facets_list
        }
        return {config_dictionary.facets_key: facets_dict}


class SurfacePrototypeConfig(SurfaceConfig):
    """
    Contains the configuration parameters for a surface prototype.

    See Also
    --------
    :class:`SurfaceConfig` : Reference to the parent class.
    """

    def __init__(self, facets_list: List[FacetConfig]) -> None:
        """
        Initialize the surface prototype configuration.

        Parameters
        ----------
        facets_list : List[FacetsConfig]
            The list of facets to be used for the surface of the heliostat prototype.
        """
        super().__init__(facets_list=facets_list)


class KinematicDeviations:
    """
    Contains the kinematic deviations.

    Attributes
    ----------
    first_joint_translation_e : Optional[float]
        The first joint translation in the east direction.
    first_joint_translation_n : Optional[float]
        The first joint translation in the north direction.
    first_joint_translation_u : Optional[float]
        The first joint translation in the up direction.
    first_joint_tilt_e : Optional[float]
        The first joint tilt in the east direction.
    first_joint_tilt_n : Optional[float]
        The first joint tilt in the north direction.
    first_joint_tilt_u : Optional[float]
        The first joint tilt in the up direction.
    second_joint_translation_e : Optional[float]
        The second joint translation in the east direction.
    second_joint_translation_n : Optional[float]
        The second joint translation in the north direction.
    second_joint_translation_u : Optional[float]
        The second joint translation in the up direction.
    second_joint_tilt_e : Optional[float]
        The second joint tilt in the east direction.
    second_joint_tilt_n : Optional[float]
        The second joint tilt in the north direction.
    second_joint_tilt_u : Optional[float]
        The second joint tilt in the up direction.
    concentrator_translation_e : Optional[float]
        The concentrator translation in the east direction.
    concentrator_translation_n : Optional[float]
        The concentrator translation in the north direction.
    concentrator_translation_u : Optional[float]
        The concentrator translation in the up direction.
    concentrator_tilt_e : Optional[float]
        The concentrator tilt in the east direction.
    concentrator_tilt_n : Optional[float]
        The concentrator tilt in the north direction.
    concentrator_tilt_u : Optional[float]
        The concentrator tilt in the up direction.

    Methods
    -------
    create_kinematic_deviations_dict()
        Create a dictionary containing the configuration parameters for the kinematic deviations.
    """

    def __init__(
        self,
        first_joint_translation_e: Optional[float] = None,
        first_joint_translation_n: Optional[float] = None,
        first_joint_translation_u: Optional[float] = None,
        first_joint_tilt_e: Optional[float] = None,
        first_joint_tilt_n: Optional[float] = None,
        first_joint_tilt_u: Optional[float] = None,
        second_joint_translation_e: Optional[float] = None,
        second_joint_translation_n: Optional[float] = None,
        second_joint_translation_u: Optional[float] = None,
        second_joint_tilt_e: Optional[float] = None,
        second_joint_tilt_n: Optional[float] = None,
        second_joint_tilt_u: Optional[float] = None,
        concentrator_translation_e: Optional[float] = None,
        concentrator_translation_n: Optional[float] = None,
        concentrator_translation_u: Optional[float] = None,
        concentrator_tilt_e: Optional[float] = None,
        concentrator_tilt_n: Optional[float] = None,
        concentrator_tilt_u: Optional[float] = None,
    ) -> None:
        """
        Initialize the kinematic deviations.

        Parameters
        ----------
        first_joint_translation_e : Optional[float]
            The first joint translation in the east direction.
        first_joint_translation_n : Optional[float]
            The first joint translation in the north direction.
        first_joint_translation_u : Optional[float]
            The first joint translation in the up direction.
        first_joint_tilt_e : Optional[float]
            The first joint tilt in the east direction.
        first_joint_tilt_n : Optional[float]
            The first joint tilt in the north direction.
        first_joint_tilt_u : Optional[float]
            The first joint tilt in the up direction.
        second_joint_translation_e : Optional[float]
            The second joint translation in the east direction.
        second_joint_translation_n : Optional[float]
            The second joint translation in the north direction.
        second_joint_translation_u : Optional[float]
            The second joint translation in the up direction.
        second_joint_tilt_e : Optional[float]
            The second joint tilt in the east direction.
        second_joint_tilt_n : Optional[float]
            The second joint tilt in the north direction.
        second_joint_tilt_u : Optional[float]
            The second joint tilt in the up direction.
        concentrator_translation_e : Optional[float]
            The concentrator translation in the east direction.
        concentrator_translation_n : Optional[float]
            The concentrator translation in the north direction.
        concentrator_translation_u : Optional[float]
            The concentrator translation in the up direction.
        concentrator_tilt_e : Optional[float]
            The concentrator tilt in the east direction.
        concentrator_tilt_n : Optional[float]
            The concentrator tilt in the north direction.
        concentrator_tilt_u : Optional[float]
            The concentrator tilt in the up direction.
        """
        self.first_joint_translation_e = first_joint_translation_e
        self.first_joint_translation_n = first_joint_translation_n
        self.first_joint_translation_u = first_joint_translation_u
        self.first_joint_tilt_e = first_joint_tilt_e
        self.first_joint_tilt_n = first_joint_tilt_n
        self.first_joint_tilt_u = first_joint_tilt_u
        self.second_joint_translation_e = second_joint_translation_e
        self.second_joint_translation_n = second_joint_translation_n
        self.second_joint_translation_u = second_joint_translation_u
        self.second_joint_tilt_e = second_joint_tilt_e
        self.second_joint_tilt_n = second_joint_tilt_n
        self.second_joint_tilt_u = second_joint_tilt_u
        self.concentrator_translation_e = concentrator_translation_e
        self.concentrator_translation_n = concentrator_translation_n
        self.concentrator_translation_u = concentrator_translation_u
        self.concentrator_tilt_e = concentrator_tilt_e
        self.concentrator_tilt_n = concentrator_tilt_n
        self.concentrator_tilt_u = concentrator_tilt_u

    def create_kinematic_deviations_dict(self) -> Dict[str, float]:
        """
        Create a dictionary containing the configuration parameters for the kinematic deviations.

        Returns
        -------
        Dict[str, float]
            A dictionary containing the configuration parameters for the kinematic deviations.
        """
        deviations_dict = {}
        if self.first_joint_translation_e is not None:
            deviations_dict.update(
                {
                    config_dictionary.first_joint_translation_e: self.first_joint_translation_e
                }
            )
        if self.first_joint_translation_n is not None:
            deviations_dict.update(
                {
                    config_dictionary.first_joint_translation_n: self.first_joint_translation_n
                }
            )
        if self.first_joint_translation_u is not None:
            deviations_dict.update(
                {
                    config_dictionary.first_joint_translation_u: self.first_joint_translation_u
                }
            )
        if self.first_joint_tilt_e is not None:
            deviations_dict.update(
                {config_dictionary.first_joint_tilt_e: self.first_joint_tilt_e}
            )
        if self.first_joint_tilt_n is not None:
            deviations_dict.update(
                {config_dictionary.first_joint_tilt_n: self.first_joint_tilt_n}
            )
        if self.first_joint_tilt_u is not None:
            deviations_dict.update(
                {config_dictionary.first_joint_tilt_u: self.first_joint_tilt_u}
            )
        if self.second_joint_translation_e is not None:
            deviations_dict.update(
                {
                    config_dictionary.second_joint_translation_e: self.second_joint_translation_e
                }
            )
        if self.second_joint_translation_n is not None:
            deviations_dict.update(
                {
                    config_dictionary.second_joint_translation_n: self.second_joint_translation_n
                }
            )
        if self.second_joint_translation_u is not None:
            deviations_dict.update(
                {
                    config_dictionary.second_joint_translation_u: self.second_joint_translation_u
                }
            )
        if self.second_joint_tilt_e is not None:
            deviations_dict.update(
                {config_dictionary.second_joint_tilt_e: self.second_joint_tilt_e}
            )
        if self.second_joint_tilt_n is not None:
            deviations_dict.update(
                {config_dictionary.second_joint_tilt_n: self.second_joint_tilt_n}
            )
        if self.second_joint_tilt_u is not None:
            deviations_dict.update(
                {config_dictionary.second_joint_tilt_u: self.second_joint_tilt_u}
            )
        if self.concentrator_translation_e is not None:
            deviations_dict.update(
                {
                    config_dictionary.concentrator_translation_e: self.concentrator_translation_e
                }
            )
        if self.concentrator_translation_n is not None:
            deviations_dict.update(
                {
                    config_dictionary.concentrator_translation_n: self.concentrator_translation_n
                }
            )
        if self.concentrator_translation_u is not None:
            deviations_dict.update(
                {
                    config_dictionary.concentrator_translation_u: self.concentrator_translation_u
                }
            )
        if self.concentrator_tilt_e is not None:
            deviations_dict.update(
                {config_dictionary.concentrator_tilt_e: self.concentrator_tilt_e}
            )
        if self.concentrator_tilt_n is not None:
            deviations_dict.update(
                {config_dictionary.concentrator_tilt_n: self.concentrator_tilt_n}
            )
        if self.concentrator_tilt_u is not None:
            deviations_dict.update(
                {config_dictionary.concentrator_tilt_u: self.concentrator_tilt_u}
            )
        return deviations_dict


class KinematicOffsets:
    """
    Contains the kinematic offsets.

    Attributes
    ----------
    kinematic_initial_orientation_offset_e : Optional[float]
        The initial orientation offset in the east direction.
    kinematic_initial_orientation_offset_n : Optional[float]
        The initial orientation offset in the north direction.
    kinematic_initial_orientation_offset_u : Optional[float]
        The initial orientation offset in the up direction.

    Methods
    -------
    create_kinematic_offsets_dict()
        Create a dictionary containing the configuration parameters for the kinematic offsets.
    """

    def __init__(
        self,
        kinematic_initial_orientation_offset_e: Optional[float] = None,
        kinematic_initial_orientation_offset_n: Optional[float] = None,
        kinematic_initial_orientation_offset_u: Optional[float] = None,
    ) -> None:
        """
        Initialize the initial orientation offsets.

        Parameters
        ----------
        kinematic_initial_orientation_offset_e : Optional[float]
            The initial orientation offset in the east direction.
        kinematic_initial_orientation_offset_n : Optional[float]
            The initial orientation offset in the north direction.
        kinematic_initial_orientation_offset_u : Optional[float]
            The initial orientation offset in the up direction.
        """
        self.kinematic_initial_orientation_offset_e = (
            kinematic_initial_orientation_offset_e
        )
        self.kinematic_initial_orientation_offset_n = (
            kinematic_initial_orientation_offset_n
        )
        self.kinematic_initial_orientation_offset_u = (
            kinematic_initial_orientation_offset_u
        )

    def create_kinematic_offsets_dict(self) -> Dict[str, float]:
        """
        Create a dictionary containing the configuration parameters for the kinematic offsets.

        Returns
        -------
        Dict[str, float]
            A dictionary containing the configuration parameters for the kinematic offsets.
        """
        offset_dict = {}
        if self.kinematic_initial_orientation_offset_e is not None:
            offset_dict.update(
                {
                    config_dictionary.kinematic_initial_orientation_offset_e: self.kinematic_initial_orientation_offset_e
                }
            )
        if self.kinematic_initial_orientation_offset_n is not None:
            offset_dict.update(
                {
                    config_dictionary.kinematic_initial_orientation_offset_n: self.kinematic_initial_orientation_offset_n
                }
            )
        if self.kinematic_initial_orientation_offset_u is not None:
            offset_dict.update(
                {
                    config_dictionary.kinematic_initial_orientation_offset_u: self.kinematic_initial_orientation_offset_u
                }
            )
        return offset_dict


class KinematicConfig:
    """
    Contains the configuration parameters for the kinematic.

    Attributes
    ----------
    kinematic_type : str
        The type of kinematic used.
    kinematic_initial_orientation_offsets : Optional[KinematicOffsets]
        The initial orientation offsets of the kinematic configuration.
    kinematic_deviations : Optional[KinematicDeviations]
        The kinematic deviations.

    Methods
    -------
    create_kinematic_dict()
        Create a dictionary containing the configuration parameters for the kinematic.
    """

    def __init__(
        self,
        kinematic_type: str,
        kinematic_initial_orientation_offsets: Optional[KinematicOffsets] = None,
        kinematic_deviations: Optional[KinematicDeviations] = None,
    ) -> None:
        """
        Initialize the kinematic configuration.

        Parameters
        ----------
        kinematic_type : str
            The type of kinematic used.
        kinematic_initial_orientation_offsets : Optional[KinematicOffsets]
            The initial orientation offsets of the kinematic configuration.
        kinematic_deviations : Optional[KinematicDeviations]
            The kinematic deviations.
        """
        self.kinematic_type = kinematic_type
        self.kinematic_initial_orientation_offsets = (
            kinematic_initial_orientation_offsets
        )

        self.kinematic_deviations = kinematic_deviations

    def create_kinematic_dict(self) -> Dict[str, Any]:
        """
        Create a dictionary containing the configuration parameters for the kinematic.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the configuration parameters for the kinematic.
        """
        kinematic_dict = {config_dictionary.kinematic_type: self.kinematic_type}
        if self.kinematic_initial_orientation_offsets is not None:
            kinematic_dict.update(
                {
                    config_dictionary.kinematic_offsets_key: self.kinematic_initial_orientation_offsets.create_kinematic_offsets_dict()
                }
            )
        if self.kinematic_deviations is not None:
            kinematic_dict.update(
                {
                    config_dictionary.kinematic_deviations_key: self.kinematic_deviations.create_kinematic_deviations_dict()
                }
            )
        return kinematic_dict


class KinematicPrototypeConfig(KinematicConfig):
    """
    Contains the configuration parameters for the kinematic prototype.

    See Also
    --------
    :class:`KinematicConfig` : Reference to the parent class.
    """

    def __init__(
        self,
        kinematic_type: str,
        kinematic_initial_orientation_offsets: Optional[KinematicOffsets] = None,
        kinematic_deviations: Optional[KinematicDeviations] = None,
    ) -> None:
        """
        Initialize the kinematic prototype configuration.

        Parameters
        ----------
        kinematic_type : str
            The type of kinematic used.
        kinematic_initial_orientation_offsets : Optional[KinematicOffsets]
            The initial orientation offsets of the kinematic configuration.
        kinematic_deviations : Optional[KinematicDeviations]
            The kinematic deviations.
        """
        super().__init__(
            kinematic_type=kinematic_type,
            kinematic_initial_orientation_offsets=kinematic_initial_orientation_offsets,
            kinematic_deviations=kinematic_deviations,
        )


class ActuatorParameters:
    """
    Contains the actuator parameters.

    Attributes
    ----------
    first_joint_increment : Optional[float]
        The increment for the first joint.
    first_joint_initial_stroke_length : Optional[float]
        The initial stroke length for the first joint.
    first_joint_actuator_offset : Optional[float]
        The initial actuator offset for the first joint.
    first_joint_radius : Optional[float]
        The radius of the first joint.
    first_joint_phi_0 : Optional[float]
        The initial phi value of the first joint.
    second_joint_increment : Optional[float]
        The increment for the second joint.
    second_joint_initial_stroke_length : Optional[float]
        The initial stroke length for the second joint.
    second_joint_actuator_offset : Optional[float]
        The initial actuator offset for the second joint.
    second_joint_radius : Optional[float]
        The radius for the second joint.
    second_joint_phi_0 : Optional[float]
        The initial phi value of the second joint.

    Methods
    -------
    create_actuator_parameters_dict()
        Create a dictionary containing the configuration parameters for the actuator parameters.
    """

    def __init__(
        self,
        first_joint_increment: Optional[float] = None,
        first_joint_initial_stroke_length: Optional[float] = None,
        first_joint_actuator_offset: Optional[float] = None,
        first_joint_radius: Optional[float] = None,
        first_joint_phi_0: Optional[float] = None,
        second_joint_increment: Optional[float] = None,
        second_joint_initial_stroke_length: Optional[float] = None,
        second_joint_actuator_offset: Optional[float] = None,
        second_joint_radius: Optional[float] = None,
        second_joint_phi_0: Optional[float] = None,
    ) -> None:
        """
        Initialize the actuator parameters.

        Parameters
        ----------
        first_joint_increment : Optional[float]
            The increment for the first joint.
        first_joint_initial_stroke_length : Optional[float]
            The initial stroke length for the first joint.
        first_joint_actuator_offset : Optional[float]
            The initial actuator offset for the first joint.
        first_joint_radius : Optional[float]
            The radius of the first joint.
        first_joint_phi_0 : Optional[float]
            The initial phi value of the first joint.
        second_joint_increment : Optional[float]
            The increment for the second joint.
        second_joint_initial_stroke_length : Optional[float]
            The initial stroke length for the second joint.
        second_joint_actuator_offset : Optional[float]
            The initial actuator offset for the second joint.
        second_joint_radius : Optional[float]
            The radius for the second joint.
        second_joint_phi_0 : Optional[float]
            The initial phi value of the second joint.
        """
        self.first_joint_increment = first_joint_increment
        self.first_joint_initial_stroke_length = first_joint_initial_stroke_length
        self.first_joint_actuator_offset = first_joint_actuator_offset
        self.first_joint_radius = first_joint_radius
        self.first_joint_phi_0 = first_joint_phi_0
        self.second_joint_increment = second_joint_increment
        self.second_joint_initial_stroke_length = second_joint_initial_stroke_length
        self.second_joint_actuator_offset = second_joint_actuator_offset
        self.second_joint_radius = second_joint_radius
        self.second_joint_phi_0 = second_joint_phi_0

    def create_actuator_parameters_dict(self) -> Dict[str, float]:
        """
        Create a dictionary containing the parameters for the actuator.

        Returns
        -------
        Dict[str, float]
            A dictionary containing the configuration parameters for the actuator.
        """
        actuator_parameters_dict = {}
        if self.first_joint_increment is not None:
            actuator_parameters_dict.update(
                {config_dictionary.first_joint_increment: self.first_joint_increment}
            )
        if self.first_joint_initial_stroke_length is not None:
            actuator_parameters_dict.update(
                {
                    config_dictionary.first_joint_initial_stroke_length: self.first_joint_initial_stroke_length
                }
            )
        if self.first_joint_actuator_offset is not None:
            actuator_parameters_dict.update(
                {
                    config_dictionary.first_joint_actuator_offset: self.first_joint_actuator_offset
                }
            )
        if self.first_joint_radius is not None:
            actuator_parameters_dict.update(
                {config_dictionary.first_joint_radius: self.first_joint_radius}
            )
        if self.first_joint_phi_0 is not None:
            actuator_parameters_dict.update(
                {config_dictionary.first_joint_phi_0: self.first_joint_phi_0}
            )
        if self.second_joint_increment is not None:
            actuator_parameters_dict.update(
                {config_dictionary.second_joint_increment: self.second_joint_increment}
            )
        if self.second_joint_initial_stroke_length is not None:
            actuator_parameters_dict.update(
                {
                    config_dictionary.second_joint_initial_stroke_length: self.second_joint_initial_stroke_length
                }
            )
        if self.second_joint_actuator_offset is not None:
            actuator_parameters_dict.update(
                {
                    config_dictionary.second_joint_actuator_offset: self.second_joint_actuator_offset
                }
            )
        if self.second_joint_radius is not None:
            actuator_parameters_dict.update(
                {config_dictionary.second_joint_radius: self.second_joint_radius}
            )
        if self.second_joint_phi_0 is not None:
            actuator_parameters_dict.update(
                {config_dictionary.second_joint_phi_0: self.second_joint_phi_0}
            )
        return actuator_parameters_dict


class ActuatorConfig:
    """
    Contains the configuration parameters for the actuator.

    Attributes
    ----------
    actuator_type : str
        The type of actuator to use, e.g. linear or ideal.
    actuator_parameters : Optional[ActuatorParameters]
        The parameters of the actuator

    Methods
    -------
    create_actuator_dict()
        Create a dictionary containing the configuration parameters for the actuator.
    """

    def __init__(
        self,
        actuator_type: str,
        actuator_parameters: Optional[ActuatorParameters] = None,
    ) -> None:
        """
        Initialize the actuator configuration.

        Parameters
        ----------
        actuator_type : str
            The type of actuator to use, e.g. linear or ideal.
        actuator_parameters : Optional[ActuatorParameters]
            The parameters of the actuator
        """
        self.actuator_type = actuator_type.lower()
        self.actuator_parameters = actuator_parameters

    def create_actuator_dict(self) -> Dict[str, Any]:
        """
        Create a dictionary containing the actuator configuration.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the actuator configuration.
        """
        actuator_dict = {config_dictionary.actuator_type_key: self.actuator_type}
        if self.actuator_parameters is not None:
            actuator_dict.update(
                {
                    config_dictionary.actuator_parameters_key: self.actuator_parameters.create_actuator_parameters_dict()
                }
            )
        return actuator_dict


class ActuatorPrototypeConfig(ActuatorConfig):
    """
    Contains the configuration parameters for the actuator.

    Attributes
    ----------
    actuator_type : str
        The type of actuator to use, e.g. linear or ideal.
    actuator_parameters : Optional[ActuatorParameters]
        The parameters of the actuator

    See Also
    --------
    --------:
    class:`ActuatorConfig` : Reference to the parent class.
    """

    def __init__(
        self,
        actuator_type: str,
        actuator_parameters: Optional[ActuatorParameters] = None,
    ) -> None:
        """
        Initialize the actuator prototype configuration.

        Parameters
        ----------
        actuator_type : str
            The type of actuator to use, e.g., linear or ideal.
        actuator_parameters : Optional[ActuatorParameters]
            The parameters of the actuator
        """
        super().__init__(
            actuator_type=actuator_type, actuator_parameters=actuator_parameters
        )


class PrototypeConfig:
    """
    Contains the prototype configuration.

    Attributes
    ----------
    surface_prototype : SurfacePrototypeConfig
        The prototype for the surface.
    kinematic_prototype : KinematicPrototypeConfig
        The prototype for the kinematic.
    actuator_prototype : ActuatorPrototypeConfig
        The prototype for the actuator.

    Methods
    -------
    create_prototype_dict : Dict[str, Any]
        Creates a dictionary containing the prototypes.
    """

    def __init__(
        self,
        surface_prototype: SurfacePrototypeConfig,
        kinematic_prototype: KinematicPrototypeConfig,
        actuator_prototype: ActuatorPrototypeConfig,
    ) -> None:
        """
        Initialize the prototype configuration.

        Parameters
        ----------
        surface_prototype : SurfacePrototypeConfig
            The prototype for the surface.
        kinematic_prototype : KinematicPrototypeConfig
            The prototype for the kinematic.
        actuator_prototype : ActuatorPrototypeConfig
            The prototype for the actuator.
        """
        self.surface_prototype = surface_prototype
        self.kinematic_prototype = kinematic_prototype
        self.actuator_prototype = actuator_prototype

    def create_prototype_dict(self) -> Dict[str, Any]:
        """
        Create a dictionary containing the prototypes.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the prototypes.
        """
        return {
            config_dictionary.surface_prototype_key: self.surface_prototype.create_surface_dict(),
            config_dictionary.kinematic_prototype_key: self.kinematic_prototype.create_kinematic_dict(),
            config_dictionary.actuator_prototype_key: self.actuator_prototype.create_actuator_dict(),
        }


class HeliostatConfig:
    """
    Contains the configurations for a single heliostat.

    Attributes
    ----------
    heliostat_key : str
        The key used to identify the heliostat in the HDF5 file.
    heliostat_id : int
        The numerical ID of the heliostat.
    heliostat_position : torch.Tensor
        The position of the heliostat.
    heliostat_aim_point : torch.Tensor
        The position of the heliostat aim point.
    heliostat_surface : Optional[SurfaceConfig]
        An optional individual surface config for the heliostat.
    heliostat_kinematic : Optional[KinematicConfig]
        An optional kinematic config for the heliostat.
    heliostat_actuator : Optional[ActuatorConfig]
        An optional actuator config for the heliostat.

    Methods
    -------
    create_heliostat_config_dict()
        Create a dictionary containing the configuration parameters for a heliostat.
    """

    def __init__(
        self,
        heliostat_key: str,
        heliostat_id: int,
        heliostat_position: torch.Tensor,
        heliostat_aim_point: torch.Tensor,
        heliostat_surface: Optional[SurfaceConfig] = None,
        heliostat_kinematic: Optional[KinematicConfig] = None,
        heliostat_actuator: Optional[ActuatorConfig] = None,
    ) -> None:
        """
        Initialize the single heliostat configuration.

        Parameters
        ----------
         heliostat_key : str
            The key used to identify the heliostat in the HDF5 file.
        heliostat_id : int
            The numerical ID of the heliostat.
        heliostat_position : torch.Tensor
            The position of the heliostat.
        heliostat_aim_point : torch.Tensor
            The position of the heliostat aim point.
        heliostat_surface : Optional[SurfaceConfig]
            An optional individual surface config for the heliostat.
        heliostat_kinematic : Optional[KinematicConfig]
            An optional kinematic config for the heliostat.
        heliostat_actuator : Optional[ActuatorConfig]
            An optional actuator config for the heliostat.

        """
        self.heliostat_key = heliostat_key
        self.heliostat_id = heliostat_id
        self.heliostat_position = heliostat_position
        self.heliostat_aim_point = heliostat_aim_point
        self.heliostat_surface = heliostat_surface
        self.heliostat_kinematic = heliostat_kinematic
        self.heliostat_actuator = heliostat_actuator

    def create_heliostat_config_dict(self) -> Dict[str, Any]:
        """
        Create a dictionary containing the heliostat configuration parameters.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the heliostat configuration parameters.
        """
        heliostat_dict = {
            config_dictionary.heliostat_id: self.heliostat_id,
            config_dictionary.heliostat_position: self.heliostat_position,
            config_dictionary.heliostat_aim_point: self.heliostat_aim_point,
        }
        if self.heliostat_surface is not None:
            heliostat_dict.update(
                {
                    config_dictionary.heliostat_surface_key: self.heliostat_surface.create_surface_dict()
                }
            )
        if self.heliostat_kinematic is not None:
            heliostat_dict.update(
                {
                    config_dictionary.heliostat_kinematic_key: self.heliostat_kinematic.create_kinematic_dict()
                }
            )
        if self.heliostat_actuator is not None:
            heliostat_dict.update(
                {
                    config_dictionary.heliostat_actuator_key: self.heliostat_actuator.create_actuator_dict()
                }
            )

        return heliostat_dict


class HeliostatListConfig:
    """
    Contains the configurations for the list of heliostats included in the scenario.

    Attributes
    ----------
    heliostat_list : List[HeliostatConfig]
        The list of heliostats to include.

    Methods
    -------
    create_heliostat_list_dict()
        Create a dict containing the parameters for the heliostat list configuration.
    """

    def __init__(
        self,
        heliostat_list: List[HeliostatConfig],
    ) -> None:
        """
        Initialize the heliostat list configuration.

        Parameters
        ----------
        heliostat_list : List[HeliostatConfig]
            The list of heliostats to include.
        """
        self.heliostat_list = heliostat_list

    def create_heliostat_list_dict(self) -> Dict[str, Any]:
        """
        Create a dictionary containing the heliostat list configuration parameters.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the heliostat list configuration parameters.
        """
        return {
            heliostat.heliostat_key: heliostat.create_heliostat_config_dict()
            for heliostat in self.heliostat_list
        }


class ScenarioGenerator:
    """
    Generate an ARTIST scenario, saving it as an HDF5 file.

    Attributes
    ----------
    file_path : str
        File path to the HDF5 to be saved.
    receiver_list_config : ReceiverListConfig
        The receiver list configuration object.
    light_source_list_config : LightSourceListConfig
        The light source list configuration object.
    heliostat_list_config : HeliostatListConfig
        The heliostat_list configuration object.
    prototype_config : PrototypeConfig
        The prototype configuration object,
    version : Optional[float]
        The version of the scenario generator being used.
    log : logging.Logger
        The logger.

    Methods
    -------
    flatten_dict()
        Flatten nested dictionaries to first-level keys.
    include_parameters()
        Include the parameters from a parameter dictionary.
    generate_scenario()
        Generate the scenario according to the given parameters.
    """

    def __init__(
        self,
        file_path: str,
        receiver_list_config: ReceiverListConfig,
        light_source_list_config: LightSourceListConfig,
        heliostat_list_config: HeliostatListConfig,
        prototype_config: PrototypeConfig,
        version: Optional[float] = 1.0,
        log_level: Optional[int] = logging.INFO,
    ) -> None:
        """
        Initialize the scenario generator.

        Parameters
        ----------
        file_path : str
            File path to the HDF5 to be saved.
        receiver_list_config : ReceiverListConfig
            The receiver list configuration object.
        light_source_list_config : LightSourceListConfig
            The light source list configuration object.
        heliostat_list_config : HeliostatListConfig
            The heliostat_list configuration object.
        prototype_config : PrototypeConfig
            The prototype configuration object,
        version : Optional[float]
            The version of the scenario generator being used.
        log_level : Optional[int]
            The log level applied to the logger.
        """
        self.file_path = file_path
        self.receiver_list_config = receiver_list_config
        self.light_source_list_config = light_source_list_config
        self.heliostat_list_config = heliostat_list_config
        self.prototype_config = prototype_config
        self.version = version
        log = logging.getLogger("scenario-generator")  # Get logger instance.
        log_formatter = colorlog.ColoredFormatter(
            fmt="[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s]"
            "[%(log_color)s%(levelname)s%(reset)s] - %(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
            secondary_log_colors={},
        )
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(log_formatter)
        log.addHandler(handler)
        log.setLevel(log_level)
        self.log = log

    def flatten_dict(
        self, dictionary: MutableMapping, parent_key: str = "", sep: str = "/"
    ) -> Dict[str, Any]:
        """
        Flatten nested dictionaries to first-level keys.

        Parameters
        ----------
        dictionary : MutableMapping
            Original nested dictionary to flatten.
        parent_key : str
            The parent key of nested dictionaries. Should be empty upon initialization.
        sep : str
            The separator used to separate keys in nested dictionaries.

        Returns
        -------
        Dict
            A flattened version of the original dictionary.
        """
        return dict(self._flatten_dict_gen(dictionary, parent_key, sep))

    def _flatten_dict_gen(self, d: MutableMapping, parent_key: str, sep: str) -> None:
        # Flattens the keys in a nested dictionary so that the resulting key is a concatenation of all nested keys
        # separated by a defined separator.
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, MutableMapping):
                yield from self.flatten_dict(v, new_key, sep=sep).items()
            else:
                yield new_key, v

    @staticmethod
    def include_parameters(file: h5py.File, prefix: str, parameters: Dict) -> None:
        """
        Include the parameters from a parameter dictionary.

        Parameters
        ----------
        file : h5py.File
            The HDF5 file to write to.
        prefix : str
            The prefix used for naming the parameters.
        parameters : Dict
            The parameters to be included into the HFD5 file.
        """
        for key, value in parameters.items():
            file[f"{prefix}/{key}"] = value

    def generate_scenario(self) -> None:
        """Generate the scenario according to the given parameters."""
        self.log.info(f"Generating a scenario saved to: {self.file_path}")
        with h5py.File(f"{self.file_path}.h5", "w") as f:
            # Set scenario version as attribute.
            self.log.info(f"Using scenario generator version {self.version}")
            f.attrs["version"] = self.version

            # Include parameters for the receivers.
            self.log.info("Including parameters for the receivers")
            self.include_parameters(
                file=f,
                prefix=config_dictionary.receiver_key,
                parameters=self.flatten_dict(
                    self.receiver_list_config.create_receiver_list_dict()
                ),
            )

            # Include parameters for the light sources.
            self.log.info("Including parameters for the light sources")
            self.include_parameters(
                file=f,
                prefix=config_dictionary.light_source_key,
                parameters=self.flatten_dict(
                    self.light_source_list_config.create_light_source_list_dict()
                ),
            )

            # Include parameters for the prototype.
            self.log.info("Including parameters for the prototype")
            self.include_parameters(
                file=f,
                prefix=config_dictionary.prototype_key,
                parameters=self.flatten_dict(
                    self.prototype_config.create_prototype_dict()
                ),
            )

            # Include heliostat parameters.
            self.log.info("Including parameters for the heliostats")
            self.include_parameters(
                file=f,
                prefix=config_dictionary.heliostat_key,
                parameters=self.flatten_dict(
                    self.heliostat_list_config.create_heliostat_list_dict()
                ),
            )
