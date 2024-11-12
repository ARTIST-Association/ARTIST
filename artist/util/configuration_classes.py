from typing import Any, Optional

import torch

from artist.util import config_dictionary

class PowerPlantConfig:
    """
    Store the power plant configuration parameters.

    Attributes
    ----------
    power_plant_position : torch.Tensor
        The position of the power plant in lat, lon, alt.

    Methods
    -------
    create_power_plant_dict
       Create a dictionary containing the configuration parameters for the power plant.
    """

    def __init__(
        self,
        power_plant_position: torch.Tensor,
    ) -> None:
        """
        Initialize the power plant configuration.

        Parameters
        ----------
        power_plant_position : torch.Tensor
            The position of the power plant in lat, lon, alt.
        """
        self.power_plant_position = power_plant_position

    def create_power_plant_dict(self) -> dict[str, Any]:
        """
        Create a dictionary containing the configuration parameters for the power plant.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the configuration parameters for the power plant.
        """
        power_plant_dict = {
            config_dictionary.power_plant_position: self.power_plant_position,
        }
        return power_plant_dict


class ReceiverConfig:
    """
    Store the receiver configuration parameters.

    Attributes
    ----------
    receiver_key : str
        The ID string used to identify the receiver in the HDF5 file.
    receiver_type : str
        The type of receiver, e.g., planar.
    position_center : torch.Tensor
        The position of the receiver's center.
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
        curvature_e: float, optional
            The curvature of the receiver in the east direction.
        curvature_u: float, optional
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

    def create_receiver_dict(self) -> dict[str, Any]:
        """
        Create a dictionary containing the configuration parameters for the receiver.

        Returns
        -------
        dict[str, Any]
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
    Store the receiver list configuration parameters.

    Attributes
    ----------
    receiver_list : list[ReceiverConfig]
        A list of receiver configurations to be included in the scenario.

    Methods
    -------
    create_receiver_list_dict()
       Create a dictionary containing the configuration parameters for the list of receivers.
    """

    def __init__(self, receiver_list: list[ReceiverConfig]) -> None:
        """
        Initialize the receiver list configuration.

        Parameters
        ----------
        receiver_list : list[ReceiverConfig]
            The list of receiver configurations included in the scenario.
        """
        self.receiver_list = receiver_list

    def create_receiver_list_dict(self) -> dict[str, Any]:
        """
        Create a dictionary containing the configuration parameters for the list of receivers.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the configuration parameters for the list of receivers.
        """
        return {
            receiver.receiver_key: receiver.create_receiver_dict()
            for receiver in self.receiver_list
        }


class LightSourceConfig:
    """
    Store the light source configuration parameters.

    Attributes
    ----------
    light_source_key : str
        The ID string used to identify the light source in the HDF5 file.
    light_source_type:
        The type of light source used, e.g., a sun.
    number_of_rays : int
        The number of rays generated by the light source.
    distribution_type : str
        The distribution type used to model the light source.
    mean : float, optional
        The mean used for modeling the sun.
    covariance : float, optional
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
        mean: Optional[float] = None,
        covariance: Optional[float] = None,
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
        mean : float, optional
            The mean used for modeling the sun.
        covariance : float, optional
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

    def create_light_source_dict(self) -> dict[str, Any]:
        """
        Create a dictionary containing the configuration parameters for the light source.

        Returns
        -------
        dict[str, Any]
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
            config_dictionary.light_source_type: self.light_source_type,
            config_dictionary.light_source_number_of_rays: self.number_of_rays,
            config_dictionary.light_source_distribution_parameters: light_source_distribution_parameters_dict,
        }


class LightSourceListConfig:
    """
    Store the light source list configuration parameters.

    Attributes
    ----------
    light_source_list : list[LightSourceConfig]
        The list of light source configs to be included in the scenario.

    Methods
    -------
    create_light_list_dict()
       Create a dictionary containing the configuration parameters for the light source list.
    """

    def __init__(self, light_source_list: list[LightSourceConfig]) -> None:
        """
        Initialize the light source list configuration.

        Parameters
        ----------
        light_source_list : list[LightSourceConfig]
            The list of light source configs to be included in the scenario.
        """
        self.light_source_list = light_source_list

    def create_light_source_list_dict(self) -> dict[str, Any]:
        """
        Create a dictionary containing the configuration parameters for the light source list.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the configuration parameters for the light source list.
        """
        return {
            ls.light_source_key: ls.create_light_source_dict()
            for ls in self.light_source_list
        }


class FacetConfig:
    """
    Store the facet configuration parameters.

    Attributes
    ----------
    facet_key : str
        The key used to identify the facet in the HDF5 file.
    control_points : torch.Tensor
        The NURBS control points.
    degree_e : torch.Tensor
        The NURBS degree in the east direction.
    degree_n : torch.Tensor
        The NURBS degree in the north direction.
    number_eval_points_e : int
        The number of evaluation points for the NURBS surface in the east direction.
    number_eval_points_n : int
        The number of evaluation points for the NURBS surface in the north direction.
    width : float
        The width of the facet.
    height : float
        The height of the facet.
    translation_vector : torch.Tensor
        The translation_vector of the facet.
    canting_e : torch.Tensor
        The canting vector in the east direction.
    canting_n : torch.Tensor
        The canting vector in the north direction.

    Methods
    -------
    create_facet_dict()
       Create a dictionary containing the configuration parameters for a facet.
    """

    def __init__(
        self,
        facet_key: str,
        control_points: torch.Tensor,
        degree_e: int,
        degree_n: int,
        number_eval_points_e: int,
        number_eval_points_n: int,
        width: float,
        height: float,
        translation_vector: torch.Tensor,
        canting_e: torch.Tensor,
        canting_n: torch.Tensor,
    ) -> None:
        """
        Initialize the facet configuration.

        Parameters
        ----------
        facet_key : str
            The key used to identify the facet in the HDF5 file.
        control_points : torch.Tensor
            The NURBS control points.
        degree_e : torch.Tensor
            The NURBS degree in the east direction.
        degree_n : torch.Tensor
            The NURBS degree in the north direction.
        number_eval_points_e : int
            The number of evaluation points for the NURBS surface in the east direction.
        number_eval_points_n : int
            The number of evaluation points for the NURBS surface in the north direction.
        width : float
            The width of the facet.
        height : float
            The height of the facet.
        translation_vector : torch.Tensor
            The translation_vector of the facet.
        canting_e : torch.Tensor
            The canting vector in the east direction.
        canting_n : torch.Tensor
            The canting vector in the north direction.
        """
        self.facet_key = facet_key
        self.control_points = control_points
        self.degree_e = degree_e
        self.degree_n = degree_n
        self.number_eval_points_e = number_eval_points_e
        self.number_eval_points_n = number_eval_points_n
        self.width = width
        self.height = height
        self.translation_vector = translation_vector
        self.canting_e = canting_e
        self.canting_n = canting_n

    def create_facet_dict(self) -> dict[str, Any]:
        """
        Create a dictionary containing the configuration parameters for a facet.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the configuration parameters for the facet.
        """
        return {
            config_dictionary.facet_control_points: self.control_points,
            config_dictionary.facet_degree_e: self.degree_e,
            config_dictionary.facet_degree_n: self.degree_n,
            config_dictionary.facet_number_eval_e: self.number_eval_points_e,
            config_dictionary.facet_number_eval_n: self.number_eval_points_n,
            config_dictionary.facets_width: self.width,
            config_dictionary.facets_height: self.height,
            config_dictionary.facets_translation_vector: self.translation_vector,
            config_dictionary.facets_canting_e: self.canting_e,
            config_dictionary.facets_canting_n: self.canting_n,
        }


class SurfaceConfig:
    """
    Store the surface configuration parameters.

    Attributes
    ----------
    facets_list : list[FacetsConfiguration]
        The list of facets to be used for the surface of the heliostat.

    Methods
    -------
    create_surface_dict()
       Create a dictionary containing the configuration parameters for the surface.
    """

    def __init__(self, facets_list: list[FacetConfig]) -> None:
        """
        Initialize the surface configuration.

        Parameters
        ----------
        facets_list : list[FacetsConfig]
            The list of facets to be used for the surface of the heliostat.
        """
        self.facets_list = facets_list

    def create_surface_dict(self) -> dict[str, Any]:
        """
        Create a dictionary containing the configuration parameters for the surface.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the configuration parameters for the surface.
        """
        facets_dict = {
            facet.facet_key: facet.create_facet_dict() for facet in self.facets_list
        }
        return {config_dictionary.facets_key: facets_dict}


class SurfacePrototypeConfig(SurfaceConfig):
    """
    Store the configuration parameters for a surface prototype.

    See Also
    --------
    :class:`SurfaceConfig` : Reference to the parent class.
    """

    def __init__(self, facets_list: list[FacetConfig]) -> None:
        """
        Initialize the surface prototype configuration.

        Parameters
        ----------
        facets_list : list[FacetsConfig]
            The list of facets to be used for the surface of the heliostat prototype.
        """
        super().__init__(facets_list=facets_list)


class KinematicDeviations:
    """
    Store the kinematic deviations.

    Attributes
    ----------
    first_joint_translation_e : torch.Tensor, optional
        The first joint translation in the east direction.
    first_joint_translation_n : torch.Tensor, optional
        The first joint translation in the north direction.
    first_joint_translation_u : torch.Tensor, optional
        The first joint translation in the up direction.
    first_joint_tilt_e : torch.Tensor, optional
        The first joint tilt in the east direction.
    first_joint_tilt_n : torch.Tensor, optional
        The first joint tilt in the north direction.
    first_joint_tilt_u : torch.Tensor, optional
        The first joint tilt in the up direction.
    second_joint_translation_e : torch.Tensor, optional
        The second joint translation in the east direction.
    second_joint_translation_n : torch.Tensor, optional
        The second joint translation in the north direction.
    second_joint_translation_u : torch.Tensor, optional
        The second joint translation in the up direction.
    second_joint_tilt_e : torch.Tensor, optional
        The second joint tilt in the east direction.
    second_joint_tilt_n : torch.Tensor, optional
        The second joint tilt in the north direction.
    second_joint_tilt_u : torch.Tensor, optional
        The second joint tilt in the up direction.
    concentrator_translation_e : torch.Tensor, optional
        The concentrator translation in the east direction.
    concentrator_translation_n : torch.Tensor, optional
        The concentrator translation in the north direction.
    concentrator_translation_u : torch.Tensor, optional
        The concentrator translation in the up direction.
    concentrator_tilt_e : torch.Tensor, optional
        The concentrator tilt in the east direction.
    concentrator_tilt_n : torch.Tensor, optional
        The concentrator tilt in the north direction.
    concentrator_tilt_u : torch.Tensor, optional
        The concentrator tilt in the up direction.

    Methods
    -------
    create_kinematic_deviations_dict()
        Create a dictionary containing the configuration parameters for the kinematic deviations.
    """

    def __init__(
        self,
        first_joint_translation_e: Optional[torch.Tensor] = None,
        first_joint_translation_n: Optional[torch.Tensor] = None,
        first_joint_translation_u: Optional[torch.Tensor] = None,
        first_joint_tilt_e: Optional[torch.Tensor] = None,
        first_joint_tilt_n: Optional[torch.Tensor] = None,
        first_joint_tilt_u: Optional[torch.Tensor] = None,
        second_joint_translation_e: Optional[torch.Tensor] = None,
        second_joint_translation_n: Optional[torch.Tensor] = None,
        second_joint_translation_u: Optional[torch.Tensor] = None,
        second_joint_tilt_e: Optional[torch.Tensor] = None,
        second_joint_tilt_n: Optional[torch.Tensor] = None,
        second_joint_tilt_u: Optional[torch.Tensor] = None,
        concentrator_translation_e: Optional[torch.Tensor] = None,
        concentrator_translation_n: Optional[torch.Tensor] = None,
        concentrator_translation_u: Optional[torch.Tensor] = None,
        concentrator_tilt_e: Optional[torch.Tensor] = None,
        concentrator_tilt_n: Optional[torch.Tensor] = None,
        concentrator_tilt_u: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initialize the kinematic deviations.

        Parameters
        ----------
        first_joint_translation_e : torch.Tensor, optional
            The first joint translation in the east direction.
        first_joint_translation_n : torch.Tensor, optional
            The first joint translation in the north direction.
        first_joint_translation_u : torch.Tensor, optional
            The first joint translation in the up direction.
        first_joint_tilt_e : torch.Tensor, optional
            The first joint tilt in the east direction.
        first_joint_tilt_n : torch.Tensor, optional
            The first joint tilt in the north direction.
        first_joint_tilt_u : torch.Tensor, optional
            The first joint tilt in the up direction.
        second_joint_translation_e : torch.Tensor, optional
            The second joint translation in the east direction.
        second_joint_translation_n : torch.Tensor, optional
            The second joint translation in the north direction.
        second_joint_translation_u : torch.Tensor, optional
            The second joint translation in the up direction.
        second_joint_tilt_e : torch.Tensor, optional
            The second joint tilt in the east direction.
        second_joint_tilt_n : torch.Tensor, optional
            The second joint tilt in the north direction.
        second_joint_tilt_u : torch.Tensor, optional
            The second joint tilt in the up direction.
        concentrator_translation_e : torch.Tensor, optional
            The concentrator translation in the east direction.
        concentrator_translation_n : torch.Tensor, optional
            The concentrator translation in the north direction.
        concentrator_translation_u : torch.Tensor, optional
            The concentrator translation in the up direction.
        concentrator_tilt_e : torch.Tensor, optional
            The concentrator tilt in the east direction.
        concentrator_tilt_n : torch.Tensor, optional
            The concentrator tilt in the north direction.
        concentrator_tilt_u : torch.Tensor, optional
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

    def create_kinematic_deviations_dict(self) -> dict[str, Any]:
        """
        Create a dictionary containing the configuration parameters for the kinematic deviations.

        Returns
        -------
        dict[str, Any]
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
    Store the kinematic offsets.

    Attributes
    ----------
    kinematic_initial_orientation_offset_e : torch.Tensor, optional
        The initial orientation offset in the east direction.
    kinematic_initial_orientation_offset_n : torch.Tensor, optional
        The initial orientation offset in the north direction.
    kinematic_initial_orientation_offset_u : torch.Tensor, optional
        The initial orientation offset in the up direction.

    Methods
    -------
    create_kinematic_offsets_dict()
        Create a dictionary containing the configuration parameters for the kinematic offsets.
    """

    def __init__(
        self,
        kinematic_initial_orientation_offset_e: Optional[torch.Tensor] = None,
        kinematic_initial_orientation_offset_n: Optional[torch.Tensor] = None,
        kinematic_initial_orientation_offset_u: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initialize the initial orientation offsets.

        Parameters
        ----------
        kinematic_initial_orientation_offset_e : torch.Tensor, optional
            The initial orientation offset in the east direction.
        kinematic_initial_orientation_offset_n : torch.Tensor, optional
            The initial orientation offset in the north direction.
        kinematic_initial_orientation_offset_u : torch.Tensor, optional
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

    def create_kinematic_offsets_dict(self) -> dict[str, Any]:
        """
        Create a dictionary containing the configuration parameters for the kinematic offsets.

        Returns
        -------
        dict[str, Any]
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
    Store the configuration parameters for the kinematic.

    Attributes
    ----------
    kinematic_type : str
        The type of kinematic used.
    kinematic_initial_orientation_offsets : KinematicOffsets, optional
        The initial orientation offsets of the kinematic configuration.
    kinematic_deviations : KinematicDeviations, optional
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
        kinematic_initial_orientation_offsets : KinematicOffsets, optional
            The initial orientation offsets of the kinematic configuration.
        kinematic_deviations : KinematicDeviations, optional
            The kinematic deviations.
        """
        self.kinematic_type = kinematic_type
        self.kinematic_initial_orientation_offsets = (
            kinematic_initial_orientation_offsets
        )
        self.kinematic_deviations = kinematic_deviations

    def create_kinematic_dict(self) -> dict[str, Any]:
        """
        Create a dictionary containing the configuration parameters for the kinematic.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the configuration parameters for the kinematic.
        """
        kinematic_dict: dict[str, Any] = {
            config_dictionary.kinematic_type: self.kinematic_type
        }
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
    Store the configuration parameters for the kinematic prototype.

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
        kinematic_initial_orientation_offsets : KinematicOffsets, optional
            The initial orientation offsets of the kinematic configuration.
        kinematic_deviations : KinematicDeviations, optional
            The kinematic deviations.
        """
        super().__init__(
            kinematic_type=kinematic_type,
            kinematic_initial_orientation_offsets=kinematic_initial_orientation_offsets,
            kinematic_deviations=kinematic_deviations,
        )


class KinematicLoadConfig:
    """
    Store the configuration parameters for the kinematic when loaded in ``ARTIST``.

    Attributes
    ----------
    kinematic_type : str
        The type of kinematic used.
    kinematic_initial_orientation_offsets : KinematicOffsets, optional
        The initial orientation offsets of the kinematic configuration.
    kinematic_deviations : KinematicDeviations, optional
        The kinematic deviations.
    """

    def __init__(
        self,
        kinematic_type: str,
        kinematic_initial_orientation_offsets: KinematicOffsets,
        kinematic_deviations: KinematicDeviations,
    ) -> None:
        """
        Initialize the kinematic configuration for loading in ``ARTIST``.

        Parameters
        ----------
        kinematic_type : str
            The type of kinematic used.
        kinematic_initial_orientation_offsets : KinematicOffsets, optional
            The initial orientation offsets of the kinematic configuration.
        kinematic_deviations : KinematicDeviations, optional
            The kinematic deviations.
        """
        self.kinematic_type = kinematic_type
        self.kinematic_initial_orientation_offsets = (
            kinematic_initial_orientation_offsets
        )
        self.kinematic_deviations = kinematic_deviations


class ActuatorParameters:
    """
    Store the actuator parameters.

    Attributes
    ----------
    increment : torch.Tensor, optional
        The increment for the actuator
    initial_stroke_length : torch.Tensor, optional
        The initial stroke length.
    offset : torch.Tensor, optional
        The initial actuator offset.
    radius : torch.Tensor, optional
        The radius of the considered joint.
    phi_0 : torch.Tensor, optional
        The initial phi value of the actuator.

    Methods
    -------
    create_actuator_parameters_dict()
        Create a dictionary containing the configuration parameters for the actuator parameters.
    """

    def __init__(
        self,
        increment: Optional[torch.Tensor] = None,
        initial_stroke_length: Optional[torch.Tensor] = None,
        offset: Optional[torch.Tensor] = None,
        radius: Optional[torch.Tensor] = None,
        phi_0: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initialize the actuator parameters.

        Parameters
        ----------
        increment : torch.Tensor, optional
            The increment for the actuator
        initial_stroke_length : torch.Tensor, optional
            The initial stroke length.
        offset : torch.Tensor, optional
            The initial actuator offset.
        radius : torch.Tensor, optional
            The radius of the considered joint.
        phi_0 : torch.Tensor, optional
            The initial phi value of the actuator.
        """
        self.increment = increment
        self.initial_stroke_length = initial_stroke_length
        self.offset = offset
        self.radius = radius
        self.phi_0 = phi_0

    def create_actuator_parameters_dict(self) -> dict[str, torch.Tensor]:
        """
        Create a dictionary containing the parameters for the actuator.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing the configuration parameters for the actuator.
        """
        actuator_parameters_dict = {}
        if self.increment is not None:
            actuator_parameters_dict.update(
                {config_dictionary.actuator_increment: self.increment}
            )
        if self.initial_stroke_length is not None:
            actuator_parameters_dict.update(
                {
                    config_dictionary.actuator_initial_stroke_length: self.initial_stroke_length
                }
            )
        if self.offset is not None:
            actuator_parameters_dict.update(
                {config_dictionary.actuator_offset: self.offset}
            )
        if self.radius is not None:
            actuator_parameters_dict.update(
                {config_dictionary.actuator_radius: self.radius}
            )
        if self.phi_0 is not None:
            actuator_parameters_dict.update(
                {config_dictionary.actuator_phi_0: self.phi_0}
            )
        return actuator_parameters_dict


class ActuatorConfig:
    """
    Store the configuration parameters for the actuator.

    Attributes
    ----------
    actuator_type : str
        The type of actuator to use, e.g., linear or ideal.
    actuator_clockwise : bool
        Boolean indicating if the actuator operates in a clockwise manner.
    actuator_parameters : ActuatorParameters, optional
        The parameters of the actuator

    Methods
    -------
    create_actuator_dict()
        Create a dictionary containing the configuration parameters for the actuator.
    """

    def __init__(
        self,
        actuator_key: str,
        actuator_type: str,
        actuator_clockwise: bool,
        actuator_parameters: Optional[ActuatorParameters] = None,
    ) -> None:
        """
        Initialize the actuator configuration.

        Parameters
        ----------
        actuator_type : str
            The type of actuator to use, e.g. linear or ideal.
        actuator_clockwise : bool
            Boolean indicating if the actuator operates in a clockwise manner.
        actuator_parameters : ActuatorParameters, optional
            The parameters of the actuator.
        """
        self.actuator_key = actuator_key
        self.actuator_type = actuator_type.lower()
        self.actuator_clockwise = actuator_clockwise
        self.actuator_parameters = actuator_parameters

    def create_actuator_dict(self) -> dict[str, Any]:
        """
        Create a dictionary containing the actuator configuration.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the actuator configuration.
        """
        actuator_dict = {
            config_dictionary.actuator_type_key: self.actuator_type,
            config_dictionary.actuator_clockwise: self.actuator_clockwise,
        }
        if self.actuator_parameters is not None:
            actuator_dict.update(
                {
                    config_dictionary.actuator_parameters_key: self.actuator_parameters.create_actuator_parameters_dict()
                }
            )
        return actuator_dict


class ActuatorListConfig:
    """
    Store the configuration parameters for a list of actuators.

    Attributes
    ----------
    actuator_list : list[ActuatorConfig]
        A list of actuator configurations.

    Methods
    -------
    create_actuator_list_dict()
        Creates a dictionary containing a list of actuator configurations.
    """

    def __init__(self, actuator_list: list[ActuatorConfig]) -> None:
        """
        Initialize the actuator list configuration.

        Parameters
        ----------
        actuator_list : list[ActuatorConfig]
            A list of actuator configurations.
        """
        self.actuator_list = actuator_list

    def create_actuator_list_dict(self) -> dict[str, Any]:
        """
        Create a dictionary containing a list of actuator configurations.

        Returns
        -------
        dict[str, Any]
            A dictionary containing a list of actuator configurations.
        """
        return {
            actuator_config.actuator_key: actuator_config.create_actuator_dict()
            for actuator_config in self.actuator_list
        }


class ActuatorPrototypeConfig(ActuatorListConfig):
    """
    Store the configuration parameters for the actuator prototype.

    Attributes
    ----------
    actuator_list : list[ActuatorConfig]
        A list of actuator configurations.

    See Also
    --------
    class:`ActuatorListConfig` : Reference to the parent class.
    """

    def __init__(
        self,
        actuator_list: list[ActuatorConfig],
    ) -> None:
        """
        Initialize the actuator list prototype configuration.

        Parameters
        ----------
        actuator_list : list[ActuatorConfig]
            A list of actuator configurations.
        """
        super().__init__(actuator_list=actuator_list)


class PrototypeConfig:
    """
    Store the prototype configuration.

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
    create_prototype_dict()
        Create a dictionary containing the prototypes.
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

    def create_prototype_dict(self) -> dict[str, Any]:
        """
        Create a dictionary containing the prototypes.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the prototypes.
        """
        return {
            config_dictionary.surface_prototype_key: self.surface_prototype.create_surface_dict(),
            config_dictionary.kinematic_prototype_key: self.kinematic_prototype.create_kinematic_dict(),
            config_dictionary.actuator_prototype_key: self.actuator_prototype.create_actuator_list_dict(),
        }


class HeliostatConfig:
    """
    Store the configurations for a single heliostat.

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
    heliostat_surface : SurfaceConfig, optional
        An optional individual surface config for the heliostat.
    heliostat_kinematic : KinematicConfig, optional
        An optional kinematic config for the heliostat.
    heliostat_actuator : ActuatorListConfig, optional
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
        heliostat_actuator: Optional[ActuatorListConfig] = None,
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
        heliostat_surface : SurfaceConfig, optional
            An optional individual surface config for the heliostat.
        heliostat_kinematic : KinematicConfig, optional
            An optional kinematic config for the heliostat.
        heliostat_actuator : ActuatorListConfig, optional
            An optional actuator config for the heliostat.
        """
        self.heliostat_key = heliostat_key
        self.heliostat_id = heliostat_id
        self.heliostat_position = heliostat_position
        self.heliostat_aim_point = heliostat_aim_point
        self.heliostat_surface = heliostat_surface
        self.heliostat_kinematic = heliostat_kinematic
        self.heliostat_actuator = heliostat_actuator

    def create_heliostat_config_dict(self) -> dict[str, Any]:
        """
        Create a dictionary containing the heliostat configuration parameters.

        Returns
        -------
        dict[str, Any]
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
                    config_dictionary.heliostat_actuator_key: self.heliostat_actuator.create_actuator_list_dict()
                }
            )

        return heliostat_dict


class HeliostatListConfig:
    """
    Store the configurations for the list of heliostats included in the scenario.

    Attributes
    ----------
    heliostat_list : list[HeliostatConfig]
        The list of heliostats to include.

    Methods
    -------
    create_heliostat_list_dict()
        Create a dict containing the parameters for the heliostat list configuration.
    """

    def __init__(
        self,
        heliostat_list: list[HeliostatConfig],
    ) -> None:
        """
        Initialize the heliostat list configuration.

        Parameters
        ----------
        heliostat_list : list[HeliostatConfig]
            The list of heliostats to include.
        """
        self.heliostat_list = heliostat_list

    def create_heliostat_list_dict(self) -> dict[str, Any]:
        """
        Create a dictionary containing the heliostat list configuration parameters.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the heliostat list configuration parameters.
        """
        return {
            heliostat.heliostat_key: heliostat.create_heliostat_config_dict()
            for heliostat in self.heliostat_list
        }
