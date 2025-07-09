from typing import Any

import torch

from artist.util import config_dictionary


class PowerPlantConfig:
    """
    Store the power plant configuration parameters.

    Attributes
    ----------
    power_plant_position : torch.Tensor
        The position of the power plant as latitude, longitude, altitude.

    Methods
    -------
    create_power_plant_dict()
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
            The position of the power plant as latitude, longitude, altitude.
        """
        self.power_plant_position = power_plant_position

    def create_power_plant_dict(self) -> dict[str, Any]:
        """
        Create a dictionary containing the configuration parameters for the power plant.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the configuration parameters for the power plant.
        """
        power_plant_dict = {
            config_dictionary.power_plant_position: self.power_plant_position,
        }
        return power_plant_dict


class TargetAreaConfig:
    """
    Store the tower target area configuration parameters.

    Attributes
    ----------
    target_area_key : str
        The ID string used to identify the target area in the HDF5 file.
    geometry : str
        The type of target area, e.g., planar.
    center : torch.Tensor
        The position of the target area's center.
    normal_vector : torch.Tensor
        The normal vector to the target plane.
    plane_e : float
        The size of the target area in the east direction.
    plane_u : float
        The size of the target area in the up direction.
    curvature_e : float | None
        The curvature of the target area in the east direction.
    curvature_u : float | None
        The curvature of the target area in the up direction.

    Methods
    -------
    create_receiver_dict()
       Create a dictionary containing the configuration parameters for the target area.
    """

    def __init__(
        self,
        target_area_key: str,
        geometry: str,
        center: torch.Tensor,
        normal_vector: torch.Tensor,
        plane_e: float,
        plane_u: float,
        curvature_e: float | None = None,
        curvature_u: float | None = None,
    ) -> None:
        """
        Initialize the target area configuration.

        Parameters
        ----------
        target_area_key : str
            The ID string used to identify the target area in the HDF5 file.
        geometry : str
            The type of target area, e.g., planar.
        center : torch.Tensor
            The position of the target area's center.
        normal_vector : torch.Tensor
            The normal vector to the target plane.
        plane_e : float
            The size of the target area in the east direction.
        plane_u : float
            The size of the target area in the up direction.
        curvature_e: float | None
            The curvature of the target area in the east direction.
        curvature_u: float | None
            The curvature of the target area in the up direction.
        """
        self.target_area_key = target_area_key
        self.geometry = geometry
        self.center = center
        self.normal_vector = normal_vector
        self.plane_e = plane_e
        self.plane_u = plane_u
        self.curvature_e = curvature_e
        self.curvature_u = curvature_u

    def create_target_area_dict(self) -> dict[str, Any]:
        """
        Create a dictionary containing the configuration parameters for the target area.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the configuration parameters for the target area.
        """
        target_area_dict = {
            config_dictionary.target_area_geometry: self.geometry,
            config_dictionary.target_area_position_center: self.center,
            config_dictionary.target_area_normal_vector: self.normal_vector,
            config_dictionary.target_area_plane_e: self.plane_e,
            config_dictionary.target_area_plane_u: self.plane_u,
        }
        if self.curvature_e is not None:
            target_area_dict.update(
                {config_dictionary.target_area_curvature_e: self.curvature_e}
            )
        if self.curvature_u is not None:
            target_area_dict.update(
                {config_dictionary.target_area_curvature_u: self.curvature_u}
            )

        return target_area_dict


class TargetAreaListConfig:
    """
    Store the target area list configuration parameters.

    Attributes
    ----------
    target_area_list : list[TargetAreaConfig]
        A list of target area configurations to be included in the scenario.

    Methods
    -------
    create_target_area_list_dict()
       Create a dictionary containing the configuration parameters for the list of target areas.
    """

    def __init__(self, target_area_list: list[TargetAreaConfig]) -> None:
        """
        Initialize the target area list configuration.

        Parameters
        ----------
        target_area_list : list[TargetAreaConfig]
            The list of target area configurations included in the scenario.
        """
        self.target_area_list = target_area_list

    def create_target_area_list_dict(self) -> dict[str, Any]:
        """
        Create a dictionary containing the configuration parameters for the list of target areas.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the configuration parameters for the list of target areas.
        """
        return {
            target_area.target_area_key: target_area.create_target_area_dict()
            for target_area in self.target_area_list
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
    mean : float | None
        The mean used for modeling the light source.
    covariance : float | None
        The covariance used for modeling the light source.

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
        mean: float | None = None,
        covariance: float | None = None,
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
        mean : float | None
            The mean used for modeling the light source.
        covariance : float | None
            The covariance used for modeling the light source.

        Raises
        ------
        ValueError
            If the specified light source distribution type is unknown.
        """
        self.light_source_key = light_source_key
        self.light_source_type = light_source_type
        self.number_of_rays = number_of_rays
        self.distribution_type = distribution_type

        if (
            self.distribution_type
            != config_dictionary.light_source_distribution_is_normal
        ):
            raise ValueError("Unknown light source distribution type.")

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
    degrees : torch.Tensor
        The NURBS degree in the east and north direction.
    translation_vector : torch.Tensor
        The translation_vector of the facet.
    canting: torch.Tensor
        The canting vectors in the east and north direction.

    Methods
    -------
    create_facet_dict()
       Create a dictionary containing the configuration parameters for a facet.
    """

    def __init__(
        self,
        facet_key: str,
        control_points: torch.Tensor,
        degrees: torch.Tensor,
        translation_vector: torch.Tensor,
        canting: torch.Tensor,
    ) -> None:
        """
        Initialize the facet configuration.

        Parameters
        ----------
        facet_key : str
            The key used to identify the facet in the HDF5 file.
        control_points : torch.Tensor
            The NURBS control points.
        degrees : torch.Tensor
            The NURBS degree in the east and north direction.
        translation_vector : torch.Tensor
            The translation_vector of the facet.
        canting: torch.Tensor
            The canting vectors in the east and north direction.
        """
        self.facet_key = facet_key
        self.control_points = control_points
        self.degrees = degrees
        self.translation_vector = translation_vector
        self.canting = canting

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
            config_dictionary.facet_degrees: self.degrees,
            config_dictionary.facets_translation_vector: self.translation_vector,
            config_dictionary.facets_canting: self.canting,
        }


class SurfaceConfig:
    """
    Store the surface configuration parameters.

    Attributes
    ----------
    facet_list : list[FacetsConfiguration]
        The list of facets to be used for the surface of the heliostat.

    Methods
    -------
    create_surface_dict()
       Create a dictionary containing the configuration parameters for the surface.
    """

    def __init__(self, facet_list: list[FacetConfig]) -> None:
        """
        Initialize the surface configuration.

        Parameters
        ----------
        facet_list : list[FacetsConfig]
            The list of facets to be used for the surface of the heliostat.
        """
        self.facet_list = facet_list

    def create_surface_dict(self) -> dict[str, Any]:
        """
        Create a dictionary containing the configuration parameters for the surface.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the configuration parameters for the surface.
        """
        facet_dict = {
            facet.facet_key: facet.create_facet_dict() for facet in self.facet_list
        }
        return {config_dictionary.facets_key: facet_dict}


class SurfacePrototypeConfig(SurfaceConfig):
    """
    Store the configuration parameters for a surface prototype.

    See Also
    --------
    :class:`SurfaceConfig` : Reference to the parent class.
    """

    def __init__(self, facet_list: list[FacetConfig]) -> None:
        """
        Initialize the surface prototype configuration.

        Parameters
        ----------
        facet_list : list[FacetsConfig]
            The list of facets to be used for the surface of the heliostat prototype.
        """
        super().__init__(facet_list=facet_list)


class KinematicDeviations:
    """
    Store the kinematic deviations.

    Attributes
    ----------
    first_joint_translation_e : torch.Tensor | None
        The first joint translation in the east direction.
    first_joint_translation_n : torch.Tensor | None
        The first joint translation in the north direction.
    first_joint_translation_u : torch.Tensor | None
        The first joint translation in the up direction.
    first_joint_tilt_e : torch.Tensor | None
        The first joint tilt in the east direction.
    first_joint_tilt_n : torch.Tensor | None
        The first joint tilt in the north direction.
    first_joint_tilt_u : torch.Tensor | None
        The first joint tilt in the up direction.
    second_joint_translation_e : torch.Tensor | None
        The second joint translation in the east direction.
    second_joint_translation_n : torch.Tensor | None
        The second joint translation in the north direction.
    second_joint_translation_u : torch.Tensor | None
        The second joint translation in the up direction.
    second_joint_tilt_e : torch.Tensor | None
        The second joint tilt in the east direction.
    second_joint_tilt_n : torch.Tensor | None
        The second joint tilt in the north direction.
    second_joint_tilt_u : torch.Tensor | None
        The second joint tilt in the up direction.
    concentrator_translation_e : torch.Tensor | None
        The concentrator translation in the east direction.
    concentrator_translation_n : torch.Tensor | None
        The concentrator translation in the north direction.
    concentrator_translation_u : torch.Tensor | None
        The concentrator translation in the up direction.
    concentrator_tilt_e : torch.Tensor | None
        The concentrator tilt in the east direction.
    concentrator_tilt_n : torch.Tensor | None
        The concentrator tilt in the north direction.
    concentrator_tilt_u : torch.Tensor | None
        The concentrator tilt in the up direction.

    Methods
    -------
    create_kinematic_deviations_dict()
        Create a dictionary containing the configuration parameters for the kinematic deviations.
    """

    def __init__(
        self,
        first_joint_translation_e: torch.Tensor | None = None,
        first_joint_translation_n: torch.Tensor | None = None,
        first_joint_translation_u: torch.Tensor | None = None,
        first_joint_tilt_e: torch.Tensor | None = None,
        first_joint_tilt_n: torch.Tensor | None = None,
        first_joint_tilt_u: torch.Tensor | None = None,
        second_joint_translation_e: torch.Tensor | None = None,
        second_joint_translation_n: torch.Tensor | None = None,
        second_joint_translation_u: torch.Tensor | None = None,
        second_joint_tilt_e: torch.Tensor | None = None,
        second_joint_tilt_n: torch.Tensor | None = None,
        second_joint_tilt_u: torch.Tensor | None = None,
        concentrator_translation_e: torch.Tensor | None = None,
        concentrator_translation_n: torch.Tensor | None = None,
        concentrator_translation_u: torch.Tensor | None = None,
        concentrator_tilt_e: torch.Tensor | None = None,
        concentrator_tilt_n: torch.Tensor | None = None,
        concentrator_tilt_u: torch.Tensor | None = None,
    ) -> None:
        """
        Initialize the kinematic deviations.

        Parameters
        ----------
        first_joint_translation_e : torch.Tensor | None
            The first joint translation in the east direction.
        first_joint_translation_n : torch.Tensor | None
            The first joint translation in the north direction.
        first_joint_translation_u : torch.Tensor | None
            The first joint translation in the up direction.
        first_joint_tilt_e : torch.Tensor | None
            The first joint tilt in the east direction.
        first_joint_tilt_n : torch.Tensor | None
            The first joint tilt in the north direction.
        first_joint_tilt_u : torch.Tensor | None
            The first joint tilt in the up direction.
        second_joint_translation_e : torch.Tensor | None
            The second joint translation in the east direction.
        second_joint_translation_n : torch.Tensor | None
            The second joint translation in the north direction.
        second_joint_translation_u : torch.Tensor | None
            The second joint translation in the up direction.
        second_joint_tilt_e : torch.Tensor | None
            The second joint tilt in the east direction.
        second_joint_tilt_n : torch.Tensor | None
            The second joint tilt in the north direction.
        second_joint_tilt_u : torch.Tensor | None
            The second joint tilt in the up direction.
        concentrator_translation_e : torch.Tensor | None
            The concentrator translation in the east direction.
        concentrator_translation_n : torch.Tensor | None
            The concentrator translation in the north direction.
        concentrator_translation_u : torch.Tensor | None
            The concentrator translation in the up direction.
        concentrator_tilt_e : torch.Tensor | None
            The concentrator tilt in the east direction.
        concentrator_tilt_n : torch.Tensor | None
            The concentrator tilt in the north direction.
        concentrator_tilt_u : torch.Tensor | None
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


class KinematicConfig:
    """
    Store the configuration parameters for the kinematic.

    Attributes
    ----------
    type : str
        The type of kinematic used.
    initial_orientation : torch.Tensor
        The initial orientation of the kinematic configuration.
    deviations : KinematicDeviations | None
        The kinematic deviations.

    Methods
    -------
    create_kinematic_dict()
        Create a dictionary containing the configuration parameters for the kinematic.
    """

    def __init__(
        self,
        type: str,
        initial_orientation: torch.Tensor,
        deviations: KinematicDeviations | None = None,
    ) -> None:
        """
        Initialize the kinematic configuration.

        Parameters
        ----------
        type : str
            The type of kinematic used.
        initial_orientation : torch.Tensor
            The initial orientation of the kinematic configuration.
        deviations : KinematicDeviations | None
            The kinematic deviations.
        """
        self.type = type
        self.initial_orientation = initial_orientation
        self.deviations = deviations

    def create_kinematic_dict(self) -> dict[str, Any]:
        """
        Create a dictionary containing the configuration parameters for the kinematic.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the configuration parameters for the kinematic.
        """
        kinematic_dict: dict[str, Any] = {
            config_dictionary.kinematic_type: self.type,
            config_dictionary.kinematic_initial_orientation: self.initial_orientation,
        }
        if self.deviations is not None:
            kinematic_dict.update(
                {
                    config_dictionary.kinematic_deviations: self.deviations.create_kinematic_deviations_dict()
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
        type: str,
        initial_orientation: torch.Tensor,
        deviations: KinematicDeviations | None = None,
    ) -> None:
        """
        Initialize the kinematic prototype configuration.

        Parameters
        ----------
        type : str
            The type of kinematic used.
        initial_orientation : torch.Tensor
            The initial orientation of the kinematic configuration.
        deviations : KinematicDeviations | None
            The kinematic deviations.
        """
        super().__init__(
            type=type,
            initial_orientation=initial_orientation,
            deviations=deviations,
        )


class KinematicLoadConfig:
    """
    Store the configuration parameters for the kinematic when loaded in ``ARTIST``.

    Attributes
    ----------
    type : str
        The type of kinematic used.
    initial_orientation : torch.Tensor
        The initial orientation of the kinematic configuration.
    deviations : KinematicDeviations
        The kinematic deviations.
    """

    def __init__(
        self,
        type: str,
        initial_orientation: torch.Tensor,
        deviations: KinematicDeviations,
    ) -> None:
        """
        Initialize the kinematic configuration for loading in ``ARTIST``.

        Parameters
        ----------
        type : str
            The type of kinematic used.
        initial_orientation : torch.Tensor
            The initial orientation of the kinematic configuration.
        deviations : KinematicDeviations
            The kinematic deviations.
        """
        self.type = type
        self.initial_orientation = initial_orientation
        self.deviations = deviations


class ActuatorParameters:
    """
    Store the actuator parameters.

    Attributes
    ----------
    increment : torch.Tensor | None
        The increment for the actuator
    initial_stroke_length : torch.Tensor | None
        The initial stroke length.
    offset : torch.Tensor | None
        The initial actuator offset.
    pivot_radius : torch.Tensor | None
        The pivot radius of the considered joint.
    initial_angle : torch.Tensor | None
        The initial angle of the actuator.

    Methods
    -------
    create_actuator_parameters_dict()
        Create a dictionary containing the configuration parameters for the actuator parameters.
    """

    def __init__(
        self,
        increment: torch.Tensor | None = None,
        initial_stroke_length: torch.Tensor | None = None,
        offset: torch.Tensor | None = None,
        pivot_radius: torch.Tensor | None = None,
        initial_angle: torch.Tensor | None = None,
    ) -> None:
        """
        Initialize the actuator parameters.

        Parameters
        ----------
        increment : torch.Tensor | None
            The increment for the actuator
        initial_stroke_length : torch.Tensor | None
            The initial stroke length.
        offset : torch.Tensor | None
            The initial actuator offset.
        pivot_radius : torch.Tensor | None
            The pivot radius of the considered joint.
        initial_angle : torch.Tensor | None
            The initial angle of the actuator.
        """
        self.increment = increment
        self.initial_stroke_length = initial_stroke_length
        self.offset = offset
        self.pivot_radius = pivot_radius
        self.initial_angle = initial_angle

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
        if self.pivot_radius is not None:
            actuator_parameters_dict.update(
                {config_dictionary.actuator_pivot_radius: self.pivot_radius}
            )
        if self.initial_angle is not None:
            actuator_parameters_dict.update(
                {config_dictionary.actuator_initial_angle: self.initial_angle}
            )
        return actuator_parameters_dict


class ActuatorConfig:
    """
    Store the configuration parameters for the actuator.

    Attributes
    ----------
    key : str
        The name or descriptor of the actuator.
    type : str
        The type of actuator to use, e.g. linear or ideal.
    clockwise_axis_movement : bool
        Boolean indicating if the actuator operates in a clockwise manner.
    parameters : ActuatorParameters | None
        The parameters of the actuator

    Methods
    -------
    create_actuator_dict()
        Create a dictionary containing the configuration parameters for the actuator.
    """

    def __init__(
        self,
        key: str,
        type: str,
        clockwise_axis_movement: bool,
        parameters: ActuatorParameters | None = None,
    ) -> None:
        """
        Initialize the actuator configuration.

        Parameters
        ----------
        key : str
            The name or descriptor of the actuator.
        type : str
            The type of actuator to use, e.g. linear or ideal.
        clockwise_axis_movement : bool
            Boolean indicating if the actuator operates in a clockwise or counterclockwise manner.
        parameters : ActuatorParameters | None
            The parameters of the actuator.
        """
        self.key = key
        self.type = type
        self.clockwise_axis_movement = clockwise_axis_movement
        self.parameters = parameters

    def create_actuator_dict(self) -> dict[str, Any]:
        """
        Create a dictionary containing the actuator configuration.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the actuator configuration.
        """
        actuator_dict = {
            config_dictionary.actuator_type_key: self.type,
            config_dictionary.actuator_clockwise_axis_movement: self.clockwise_axis_movement,
        }
        if self.parameters is not None:
            actuator_dict.update(
                {
                    config_dictionary.actuator_parameters_key: self.parameters.create_actuator_parameters_dict()
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
            actuator_config.key: actuator_config.create_actuator_dict()
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
    actuators_prototype : ActuatorPrototypeConfig
        The prototype for the actuators.

    Methods
    -------
    create_prototype_dict()
        Create a dictionary containing the prototypes.
    """

    def __init__(
        self,
        surface_prototype: SurfacePrototypeConfig,
        kinematic_prototype: KinematicPrototypeConfig,
        actuators_prototype: ActuatorPrototypeConfig,
    ) -> None:
        """
        Initialize the prototype configuration.

        Parameters
        ----------
        surface_prototype : SurfacePrototypeConfig
            The prototype for the surface.
        kinematic_prototype : KinematicPrototypeConfig
            The prototype for the kinematic.
        actuators_prototype : ActuatorPrototypeConfig
            The prototype for the actuators.
        """
        self.surface_prototype = surface_prototype
        self.kinematic_prototype = kinematic_prototype
        self.actuators_prototype = actuators_prototype

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
            config_dictionary.actuators_prototype_key: self.actuators_prototype.create_actuator_list_dict(),
        }


class HeliostatConfig:
    """
    Store the configurations for a single heliostat.

    Attributes
    ----------
    name : str
        The name used to identify the heliostat in the HDF5 file.
    id : int
        The numerical ID of the heliostat.
    position : torch.Tensor
        The position of the heliostat.
    aim_point : torch.Tensor
        The position of the heliostat aim point.
    surface : SurfaceConfig | None
        An optional individual surface config for the heliostat.
    kinematic : KinematicConfig | None
        An optional kinematic config for the heliostat.
    actuators : ActuatorListConfig | None
        An optional actuator list config for the heliostat.

    Methods
    -------
    create_heliostat_config_dict()
        Create a dictionary containing the configuration parameters for a heliostat.
    """

    def __init__(
        self,
        name: str,
        id: int,
        position: torch.Tensor,
        aim_point: torch.Tensor,
        surface: SurfaceConfig | None = None,
        kinematic: KinematicConfig | None = None,
        actuators: ActuatorListConfig | None = None,
    ) -> None:
        """
        Initialize the single heliostat configuration.

        Parameters
        ----------
        name : str
            The name used to identify the heliostat in the HDF5 file.
        id : int
            The numerical ID of the heliostat.
        position : torch.Tensor
            The position of the heliostat.
        aim_point : torch.Tensor
            The position of the heliostat aim point.
        surface : SurfaceConfig | None
            An optional individual surface config for the heliostat.
        kinematic : KinematicConfig | None
            An optional kinematic config for the heliostat.
        actuators : ActuatorListConfig | None
            An optional actuator list config for the heliostat.
        """
        self.name = name
        self.id = id
        self.position = position
        self.aim_point = aim_point
        self.surface = surface
        self.kinematic = kinematic
        self.actuators = actuators

    def create_heliostat_config_dict(self) -> dict[str, Any]:
        """
        Create a dictionary containing the heliostat configuration parameters.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the heliostat configuration parameters.
        """
        heliostat_dict = {
            config_dictionary.heliostat_id: self.id,
            config_dictionary.heliostat_position: self.position,
            config_dictionary.heliostat_aim_point: self.aim_point,
        }
        if self.surface is not None:
            heliostat_dict.update(
                {
                    config_dictionary.heliostat_surface_key: self.surface.create_surface_dict()
                }
            )
        if self.kinematic is not None:
            heliostat_dict.update(
                {
                    config_dictionary.heliostat_kinematic_key: self.kinematic.create_kinematic_dict()
                }
            )
        if self.actuators is not None:
            heliostat_dict.update(
                {
                    config_dictionary.heliostat_actuator_key: self.actuators.create_actuator_list_dict()
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
            heliostat.name: heliostat.create_heliostat_config_dict()
            for heliostat in self.heliostat_list
        }
