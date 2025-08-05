import torch


class Rays:
    """
    Model rays used for ray tracing that have a direction vector and magnitude.

    Attributes
    ----------
    ray_directions : torch.Tensor
        The direction of the rays.
        Tensor of shape [number_of_active_heliostats, number_of_rays, number_of_combined_surface_normals_all_facets, 4].
    ray_magnitudes : torch.Tensor
        The magnitudes of the rays.
        Tensor of shape [number_of_active_heliostats, number_of_rays, number_of_combined_surface_normals_all_facets].
    """

    def __init__(
        self, ray_directions: torch.Tensor, ray_magnitudes: torch.Tensor
    ) -> None:
        """
        Initialize the ``Rays`` class.

        The rays in ``ARTIST`` have a direction vector and a magnitude. They are used for ray tracing.
        The direction vector determines the direction of the rays, i.e., the path they are taking through space.
        The magnitude is important for considering atmospheric losses and cloud coverage. If a ray
        travels through a cloud, the magnitude changes.

        Parameters
        ----------
        ray_directions : torch.Tensor
            The direction of the rays.
            Tensor of shape [number_of_active_heliostats, number_of_rays, number_of_combined_surface_normals_all_facets, 4].
        ray_magnitudes : torch.Tensor
            The magnitudes of the rays.
            Tensor of shape [number_of_active_heliostats, number_of_rays, number_of_combined_surface_normals_all_facets].
        
        Raises
        ------
        ValueError
            If the shapes of the ray directions does not match the shapes of the ray magnitudes.
        """
        if (
            ray_directions.shape[:-1] == ray_magnitudes.shape
            and ray_directions.shape[-1] == 4
        ):
            self.ray_directions = ray_directions
            self.ray_magnitudes = ray_magnitudes
        else:
            raise ValueError("Ray directions and magnitudes have incompatible sizes!")
