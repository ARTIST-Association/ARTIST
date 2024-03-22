from typing import Any, Dict, Tuple, Union

import h5py
import torch
from typing_extensions import Self

from artist.scene.light_source import ALightSource
from artist.util import config_dictionary, utils


class Sun(ALightSource):
    """
    This class implements the sun as a light source.

    Attributes
    ----------
    distribution_parameters : Dict[str, Any]
        Parameters of the distribution used to model the sun.
    ray_count : int
        The number of sent-out rays sampled from the sun distribution.

    Methods
    -------
    from_hdf5()
        Class method to initialize heliostat from an h5 file.
    sample()
        Sample rays from a given distribution.
    scatter_rays()
        Scatter the reflected rays around the preferred ray_direction.
    line_plane_intersections()
        Compute line-plane intersections of ray directions and the (receiver) plane.
    get_preferred_reflection_direction()
        Reflect incoming rays according to a normal vector.
    sample_bitmap()
        Sample a bitmap (flux density distribution of the reflected rays on the receiver).
    normalize_bitmap()
        Normalize a bitmap.

    See Also
    --------
    :class:ALightSource : The parent class.
    """

    def __init__(
        self,
        distribution_parameters: Dict[str, Any] = dict(
            distribution_type="normal", mean=0.0, covariance=4.3681e-06
        ),
        ray_count: int = 200,
    ) -> None:
        """
        Initialize the sun as a light source.

        Parameters
        ----------
        distribution_parameters
            Parameters of the distribution used to model the sun.
        ray_count : int
            The number of sent-out rays sampled from the sun distribution.

        Raises
        ------
        Union[ValueError, NotImplementedError]
            If the specified distribution type is unknown.
        """
        super().__init__()

        self.distribution_parameters = distribution_parameters
        self.ray_count = ray_count

        assert (
            self.distribution_parameters["distribution_type"] == "normal"
        ), "Unknown sunlight distribution type."

        if self.distribution_parameters["distribution_type"] == "normal":
            mean = torch.tensor(
                [
                    self.distribution_parameters["mean"],
                    self.distribution_parameters["mean"],
                ],
                dtype=torch.float,
            )
            covariance = torch.tensor(
                [
                    [self.distribution_parameters["covariance"], 0],
                    [0, self.distribution_parameters["covariance"]],
                ],
                dtype=torch.float,
            )

            self.distribution = torch.distributions.MultivariateNormal(mean, covariance)

    @classmethod
    def from_hdf5(cls, config_file: h5py.File) -> Self:
        """
        Class method that initializes a sun from an hdf5 file.

        Parameters
        ----------
        config_file : h5py.File
            The hdf5 file containing the information about the sun.

        Returns
        -------
        Sun
            A sun initialized from an hdf5 file.
        """
        distribution_parameters = {
            config_dictionary.sun_distribution_type: config_file[
                config_dictionary.sun_prefix
            ][config_dictionary.sun_distribution_parameters][
                config_dictionary.sun_distribution_type
            ][()].decode("utf-8"),
            config_dictionary.sun_mean: config_file[config_dictionary.sun_prefix][
                config_dictionary.sun_distribution_parameters
            ][config_dictionary.sun_mean][()],
            config_dictionary.sun_covariance: config_file[config_dictionary.sun_prefix][
                config_dictionary.sun_distribution_parameters
            ][config_dictionary.sun_covariance][()],
        }
        num_rays = config_file[config_dictionary.sun_prefix][
            config_dictionary.sun_number_of_rays
        ][()]

        return cls(distribution_parameters=distribution_parameters, ray_count=num_rays)

    def sample(
        self,
        num_preferred_ray_directions: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample rays from a given distribution.

        Parameters
        ----------
        num_preferred_ray_directions : int
            The number of rays on the heliostat.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The distortion in north and up direction.

        Raises
        ------
        ValueError
            If the distribution type is not valid, currently only the normal distribution is implemented.
        """
        if (
            self.distribution_parameters[config_dictionary.sun_distribution_type]
            == config_dictionary.sun_distribution_is_normal
        ):
            distortions_n, distortions_u = self.distribution.sample(
                (self.ray_count, num_preferred_ray_directions),
            ).permute(2, 0, 1)
            return distortions_n, distortions_u
        else:
            raise ValueError("Unknown light distribution type.")

    def scatter_rays(
        self,
        ray_directions: torch.Tensor,
        distortion_u: torch.Tensor,
        distortion_e: torch.Tensor,
    ) -> torch.Tensor:
        """
        Scatter the reflected rays around the preferred ray direction.

        Parameters
        ----------
        ray_directions : torch.Tensor
            The preferred ray direction.
        distortion_u : torch.Tensor
            The distortions in up direction (angles for scattering).
        distortion_e : torch.Tensor
            The distortions in east direction (angles for scattering).

        Returns
        -------
        torch.Tensor
            Scattered rays around the preferred direction.
        """
        ray_directions = ray_directions[:, :3] / torch.linalg.norm(
            ray_directions[:, :3], dim=1
        ).unsqueeze(-1)
        ray_directions = torch.cat(
            (ray_directions, torch.zeros(ray_directions.size(0), 1)), dim=1
        )

        scattered_rays = utils.rotate_distortions(
            u=distortion_u, e=distortion_e
        ) @ ray_directions.unsqueeze(-1)

        return scattered_rays.squeeze(-1)

    @staticmethod
    def line_plane_intersections(
        plane_normal: torch.Tensor,
        plane_point: torch.Tensor,
        ray_directions: torch.Tensor,
        surface_points: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> torch.Tensor:
        """
        Compute line-plane intersections of ray directions and the (receiver) plane.

        Parameters
        ----------
        plane_normal : torch.Tensor
            The normal vector of the intersecting plane (normal vector of the receiver).
        plane_point : torch.Tensor
            Point on the plane (center point of the receiver).
        ray_directions : torch.Tensor
            The direction of the reflected sunlight.
        surface_points : torch.Tensor
            Points on which the rays are to be reflected.
        epsilon : float
            A small value corresponding to the upper limit.

        Returns
        -------
        torch.Tensor
            The intersections of the lines and plane.

        Raises
        ------
        RuntimeError
            When there are no intersections between the line and the plane.
        """
        ndotu = ray_directions @ plane_normal
        assert (
            torch.abs(ndotu) >= epsilon
        ).all(), "No intersection or line is within plane."
        ds = (plane_point - surface_points) @ plane_normal / ndotu

        return surface_points + ray_directions * ds.unsqueeze(-1)

    @staticmethod
    def get_preferred_reflection_direction(
        rays: torch.Tensor, normals: torch.Tensor
    ) -> torch.Tensor:
        """
        Reflect incoming rays according to a normal vector.

        Parameters
        ----------
        rays : torch.Tensor
            The incoming rays (from the sun) to be reflected.
        normals : torch.Tensor
            Surface normals on which the rays are reflected.

        Returns
        -------
        torch.Tensor
            The preferred direction of the reflected rays for each heliostat surface normal.
        """
        return rays - 2 * utils.batch_dot(rays, normals) * normals

    @staticmethod
    def sample_bitmap(
        dx_ints: torch.Tensor,
        dy_ints: torch.Tensor,
        indices: torch.Tensor,
        plane_x: float,
        plane_y: float,
        bitmap_height: int,
        bitmap_width: int,
    ) -> torch.Tensor:
        # TODO : Complete docstring.
        """
        Sample a bitmap (flux density distribution of the reflected rays on the receiver).

        Parameters
        ----------
        dx_ints : torch.Tensor

        dy_ints : torch.Tensor

        indices : torch.Tensor

        plane_x : float
            x dimension of the receiver plane.
        plane_y : float
            y dimension of the receiver plane.
        bitmap_height : int
            Resolution of the resulting bitmap (x direction) -> height
        bitmap_width : int
            Resolution of the resulting bitmap (y direction) -> width

        Returns
        -------
        torch.Tensor
            The flux density distribution of the reflected rays on the receiver
        """
        x_ints = dx_ints[indices] / plane_x * bitmap_height
        y_ints = dy_ints[indices] / plane_y * bitmap_width

        # We assume a continuously positioned value in-between four
        # discretely positioned pixels, similar to this:
        #
        # 4|3
        # -.-
        # 1|2
        #
        # where the numbers are the four neighboring, discrete pixels, the
        # "-" and "|" are the discrete pixel borders, and the "." is the
        # continuous value anywhere in-between the four pixels we sample.
        # That the "." may be anywhere in-between the four pixels is not
        # shown in the ASCII diagram, but is important to keep in mind.

        # The lower-valued neighboring pixels (for x this corresponds to 1
        # and 4, for y to 1 and 2).
        x_inds_low = x_ints.floor().long()
        y_inds_low = y_ints.floor().long()
        # The higher-valued neighboring pixels (for x this corresponds to 2
        # and 3, for y to 3 and 4).
        x_inds_high = x_inds_low + 1
        y_inds_high = y_inds_low + 1

        # When distributing the continuously positioned value/intensity to
        # the discretely positioned pixels, we give the corresponding
        # "influence" of the value to each neighbor. Here, we calculate this
        # influence for each neighbor.

        # x-value influence in 1 and 4
        x_ints_low = x_inds_high - x_ints
        # y-value influence in 1 and 2
        y_ints_low = y_inds_high - y_ints
        # x-value influence in 2 and 3
        x_ints_high = x_ints - x_inds_low
        # y-value influence in 3 and 4
        y_ints_high = y_ints - y_inds_low
        del x_ints
        del y_ints

        # We now calculate the distributed intensities for each neighboring
        # pixel and assign the correctly ordered indices to the intensities
        # so we know where to position them. The numbers correspond to the
        # ASCII diagram above.
        x_inds_1 = x_inds_low
        y_inds_1 = y_inds_low
        ints_1 = x_ints_low * y_ints_low

        x_inds_2 = x_inds_high
        y_inds_2 = y_inds_low
        ints_2 = x_ints_high * y_ints_low
        del y_inds_low
        del y_ints_low

        x_inds_3 = x_inds_high
        y_inds_3 = y_inds_high
        ints_3 = x_ints_high * y_ints_high
        del x_inds_high
        del x_ints_high

        x_inds_4 = x_inds_low
        y_inds_4 = y_inds_high
        ints_4 = x_ints_low * y_ints_high
        del x_inds_low
        del y_inds_high
        del x_ints_low
        del y_ints_high

        # Combine all indices and intensities in the correct order.
        x_inds = torch.hstack([x_inds_1, x_inds_2, x_inds_3, x_inds_4]).long().ravel()
        del x_inds_1
        del x_inds_2
        del x_inds_3
        del x_inds_4

        y_inds = torch.hstack([y_inds_1, y_inds_2, y_inds_3, y_inds_4]).long().ravel()
        del y_inds_1
        del y_inds_2
        del y_inds_3
        del y_inds_4

        ints = torch.hstack([ints_1, ints_2, ints_3, ints_4]).ravel()
        del ints_1
        del ints_2
        del ints_3
        del ints_4

        # For distribution, we regard even those neighboring pixels that are
        # _not_ part of the image. That is why here, we set up a mask to
        # choose only those indices that are actually in the bitmap (i.e. we
        # prevent out-of-bounds access).
        indices = (
            (0 <= x_inds)
            & (x_inds < bitmap_height)
            & (0 <= y_inds)
            & (y_inds < bitmap_width)
        )

        # Flux density map for heliostat field
        total_bitmap = torch.zeros(
            [bitmap_height, bitmap_width],
            dtype=dx_ints.dtype,
            device=dx_ints.device,
        )
        # Add up all distributed intensities in the corresponding indices.
        total_bitmap.index_put_(
            (x_inds[indices], y_inds[indices]),
            ints[indices],
            accumulate=True,
        )

        return total_bitmap

    @staticmethod
    def normalize_bitmap(
        bitmap: torch.Tensor,
        total_intensity: Union[float, torch.Tensor],
        plane_x: float,
        plane_y: float,
    ) -> torch.Tensor:
        """
        Normalize a bitmap.

        Parameters
        ----------
        bitmap : torch.Tensor
            The bitmap to be normalized.
        total_intensity : Union[float, torch.Tensor]
            The total intensity of the bitmap.
        plane_x : float
            x dimension of the receiver plane.
        plane_y : float
            y dimension of the receiver plane.

        Returns
        -------
        The normalized bitmap.
        """
        bitmap_height = bitmap.shape[0]
        bitmap_width = bitmap.shape[1]

        plane_area = plane_x * plane_y
        num_pixels = bitmap_height * bitmap_width
        plane_area_per_pixel = plane_area / num_pixels

        return bitmap / (total_intensity * plane_area_per_pixel)
