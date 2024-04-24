from typing import Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from artist.scene import LightSource
from artist.util import utils


class DistortionsDataset(Dataset):
    """
    This class contains a data set of distortions based on the model of the light source.

    Attributes
    ----------
    distortions_u : torch.Tensor
        The distortions in the up direction.
    distortions_e : torch.Tensor
        The distortions in the east direction
    number_of_heliostats : int
        The number of heliostats in the scenario
    """

    def __init__(
        self,
        light_source: LightSource,
        number_of_points: int,
        number_of_heliostats: Optional[int] = 1,
        random_seed: Optional[int] = 7,
    ) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
        light_source : LightSource
            The light source used to model the distortions.
        number_of_points : int
            The number of points on the heliostat for which distortions are created.
        number_of_heliostats : Optional[int]
            The number of heliostats in the scenario.
        random_seed : Optional[int]
            The random seed used for generating the distortions.
        """
        self.number_of_heliostats = number_of_heliostats
        self.distortions_u, self.distortions_e = light_source.get_distortions(
            number_of_points=number_of_points,
            number_of_heliostats=number_of_heliostats,
            random_seed=random_seed,
        )

    def __len__(self) -> int:
        """
        Calculate the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.
        """
        return self.distortions_u.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select an item from the dataset.

        Parameters
        ----------
        idx : int
            The index of the item to select.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The distortions in the up and east direction for the given index.
        """
        return (
            self.distortions_u[idx, :],
            self.distortions_e[idx, :],
        )


class HeliostatRayTracer:
    """This class contains the functionality for heliostat raytracing."""

    @staticmethod
    def scatter_rays(
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
        # Use the cosine between the ray directions and the normals to calculate the relative distribution strength of
        # the incoming rays
        relative_distribution_strengths = ray_directions @ plane_normal
        assert (
            torch.abs(relative_distribution_strengths) >= epsilon
        ).all(), "No intersection or line is within plane."
        # Calculate the final distribution strengths
        distribution_strengths = (
            (plane_point - surface_points)
            @ plane_normal
            / relative_distribution_strengths
        )

        return surface_points + ray_directions * distribution_strengths.unsqueeze(-1)

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
