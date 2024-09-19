from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from artist.scenario import Scenario

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from artist.scene import LightSource
from artist.util import utils

from . import raytracing_utils
from .rays import Rays


class DistortionsDataset(Dataset):
    """
    A dataset of distortions based on the model of the light source.

    Attributes
    ----------
    distortions_u : torch.Tensor
        The distortions in the up direction.
    distortions_e : torch.Tensor
        The distortions in the east direction.
    number_of_heliostats : int
        The number of heliostats in the scenario.
    """

    def __init__(
        self,
        light_source: LightSource,
        number_of_points: int,
        number_of_facets: int = 4,
        number_of_heliostats: int = 1,
        random_seed: int = 7,
    ) -> None:
        """
        Initialize the dataset.

        This class implements a custom dataset according to the ``torch`` interface. The content of this
        dataset is the distortions. The distortions are used in our version of "heliostat"-tracing to
        indicate how each incoming ray must be multiplied and scattered on the heliostat. According to
        ``torch``, this dataset must implement a function to return the length of the dataset and one function
        to retrieve an element through an index.

        Parameters
        ----------
        light_source : LightSource
            The light source used to model the distortions.
        number_of_points : int
            The number of points on the heliostat for which distortions are created.
        number_of_facets : int
            The number of facets per heliostat (default: 4).
        number_of_heliostats : int
            The number of heliostats in the scenario (default: 1).
        random_seed : int
            The random seed used for generating the distortions (default: 7).
        """
        self.number_of_heliostats = number_of_heliostats
        self.distortions_u, self.distortions_e = light_source.get_distortions(
            number_of_points=number_of_points,
            number_of_facets=number_of_facets,
            number_of_heliostats=number_of_heliostats,
            random_seed=random_seed,
        )

    def __len__(self) -> int:
        """
        Calculate the length of the dataset, i.e., the number of items contained.

        Returns
        -------
        int
            The length of the dataset.
        """
        return self.distortions_u.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select an item from the dataset.

        Parameters
        ----------
        idx : int
            The index of the item to select.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The distortions in the up and east direction for the given index.
        """
        return (
            self.distortions_u[idx, :],
            self.distortions_e[idx, :],
        )


class HeliostatRayTracer:
    """
    Implement the functionality for heliostat raytracing.

    Attributes
    ----------
    heliostat : Heliostat
        The heliostat considered for raytracing.
    receiver : Receiver
        The receiver considered for raytracing.
    world_size : int
        The world size for MPI, i.e., the overall number of processors / ranks.
    rank : int
        The rank, i.e., individual process ID, for MPI.
    number_of_surface_points : int
        The number of surface points on the heliostat.
    distortions_dataset : DistortionsDataset
        The dataset containing the distortions for ray scattering.
    distortions_loader : DataLoader
        The dataloader that loads the distortions.

    Methods
    -------
    trace_rays()
        Perform heliostat raytracing.
    scatter_rays()
        Scatter the reflected rays around the preferred ray direction.
    sample_bitmap()
        Sample a bitmap (flux density distribution of the reflected rays on the receiver).
    normalize_bitmap()
        Normalize a bitmap.
    """

    def __init__(
        self,
        scenario: "Scenario",
        world_size: int = 1,
        rank: int = 0,
        batch_size: int = 1,
        random_seed: int = 7,
        shuffle: bool = False,
    ) -> None:
        """
        Initialize the heliostat raytracer.

        "Heliostat"-tracing is one kind of raytracing applied in ARTIST. For this kind of raytracing,
        the rays are initialized on the heliostat. The rays originate in the discrete surface points.
        There they are multiplied, distorted, and scattered, and then they are sent to the receiver. Letting
        the rays originate on the heliostat drastically reduces the number of rays that need to be traced.

        Parameters
        ----------
        scenario : Scenario
            The scenario used to perform raytracing.
        world_size : int
            The world size for MPI (default: 1).
        rank : int
            The rank for MPI (default: 0).
        batch_size : int
            The batch size used for raytracing (default: 1).
        random_seed : int
            The random seed used for generating the distortions (default: 7).
        shuffle : bool
            A boolean flag indicating whether to shuffle the data (default: False).
        """
        self.heliostat = scenario.heliostats.heliostat_list[0]
        self.receiver = scenario.receivers.receiver_list[0]
        self.world_size = world_size
        self.rank = rank
        self.number_of_surface_points = (
            self.heliostat.current_aligned_surface_points.size(1)
        )
        # Create distortions dataset.
        self.distortions_dataset = DistortionsDataset(
            light_source=scenario.light_sources.light_source_list[0],
            number_of_points=self.number_of_surface_points,
            random_seed=random_seed,
        )
        # Create distributed sampler.
        distortions_sampler = DistributedSampler(
            dataset=self.distortions_dataset,
            shuffle=shuffle,
            num_replicas=self.world_size,
            rank=self.rank,
        )
        # Create dataloader.
        self.distortions_loader = DataLoader(
            self.distortions_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=distortions_sampler,
        )

    def trace_rays(self, incident_ray_direction: torch.Tensor) -> torch.Tensor:
        """
        Perform heliostat raytracing.

        Scatter the rays according to the distortions, calculate the line plane intersection, and calculate the
        resulting bitmap on the receiver.

        Returns
        -------
        torch.Tensor
            The resulting bitmap.
        """
        final_bitmap = torch.zeros(
            (self.receiver.resolution_e, self.receiver.resolution_u)
        )

        self.heliostat.set_preferred_reflection_direction(rays=-incident_ray_direction)

        for batch_u, batch_e in self.distortions_loader:
            rays = self.scatter_rays(
                batch_u,
                batch_e,
            )

            intersections = raytracing_utils.line_plane_intersections(
                ray_directions=rays.ray_directions,
                plane_normal_vectors=self.receiver.normal_vector,
                plane_center=self.receiver.position_center,
                points_at_ray_origin=self.heliostat.current_aligned_surface_points,
            )

            dx_ints = (
                intersections[:, :, :, 0]
                + self.receiver.plane_e / 2
                - self.receiver.position_center[0]
            )
            dy_ints = (
                intersections[:, :, :, 2]
                + self.receiver.plane_u / 2
                - self.receiver.position_center[2]
            )

            indices = (
                (-1 <= dx_ints)
                & (dx_ints < self.receiver.plane_e + 1)
                & (-1 <= dy_ints)
                & (dy_ints < self.receiver.plane_u + 1)
            )

            total_bitmap = self.sample_bitmap(
                dx_ints,
                dy_ints,
                indices,
            )

            final_bitmap += total_bitmap

        return final_bitmap

    def scatter_rays(
        self,
        distortion_u: torch.Tensor,
        distortion_e: torch.Tensor,
    ) -> Rays:
        """
        Scatter the reflected rays around the preferred ray direction.

        Parameters
        ----------
        distortion_u : torch.Tensor
            The distortions in up direction (angles for scattering).
        distortion_e : torch.Tensor
            The distortions in east direction (angles for scattering).

        Returns
        -------
        Rays
            Scattered rays around the preferred direction.
        """
        ray_directions = self.heliostat.preferred_reflection_direction[
            :, :, :3
        ] / torch.linalg.norm(
            self.heliostat.preferred_reflection_direction[:, :, :3],
            ord=2,
            dim=-1,
            keepdim=True,
        )

        ray_directions = torch.cat(
            (ray_directions, torch.zeros(4, ray_directions.size(1), 1)), dim=-1
        )

        rotations = utils.rotate_distortions(u=distortion_u, e=distortion_e)

        scattered_rays = (rotations @ ray_directions.unsqueeze(-1)).squeeze(-1)

        return Rays(
            ray_directions=scattered_rays,
            ray_magnitudes=torch.ones(
                scattered_rays.size(dim=0),
                scattered_rays.size(dim=1),
                scattered_rays.size(dim=2),
            ),
        )

    def sample_bitmap(
        self,
        dx_ints: torch.Tensor,
        dy_ints: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample a bitmap (flux density distribution of the reflected rays on the receiver).

        Parameters
        ----------
        dx_ints : torch.Tensor
            x position of intersection with receiver of shape (N, 1) where N is the resolution of the receiver along
            the x-axis.
        dy_ints : torch.Tensor
            y position of intersection with receiver of shape (N, 1) where N is the resolution of the receiver along
            the y-axis.
        indices : torch.Tensor
            Index of the pixel.

        Returns
        -------
        torch.Tensor
            The flux density distribution of the reflected rays on the receiver.
        """
        x_ints = dx_ints[indices] / self.receiver.plane_e * self.receiver.resolution_e
        y_ints = dy_ints[indices] / self.receiver.plane_u * self.receiver.resolution_u

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
            & (x_inds < self.receiver.resolution_e)
            & (0 <= y_inds)
            & (y_inds < self.receiver.resolution_u)
        )

        # Flux density map for heliostat field
        total_bitmap = torch.zeros(
            [self.receiver.resolution_e, self.receiver.resolution_u],
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

    def normalize_bitmap(
        self,
        bitmap: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normalize a bitmap.

        Parameters
        ----------
        bitmap : torch.Tensor
            The bitmap to be normalized.

        Returns
        -------
        torch.Tensor
            The normalized bitmap.
        """
        bitmap_height = bitmap.shape[0]
        bitmap_width = bitmap.shape[1]

        plane_area = self.receiver.plane_e * self.receiver.plane_u
        num_pixels = bitmap_height * bitmap_width
        plane_area_per_pixel = plane_area / num_pixels

        return bitmap / (
            self.distortions_dataset.distortions_u.numel() * plane_area_per_pixel
        )
