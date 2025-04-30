import logging
from typing import Iterator, Union

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from artist.field.heliostat_group import HeliostatGroup
from artist.field.tower_target_area import TargetArea
from artist.scene import LightSource
from artist.util import utils
from artist.util.scenario import Scenario

from . import raytracing_utils
from .rays import Rays

log = logging.getLogger(__name__)
"""A logger for the heliostat ray tracer."""


class DistortionsDataset(Dataset):
    """
    A dataset of distortions based on the model of the light source.

    Attributes
    ----------
    distortions_u : torch.Tensor
        The distortions in the up direction.
    distortions_e : torch.Tensor
        The distortions in the east direction.
    """

    def __init__(
        self,
        light_source: LightSource,
        number_of_points_per_heliostat: int,
        number_of_heliostats: int,
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
        number_of_points_per_heliostat : int
            The number of points on the heliostats for which distortions are created.
        number_of_heliostats : int
            The number of heliostats in the scenario.
        random_seed : int
            The random seed used for generating the distortions (default is 7).
        """
        self.distortions_u, self.distortions_e = light_source.get_distortions(
            number_of_points=number_of_points_per_heliostat,
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
        torch.Tensor
            The distortions in the up direction for the given index.
        torch.Tensor
            The distortions in the east direction for the given index.
        """
        return (
            self.distortions_u[idx],
            self.distortions_e[idx],
        )


class RestrictedDistributedSampler(Sampler):
    """
    Initializes a custom distributed sampler.

    The ``DistributedSampler`` from PyTorch replicates samples if the size of the dataset
    is smaller than the world size, to assign data to each rank. This custom sampler
    can leave some ranks idle if the dataset is not large enough to distribute data to
    each rank. Replicated samples would mean replicated rays that physically do not exist.

    Attributes
    ----------
    number_of_samples : int
        The number of samples in the dataset.
    world_size : int
        The world size or total number of processes.
    rank : int
        The rank of the current process.
    number_of_active_ranks : int
        The number of processes that will receive data.
    number_of_samples_per_rank : int
        The number of samples per rank.

    See Also
    --------
    :class:`torch.utils.data.Sampler` : Reference to the parent class.
    """

    def __init__(
        self,
        number_of_samples: int,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        """
        Set up a custom distributed sampler to assign data to each rank or leave them idle.

        Parameters
        ----------
        number_of_samples : int
            The length of the dataset or total number of samples.
        world_size : int
            The world size or total number of processes (default is 1).
        rank : int
            The rank of the current process (default is 0).
        """
        super().__init__()
        self.number_of_samples = number_of_samples
        self.world_size = world_size
        self.rank = rank

        # Adjust num_replicas if dataset is smaller than world_size.
        self.number_of_active_ranks = min(self.number_of_samples, self.world_size)
        if self.rank == 0:
            active_ranks_string = ", ".join(
                str(i) for i in range(self.number_of_active_ranks)
            )
            log.info(
                f"The ray tracer found {self.number_of_samples} set(s) of ray-samples to parallelize over. As {self.world_size} processes exist, the following the ranks: [{active_ranks_string}] will receive data, while all others (if more exist) are left idle."
            )

        # Only assign data to first active ranks.
        self.number_of_samples_per_rank = (
            self.number_of_samples // self.number_of_active_ranks
            if self.rank < self.number_of_active_ranks
            else 0
        )

    def __iter__(self) -> Iterator[int]:
        """
        Generate a sequence of indices for the current rank's portion of the dataset.

        Returns
        -------
        Iterator[int]
            An iterator over indices for the current rank.
        """
        rank_indices = []
        for i in range(self.rank, self.number_of_samples, self.world_size):
            rank_indices.append(i)

        return iter(rank_indices)


class HeliostatRayTracer:
    """
    Implement the functionality for heliostat ray tracing.

    Attributes
    ----------
    scenario : Scenario
        The scenario used to perform ray tracing.
    world_size : int
        The world size i.e., the overall number of processes.
    rank : int
        The rank, i.e., individual process ID.
    batch_size : int
        The amount of samples (Heliostats) processed parallel within a single rank.
    number_of_surface_points_per_heliostat : int
        The number of surface points on a single heliostat.
    distortions_dataset : DistortionsDataset
        The dataset containing the distortions for ray scattering.
    distortions_sampler : RestrictedDistributedSampler
        The distortion sampler.
    distortions_loader : DataLoader
        The dataloader that loads the distortions.
    bitmap_resolution_e : int
        The resolution of the bitmap in the east dimension.
    bitmap_resolution_u : int
        The resolution of the bitmap in the up dimension.

    Methods
    -------
    trace_rays()
        Perform heliostat ray tracing.
    scatter_rays()
        Scatter the reflected rays around the preferred ray directions for each heliostat.
    sample_bitmap()
        Sample a bitmap (flux density distribution) of the reflected rays on the target area.
    normalize_bitmap()
        Normalize a bitmap.
    """

    def __init__(
        self,
        heliostat_group: HeliostatGroup,
        light_source: LightSource,
        world_size: int = 1,
        rank: int = 0,
        batch_size: int = 1,
        random_seed: int = 7,
        bitmap_resolution_e: int = 256,
        bitmap_resolution_u: int = 256,
    ) -> None:
        """
        Initialize the heliostat ray tracer.

        "Heliostat"-tracing is one kind of ray tracing applied in ARTIST. For this kind of ray tracing,
        the rays are initialized on the heliostat. The rays originate in the discrete surface points.
        There they are multiplied, distorted, and scattered, and then they are sent to the target area.
        Letting the rays originate on the heliostat drastically reduces the number of rays that need
        to be traced.

        Parameters
        ----------
        number_of_heliostats : int
            The total number of heliostats to be ray traced.
        number_of_surface_points_per_heliostat : int
            The number of surface points per heliostat. 
        world_size : int
            The world size i.e., the overall number of processes (default is 1).
        rank : int
            The rank, i.e., individual process ID (default is 0).
        batch_size : int
            The amount of samples (Heliostats) processed parallel within a single rank (default is 1).
        random_seed : int
            The random seed used for generating the distortions (default is 7).
        bitmap_resolution_e : int
            The resolution of the bitmap in the east dimension (default is 256).
        bitmap_resolution_u : int
            The resolution of the bitmap in the up dimension (default is 256).
        """
        self.world_size = world_size
        self.rank = rank
        self.batch_size = batch_size

        self.heliostat_group = heliostat_group
        self.light_source = light_source

        # Create distortions dataset.
        self.distortions_dataset = DistortionsDataset(
            light_source=self.light_source,
            number_of_points_per_heliostat=self.heliostat_group.surface_points.shape[1],
            number_of_heliostats=self.heliostat_group.number_of_heliostats,
            random_seed=random_seed,
        )
        # Create restricted distributed sampler.
        self.distortions_sampler = RestrictedDistributedSampler(
            number_of_samples=len(self.distortions_dataset),
            world_size=self.world_size,
            rank=self.rank,
        )
        # Create dataloader.
        self.distortions_loader = DataLoader(
            self.distortions_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=self.distortions_sampler,
        )

        self.bitmap_resolution_e = bitmap_resolution_e
        self.bitmap_resolution_u = bitmap_resolution_u

    def trace_rays(
        self,
        incident_ray_direction: torch.Tensor,
        target_area: TargetArea,
        device: Union[torch.device, str] = "cuda",
    ) -> torch.Tensor:
        """
        Perform heliostat ray tracing.

        Scatter the rays according to the distortions, calculate the intersection with the target plane,
        and sample the resulting bitmap on the target area.

        Parameters
        ----------
        incident_ray_direction : torch.Tensor
            The direction of the incident ray as seen from the heliostat.
        target_area : TargetArea
            The target area used to sample the bitmap on.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Raises
        ------
        ValueError
            If not all heliostats used for ray tracing have been aligned.

        Returns
        -------
        torch.Tensor
            The resulting bitmap.
        """
        device = torch.device(device)

        final_bitmap = torch.zeros(
            (self.bitmap_resolution_u, self.bitmap_resolution_e), device=device
        )

        if not torch.all(self.heliostat_group.aligned_heliostats == 1.0):
            raise ValueError("Not all heliostats have been aligned.")

        self.heliostat_group.preferred_reflection_directions = raytracing_utils.reflect(
            incoming_ray_direction=incident_ray_direction,
            reflection_surface_normals=self.heliostat_group.current_aligned_surface_normals,
        )

        for batch_index, (batch_u, batch_e) in enumerate(self.distortions_loader):
            sampler_indices = list(self.distortions_sampler)

            heliostat_indices_per_batch = sampler_indices[
                batch_index * self.batch_size : (batch_index + 1) * self.batch_size
            ]

            rays = self.scatter_rays(
                preferred_reflection_directions=self.heliostat_group.preferred_reflection_directions,
                distortion_u=batch_u,
                distortion_e=batch_e,
                heliostat_indices=heliostat_indices_per_batch,
                device=device
            )

            intersections, absolute_intensities = (
                raytracing_utils.line_plane_intersections(
                    rays=rays,
                    plane_normal_vector=target_area.normal_vector,
                    plane_center=target_area.center,
                    points_at_ray_origin=self.heliostat_group.current_aligned_surface_points[
                        heliostat_indices_per_batch
                    ],
                )
            )

            dx_intersections = (
                intersections[:, :, :, 0]
                + target_area.plane_e / 2
                - target_area.center[0]
            )
            dy_intersections = (
                intersections[:, :, :, 2]
                + target_area.plane_u / 2
                - target_area.center[2]
            )

            intersection_indices = (
                (-1 <= dx_intersections)
                & (dx_intersections < target_area.plane_e + 1)
                & (-1 <= dy_intersections)
                & (dy_intersections < target_area.plane_u + 1)
            )

            total_bitmap = self.sample_bitmap(
                target_area=target_area,
                dx_intersections=dx_intersections,
                dy_intersections=dy_intersections,
                intersection_indices=intersection_indices,
                absolute_intensities=absolute_intensities,
                device=device,
            )

            final_bitmap = final_bitmap + total_bitmap

        return final_bitmap

    def scatter_rays(
        self,
        preferred_reflection_directions: torch.Tensor,
        distortion_u: torch.Tensor,
        distortion_e: torch.Tensor,
        heliostat_indices: list,
        device: Union[torch.device, str] = "cuda",
    ) -> Rays:
        """
        Scatter the reflected rays around the preferred ray directions for each heliostat.

        Parameters
        ----------
        distortion_u : torch.Tensor
            The distortions in up direction (angles for scattering).
        distortion_e : torch.Tensor
            The distortions in east direction (angles for scattering).
        heliostat_indices : list
            The indices of the heliostats considered in the current batch.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        Rays
            Scattered rays around the preferred directions.
        """
        device = torch.device(device)

        rotations = utils.rotate_distortions(
            u=distortion_u, e=distortion_e, device=device
        )

        scattered_rays = (
            rotations
            @ preferred_reflection_directions[
                heliostat_indices, :, :
            ]
            .unsqueeze(1)
            .unsqueeze(-1)
        ).squeeze(-1)

        return Rays(
            ray_directions=scattered_rays,
            ray_magnitudes=torch.ones(scattered_rays.shape[:-1], device=device),
        )

    def sample_bitmap(
        self,
        target_area: TargetArea,
        dx_intersections: torch.Tensor,
        dy_intersections: torch.Tensor,
        intersection_indices: torch.Tensor,
        absolute_intensities: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> torch.Tensor:
        """
        Sample a bitmap (flux density distribution) of the reflected rays on the target area.

        Parameters
        ----------
        target_area : TargetArea
            The target area used to sample the bitmap on.
        dx_intersections : torch.Tensor
            The x-position of the intersection with the target area, scaled to the bitmap resolution.
        dy_intersections : torch.Tensor
            The y-position of the intersection with the target area, scaled to the bitmap resolution.
        intersection_indices : torch.Tensor
            Indices of the pixels.
        absolute_intensities : torch.Tensor
            The absolute intensities of the rays hitting the target plane.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        torch.Tensor
            The flux density distribution of the reflected rays on the target area.
        """
        device = torch.device(device)

        # dx_intersections and dy_intersections contain intersection coordinates ranging from 0 to target_area.plane_e/_u.
        # x_intersections and y_intersections contain those intersection coordinates scaled to a range from 0 to bitmap_resolution_e/_u.
        # Additionally a mask is applied, only the intersections where intersection_indices == True are kept, the tensors are flattened.
        x_intersections = (
            dx_intersections[intersection_indices]
            / target_area.plane_e
            * self.bitmap_resolution_e
        )
        y_intersections = (
            dy_intersections[intersection_indices]
            / target_area.plane_u
            * self.bitmap_resolution_u
        )
        absolute_intensities = absolute_intensities[intersection_indices]

        # We assume a continuously positioned value in-between four
        # discretely positioned pixels, similar to this:
        #
        # 1|2
        # -.-
        # 4|3
        #
        # where the numbers are the four neighboring, discrete pixels, the
        # "-" and "|" are the discrete pixel borders, and the "." is the
        # continuous value anywhere in-between the four pixels we sample.
        # That the "." may be anywhere in-between the four pixels is not
        # shown in the ASCII diagram, but is important to keep in mind.

        # The lower-valued neighboring pixels (for x this corresponds to 1
        # and 4, for y to 3 and 4).
        x_indices_low = x_intersections.to(torch.int32)
        y_indices_low = y_intersections.to(torch.int32)
        # The higher-valued neighboring pixels (for x this corresponds to 2
        # and 3, for y to 1 and 2).
        x_indices_high = x_indices_low + 1
        y_indices_high = y_indices_low + 1

        total_intersections = x_intersections.shape[0]
        x_indices = torch.zeros(
            (total_intersections * 4), device=device, dtype=torch.int32
        )
        x_indices[:total_intersections] = x_indices_low
        x_indices[total_intersections : total_intersections * 2] = x_indices_high
        x_indices[total_intersections * 2 : total_intersections * 3] = x_indices_high
        x_indices[total_intersections * 3 :] = x_indices_low

        y_indices = torch.zeros(
            (total_intersections * 4), device=device, dtype=torch.int32
        )
        y_indices[:total_intersections] = y_indices_high
        y_indices[total_intersections : total_intersections * 2] = y_indices_high
        y_indices[total_intersections * 2 : total_intersections * 3] = y_indices_low
        y_indices[total_intersections * 3 :] = y_indices_low

        # When distributing the continuously positioned value/intensity to
        # the discretely positioned pixels, we give the corresponding
        # "influence" of the value to each neighbor. Here, we calculate this
        # influence for each neighbor.

        # x-value influence in 1 and 4
        x_low_influences = x_indices_high - x_intersections
        # y-value influence in 3 and 4
        y_low_influences = y_indices_high - y_intersections
        # x-value influence in 2 and 3
        x_high_influences = x_intersections - x_indices_low
        # y-value influence in 1 and 2
        y_high_influences = y_intersections - y_indices_low

        # We now calculate the distributed intensities for each neighboring
        # pixel and assign the correctly ordered indices to the intensities
        # so we know where to position them. The numbers correspond to the
        # ASCII diagram above.
        intensities_pixel_1 = (
            x_low_influences * y_high_influences * absolute_intensities
        )
        intensities_pixel_2 = (
            x_high_influences * y_high_influences * absolute_intensities
        )
        intensities_pixel_3 = (
            x_high_influences * y_low_influences * absolute_intensities
        )
        intensities_pixel_4 = x_low_influences * y_low_influences * absolute_intensities

        intensities = torch.zeros((total_intersections * 4), device=device)
        intensities[:total_intersections] = intensities_pixel_1
        intensities[total_intersections : total_intersections * 2] = intensities_pixel_2
        intensities[total_intersections * 2 : total_intersections * 3] = (
            intensities_pixel_3
        )
        intensities[total_intersections * 3 :] = intensities_pixel_4

        # For distribution, we regard even those neighboring pixels that are
        # _not_ part of the image. That is why here, we set up a mask to
        # choose only those indices that are actually in the bitmap (i.e. we
        # prevent out-of-bounds access).
        intersections_indices = (
            (0 <= x_indices)
            & (x_indices < self.bitmap_resolution_e)
            & (0 <= y_indices)
            & (y_indices < self.bitmap_resolution_u)
        )

        # Flux density map for heliostat field
        total_bitmap = torch.zeros(
            [self.bitmap_resolution_u, self.bitmap_resolution_e],
            dtype=dx_intersections.dtype,
            device=device,
        )
        # Add up all distributed intensities in the corresponding indices.
        total_bitmap.index_put_(
            (
                self.bitmap_resolution_u - 1 - y_indices[intersections_indices],
                self.bitmap_resolution_e - 1 - x_indices[intersections_indices],
            ),
            intensities[intersections_indices],
            accumulate=True,
        )

        return total_bitmap
