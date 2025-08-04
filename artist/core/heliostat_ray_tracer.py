import logging
from typing import TYPE_CHECKING, Iterator

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

if TYPE_CHECKING:
    from artist.field.heliostat_group import HeliostatGroup
from artist.scenario.scenario import Scenario
from artist.scene import LightSource
from artist.scene.rays import Rays
from artist.util import raytracing_utils, utils
from artist.util.environment_setup import get_device

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
    heliostat_group : HeliostatGroup
        The selected heliostat group containing active heliostats.
    world_size : int
        The world size i.e., the overall number of processes.
    rank : int
        The rank, i.e., individual process ID.
    batch_size : int
        The amount of samples (Heliostats) processed parallel within a single rank.
    light_source : LightSource
        The light source emitting the traced rays.
    distortions_dataset : DistortionsDataset
        The dataset containing the distortions for ray scattering.
    distortions_sampler : RestrictedDistributedSampler
        The distortion sampler.
    distortions_loader : DataLoader
        The dataloader that loads the distortions.
    bitmap_resolution : int
        The resolution of the bitmap.

    Methods
    -------
    trace_rays()
        Perform heliostat ray tracing.
    scatter_rays()
        Scatter the reflected rays around the preferred ray directions for each heliostat.
    sample_bitmaps()
        Sample bitmaps (flux density distributions) of the reflected rays on the target areas.
    get_bitmaps_per_target.
        Transform bitmaps per heliostat to bitmaps per target area.
    """

    def __init__(
        self,
        scenario: Scenario,
        heliostat_group: "HeliostatGroup",
        world_size: int = 1,
        rank: int = 0,
        batch_size: int = 1,
        random_seed: int = 7,
        bitmap_resolution: torch.Tensor = torch.tensor([256, 256]),
    ) -> None:
        """
        Initialize the heliostat ray tracer.

        "Heliostat"-tracing is one kind of ray tracing applied in ARTIST. For this kind of ray tracing,
        the rays are initialized on the heliostats. The rays originate in the discrete surface points.
        There they are multiplied, distorted, and scattered, and then they are sent to the aim points.
        Letting the rays originate on the heliostats, drastically reduces the number of rays that need
        to be traced.

        Parameters
        ----------
        scenario : Scenario
            The scenario used to perform ray tracing.
        heliostat_group : HeliostatGroup
            The selected heliostat group containing active heliostats.
        world_size : int
            The world size i.e., the overall number of processes (default is 1).
        rank : int
            The rank, i.e., individual process ID (default is 0).
        batch_size : int
            The amount of samples (Heliostats) processed parallel within a single rank (default is 1).
        random_seed : int
            The random seed used for generating the distortions (default is 7).
        bitmap_resolution : torch.Tensor
            The resolution of the bitmap (default is torch.tensor([256,256])).
        """
        self.scenario = scenario
        self.heliostat_group = heliostat_group

        self.world_size = world_size
        self.rank = rank
        self.batch_size = batch_size

        self.light_source = scenario.light_sources.light_source_list[0]

        # Create distortions dataset.
        self.distortions_dataset = DistortionsDataset(
            light_source=self.light_source,
            number_of_points_per_heliostat=self.heliostat_group.active_surface_points.shape[
                1
            ],
            number_of_heliostats=self.heliostat_group.number_of_active_heliostats,
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

        self.bitmap_resolution = bitmap_resolution

    def trace_rays(
        self,
        incident_ray_directions: torch.Tensor,
        active_heliostats_mask: torch.Tensor,
        target_area_mask: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Perform heliostat ray tracing.

        Scatter the rays according to the distortions, calculate the intersections with the target planes,
        and sample the resulting bitmaps on the target areas. The bitmaps are generated seperatly for each
        active heliostat and can be accessed individually or they can be combined to get the total flux
        density distribution for all heliostats on all target areas.

        Parameters
        ----------
        incident_ray_directions : torch.Tensor
            The direction of the incident rays as seen from the heliostats.
        active_heliostats_mask : torch.Tensor
            A mask where 0 indicates a deactivated heliostat and 1 an activated one.
            An integer greater than 1 indicates that this heliostat is regarded multiple times.
        target_area_mask : torch.Tensor
            The indices of the target areas for each active heliostat.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Raises
        ------
        ValueError
            If not all heliostats used for ray tracing have been aligned.

        Returns
        -------
        torch.Tensor
            The resulting bitmaps per heliostat.
        """
        device = get_device(device=device)

        assert torch.equal(
            self.heliostat_group.active_heliostats_mask, active_heliostats_mask
        ), "Some heliostats were not aligned and cannot be raytraced."

        flux_distributions = torch.zeros(
            (
                self.heliostat_group.number_of_active_heliostats,
                self.bitmap_resolution[1],
                self.bitmap_resolution[0],
            ),
            device=device,
        )

        self.heliostat_group.preferred_reflection_directions = raytracing_utils.reflect(
            incident_ray_directions=incident_ray_directions.unsqueeze(1),
            reflection_surface_normals=self.heliostat_group.active_surface_normals,
        )

        for batch_index, (batch_u, batch_e) in enumerate(self.distortions_loader):
            sampler_indices = list(self.distortions_sampler)

            active_heliostats_mask_batch = torch.zeros(
                self.heliostat_group.number_of_active_heliostats,
                dtype=torch.bool,
                device=device,
            )
            active_heliostats_mask_batch[
                sampler_indices[
                    batch_index * self.batch_size : (batch_index + 1) * self.batch_size
                ]
            ] = True

            rays = self.scatter_rays(
                distortion_u=batch_u,
                distortion_e=batch_e,
                original_ray_direction=self.heliostat_group.preferred_reflection_directions[
                    active_heliostats_mask_batch
                ],
                device=device,
            )

            intersections, absolute_intensities = (
                raytracing_utils.line_plane_intersections(
                    rays=rays,
                    points_at_ray_origins=self.heliostat_group.active_surface_points[
                        active_heliostats_mask_batch
                    ],
                    target_areas=self.scenario.target_areas,
                    target_area_mask=target_area_mask[active_heliostats_mask_batch],
                    device=device,
                )
            )

            bitmaps = self.sample_bitmaps(
                intersections=intersections,
                absolute_intensities=absolute_intensities,
                active_heliostats_mask=active_heliostats_mask_batch,
                target_area_mask=target_area_mask[active_heliostats_mask_batch],
                device=device,
            )

            flux_distributions = flux_distributions + bitmaps

        return flux_distributions

    def scatter_rays(
        self,
        distortion_u: torch.Tensor,
        distortion_e: torch.Tensor,
        original_ray_direction: torch.Tensor,
        device: torch.device | None = None,
    ) -> Rays:
        """
        Scatter the reflected rays around the preferred ray directions for each heliostat.

        Parameters
        ----------
        distortion_u : torch.Tensor
            The distortions in up direction (angles for scattering).
        distortion_e : torch.Tensor
            The distortions in east direction (angles for scattering).
        original_ray_direction : torch.Tensor
            The ray direction around which to scatter.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        Rays
            Scattered rays around the preferred reflection directions.
        """
        device = get_device(device=device)

        rotations = utils.rotate_distortions(
            u=distortion_u, e=distortion_e, device=device
        )

        scattered_rays = (
            rotations @ original_ray_direction.unsqueeze(1).unsqueeze(-1)
        ).squeeze(-1)

        return Rays(
            ray_directions=scattered_rays,
            ray_magnitudes=torch.ones(scattered_rays.shape[:-1], device=device),
        )

    def sample_bitmaps(
        self,
        intersections: torch.Tensor,
        absolute_intensities: torch.Tensor,
        active_heliostats_mask: torch.Tensor,
        target_area_mask: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Sample bitmaps (flux density distributions) of the reflected rays on the target areas.

        The bitmaps are saved for each active heliostat seperatly.

        Parameters
        ----------
        intersections : torch.Tensor
            The intersections of rays on the target area planes for each heliostat.
        absolute_intensities : torch.Tensor
            The absolute intensities of the rays hitting the target planes for each heliostat.
        active_heliostats_mask : torch.Tensor
            Used to map bitmaps per heliostat to correct index.
        target_area_mask : torch.Tensor
            The indices of target areas on which each heliostat should be raytraced.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The flux density distributions of the reflected rays on the target areas for each active heliostat.
        """
        device = get_device(device=device)

        plane_widths = (
            self.scenario.target_areas.dimensions[target_area_mask][:, 0]
            .unsqueeze(1)
            .unsqueeze(2)
        )
        plane_heights = (
            self.scenario.target_areas.dimensions[target_area_mask][:, 1]
            .unsqueeze(1)
            .unsqueeze(2)
        )
        plane_centers_e = (
            self.scenario.target_areas.centers[target_area_mask][:, 0]
            .unsqueeze(1)
            .unsqueeze(2)
        )
        plane_centers_u = (
            self.scenario.target_areas.centers[target_area_mask][:, 2]
            .unsqueeze(1)
            .unsqueeze(2)
        )
        total_intersections = intersections.shape[1] * intersections.shape[2]
        absolute_intensities = absolute_intensities.reshape(-1, total_intersections)

        # Determine the x- and y-positions of the intersections with the target areas, scaled to the bitmap resolutions.
        dx_intersections = (
            intersections[:, :, :, 0] + plane_widths / 2 - plane_centers_e
        )
        dy_intersections = (
            intersections[:, :, :, 2] + plane_heights / 2 - plane_centers_u
        )

        # Selection of valid intersection indices within the bounds of the target areas or within a little boundary outside the target areas.
        intersection_indices_1 = (
            (-1 <= dx_intersections)
            & (dx_intersections < plane_widths + 1)
            & (-1 <= dy_intersections)
            & (dy_intersections < plane_heights + 1)
        )

        # dx_intersections and dy_intersections contain intersection coordinates ranging from 0 to target_area.plane_e/_u.
        # x_intersections and y_intersections contain those intersection coordinates scaled to a range from 0 to bitmap_resolution_e/_u.
        # Additionally a mask is applied, only the intersections where intersection_indices == True are kept, the tensors are flattened.
        x_intersections = (
            dx_intersections / plane_widths * self.bitmap_resolution[0]
        ).reshape(-1, total_intersections)
        y_intersections = (
            dy_intersections / plane_heights * self.bitmap_resolution[1]
        ).reshape(-1, total_intersections)

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

        x_indices = torch.zeros(
            (intersections.shape[0], total_intersections * 4),
            device=device,
            dtype=torch.int32,
        )

        x_indices[:, :total_intersections] = x_indices_low
        x_indices[:, total_intersections : total_intersections * 2] = x_indices_high
        x_indices[:, total_intersections * 2 : total_intersections * 3] = x_indices_high
        x_indices[:, total_intersections * 3 :] = x_indices_low

        y_indices = torch.zeros(
            (intersections.shape[0], total_intersections * 4),
            device=device,
            dtype=torch.int32,
        )

        y_indices[:, :total_intersections] = y_indices_high
        y_indices[:, total_intersections : total_intersections * 2] = y_indices_high
        y_indices[:, total_intersections * 2 : total_intersections * 3] = y_indices_low
        y_indices[:, total_intersections * 3 :] = y_indices_low

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

        intensities = torch.zeros(
            (intersections.shape[0], total_intersections * 4), device=device
        )
        intensities[:, :total_intersections] = intensities_pixel_1.reshape(
            -1, total_intersections
        )
        intensities[:, total_intersections : total_intersections * 2] = (
            intensities_pixel_2.reshape(-1, total_intersections)
        )
        intensities[:, total_intersections * 2 : total_intersections * 3] = (
            intensities_pixel_3.reshape(-1, total_intersections)
        )
        intensities[:, total_intersections * 3 :] = intensities_pixel_4.reshape(
            -1, total_intersections
        )

        # For the distributions, we regarded even those neighboring pixels that are
        # _not_ part of the image but within a little boundary outside of the image as well.
        # That is why here, we set up a mask to choose only those indices that are actually
        # in the bitmap (i.e. we prevent out-of-bounds access).
        intersection_indices_2 = (
            (0 <= x_indices)
            & (x_indices < self.bitmap_resolution[0])
            & (0 <= y_indices)
            & (y_indices < self.bitmap_resolution[1])
        )

        final_intersection_indices = (
            intersection_indices_1.reshape(-1, total_intersections).repeat(1, 4)
            & intersection_indices_2
        )
        mask = final_intersection_indices.flatten()

        active_heliostat_indices = torch.nonzero(
            active_heliostats_mask, as_tuple=False
        ).squeeze()
        heliostat_indices = torch.repeat_interleave(
            active_heliostat_indices, total_intersections * 4
        )

        # Flux density maps for each active heliostat.
        bitmaps_per_heliostat = torch.zeros(
            (
                self.heliostat_group.number_of_active_heliostats,
                self.bitmap_resolution[1],
                self.bitmap_resolution[0],
            ),
            dtype=dx_intersections.dtype,
            device=device,
        )

        # Add up all distributed intensities in the corresponding indices.
        bitmaps_per_heliostat.index_put_(
            (
                heliostat_indices[mask],
                self.bitmap_resolution[1] - 1 - y_indices[final_intersection_indices],
                self.bitmap_resolution[0] - 1 - x_indices[final_intersection_indices],
            ),
            intensities[final_intersection_indices],
            accumulate=True,
        )

        return bitmaps_per_heliostat

    def get_bitmaps_per_target(
        self,
        bitmaps_per_heliostat: torch.Tensor,
        target_area_mask: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Transform bitmaps per heliostat to bitmaps per target area.

        Parameters
        ----------
        bitmaps_per_heliostat : torch.Tensor
            Bitmaps per heliostat.
        target_area_mask : torch.Tensor
            The mapping from heliostat to target area.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ARTIST will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            Bitmaps per target area.
        """
        device = get_device(device=device)

        group_bitmaps_per_target = torch.zeros(
            (
                self.scenario.target_areas.number_of_target_areas,
                self.bitmap_resolution[0],
                self.bitmap_resolution[1],
            ),
            device=device,
        )
        for index in range(self.scenario.target_areas.number_of_target_areas):
            mask = target_area_mask == index
            if mask.any():
                group_bitmaps_per_target[index] = bitmaps_per_heliostat[mask].sum(dim=0)

        return group_bitmaps_per_target
