import logging
from typing import TYPE_CHECKING, Iterator

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

import artist.util.index_mapping
from artist.core import blocking

if TYPE_CHECKING:
    from artist.field.heliostat_group import HeliostatGroup
from artist.scenario.scenario import Scenario
from artist.scene import LightSource
from artist.scene.rays import Rays
from artist.util import index_mapping, raytracing_utils, utils
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
        dataset are the distortions. The distortions are used in our version of "heliostat"-tracing to
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
        return self.distortions_u.shape[index_mapping.heliostat_dimension]

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

    The ``DistributedSampler`` from ``torch`` replicates samples if the size of the dataset
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
    rank_indices : int
        The indices corresponding to the ranks assigned samples.

    See Also
    --------
    :class:`torch.utils.data.Sampler` : Reference to the parent class.
    """

    def __init__(
        self,
        number_of_active_heliostats: int,
        number_of_samples: int,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        """
        Set up a custom distributed sampler to assign data to each rank or leave them idle.

        Parameters
        ----------
        number_of_active_heliostats : int
            Number of active heliostats.
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
        self.number_of_active_ranks = min(number_of_active_heliostats, world_size)
        self.rank_indices = []

        if self.rank < self.number_of_active_ranks:
            chunk_size = number_of_samples // number_of_active_heliostats
            indices: list[int] = []

            for chunk_idx in range(number_of_active_heliostats):
                if chunk_idx % self.number_of_active_ranks == self.rank:
                    start = chunk_idx * chunk_size
                    end = start + chunk_size
                    indices.extend(range(start, end))

            self.rank_indices = indices
        else:
            self.rank_indices = []

    def __iter__(self) -> Iterator[int]:
        """
        Generate a sequence of indices for the current rank's portion of the dataset.

        Returns
        -------
        Iterator[int]
            An iterator over indices for the current rank.
        """
        return iter(self.rank_indices)


class HeliostatRayTracer:
    """
    Implement the functionality for heliostat ray tracing.

    Attributes
    ----------
    scenario : Scenario
        The scenario used to perform ray tracing.
    heliostat_group : HeliostatGroup
        The selected heliostat group containing active heliostats.
    blocking_active : bool
        Indicates wether blocking is activated.
    world_size : int
        The world size i.e., the overall number of processes.
    rank : int
        The rank, i.e., individual process ID.
    batch_size : int
        The amount of samples (heliostats) processed in parallel within a single rank.
    light_source : LightSource
        The light source emitting the traced rays.
    distortions_dataset : DistortionsDataset
        The dataset containing the distortions for ray scattering.
    distortions_sampler : RestrictedDistributedSampler
        The distortion sampler.
    distortions_loader : DataLoader
        The dataloader that loads the distortions.
    bitmap_resolution : int
        The resolution of the bitmap in both directions.
        Tensor of shape [2].
    blocking_heliostat_surfaces : torch.Tensor
        The heliostat surfaces considered during blocking calculations.
        Tensor of shape [number_of_heliostats, number_of_combined_surface_points_all_facets, 4].
    blocking_heliostat_surfaces_active : torch.Tensor
        The aligned heliostat surfaces considered during blocking calculations.
        Tensor of shape [number_of_heliostats, number_of_combined_surface_points_all_facets, 4].

    Methods
    -------
    get_sampler_indices()
        Get the indices assigned to the current rank by the distributed sampler.
    trace_rays()
        Perform heliostat ray tracing.
    scatter_rays()
        Scatter the reflected rays around the preferred ray directions for each heliostat.
    sample_bitmaps()
        Sample bitmaps (flux density distributions) of the reflected rays on the target areas.
    get_bitmaps_per_target()
        Transform bitmaps per heliostat to bitmaps per target area.
    """

    def __init__(
        self,
        scenario: Scenario,
        heliostat_group: "HeliostatGroup",
        blocking_active: bool = True,
        world_size: int = 1,
        rank: int = 0,
        batch_size: int = 100,
        random_seed: int = 7,
        bitmap_resolution: torch.Tensor = torch.tensor(
            [
                artist.util.index_mapping.bitmap_resolution,
                artist.util.index_mapping.bitmap_resolution,
            ]
        ),
    ) -> None:
        """
        Initialize the heliostat ray tracer.

        "Heliostat"-tracing is one kind of ray tracing applied in ``ARTIST``. For this kind of ray tracing,
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
            The amount of samples (heliostats) processed in parallel within a single rank (default is 100).
        random_seed : int
            The random seed used for generating the distortions (default is 7).
        bitmap_resolution : torch.Tensor
            The resolution of the bitmap in both directions. (default is torch.tensor([256,256])).
            Tensor of shape [2].
        """
        self.scenario = scenario
        self.heliostat_group = heliostat_group
        self.blocking_active = blocking_active

        self.world_size = world_size
        self.rank = rank
        self.batch_size = batch_size

        self.light_source = scenario.light_sources.light_source_list[
            index_mapping.first_light_source
        ]

        # Create distortions dataset.
        self.distortions_dataset = DistortionsDataset(
            light_source=self.light_source,
            number_of_points_per_heliostat=self.heliostat_group.active_surface_points.shape[
                index_mapping.number_of_surface_points_dimension
            ],
            number_of_heliostats=self.heliostat_group.number_of_active_heliostats,
            random_seed=random_seed,
        )
        # Create restricted distributed sampler.
        self.distortions_sampler = RestrictedDistributedSampler(
            number_of_active_heliostats=(
                self.heliostat_group.active_heliostats_mask > 0
            ).sum(),
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

        if self.blocking_active:
            self.blocking_heliostat_surfaces = torch.cat(
                [
                    group.surface_points
                    for group in self.scenario.heliostat_field.heliostat_groups
                ]
            )
            blocking_heliostat_surfaces_active_list = []
            for group in self.scenario.heliostat_field.heliostat_groups:
                if group.active_heliostats_mask.sum() <= 0:
                    blocking_heliostat_surfaces_active_list.append(
                        group.surface_points + group.positions.unsqueeze(1)
                    )
                if group.active_heliostats_mask.sum() > 0:
                    heliostat_mask = torch.cumsum(group.active_heliostats_mask, dim=0)
                    start_indices = heliostat_mask - group.active_heliostats_mask
                    blocking_heliostat_surfaces_active_list.append(
                        group.active_surface_points[start_indices]
                    )
            self.blocking_heliostat_surfaces_active = torch.cat(
                blocking_heliostat_surfaces_active_list
            )

    def get_sampler_indices(self) -> torch.Tensor:
        """
        Get the indices assigned to the current rank by the distributed sampler.

        Returns
        -------
        torch.Tensor
            Indices of the distortions dataset that are assigned to this rank.
            Tensor of shape [number of samples assigned to the current rank].
        """
        return torch.tensor(
            self.distortions_sampler.rank_indices,
            device=self.distortions_dataset.distortions_u.device,
        )

    def trace_rays(
        self,
        incident_ray_directions: torch.Tensor,
        active_heliostats_mask: torch.Tensor,
        target_area_mask: torch.Tensor,
        ray_extinction_factor: float = 0.0,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Perform heliostat ray tracing.

        Scatter the rays according to the distortions, calculate the intersections with the target planes,
        and sample the resulting bitmaps on the target areas. The bitmaps are generated separately for each
        active heliostat and are accessed individually.
        If blocking is activated in the ``HeliostatRayTracer``, rays that are blocked by other heliostats are
        filtered out.

        Parameters
        ----------
        incident_ray_directions : torch.Tensor
            The direction of the incident rays as seen from the heliostats.
            Tensor of shape [number_of_active_heliostats, 4].
        active_heliostats_mask : torch.Tensor
            A mask where 0 indicates a deactivated heliostat and 1 an activated one.
            An integer greater than 1 indicates that this heliostat is regarded multiple times.
            Tensor of shape [number_of_heliostats].
        target_area_mask : torch.Tensor
            The indices of the target areas for each active heliostat.
            Tensor of shape [number_of_active_heliostats].
        ray_extinction_factor : float
            Amount of global ray extinction, responsible for shading (default is 0.0 -> no shading).
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Raises
        ------
        ValueError
            If not all heliostats used for ray tracing have been aligned.

        Returns
        -------
        torch.Tensor
            The resulting bitmaps per heliostat.
            Tensor of shape [number_of_active_heliostats, bitmap_resolution_e, bitmap_resolution_u].
        """
        device = get_device(device=device)

        assert torch.equal(
            self.heliostat_group.active_heliostats_mask, active_heliostats_mask
        ), "Some heliostats were not aligned and cannot be raytraced."

        self.heliostat_group.preferred_reflection_directions = raytracing_utils.reflect(
            incident_ray_directions=incident_ray_directions.unsqueeze(
                index_mapping.number_rays_per_point
            ),
            reflection_surface_normals=self.heliostat_group.active_surface_normals,
        )

        if self.blocking_active:
            (
                blocking_primitives_corners,
                blocking_primitives_spans,
                blocking_primitives_normals,
            ) = blocking.create_blocking_primitives_rectangles_by_index(
                blocking_heliostats_active_surface_points=self.blocking_heliostat_surfaces_active,
                device=device,
            )

        flux_distributions = []
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

            # The variable blocked is all zeros if there is no blocking at all in the scene.
            # If blocking was activated in the HeliostatRaytracer, blocking will be computed.
            number_of_heliostats, number_of_rays, number_of_points, _ = (
                intersections.shape
            )
            blocked = torch.zeros(
                (number_of_heliostats, number_of_rays, number_of_points),
                device=device,
            )
            if self.blocking_active:
                points_at_ray_origins = self.heliostat_group.active_surface_points[
                    active_heliostats_mask_batch, None, :, :3
                ].expand(-1, self.light_source.number_of_rays, -1, -1)
                ray_to_heliostat_mapping = torch.arange(
                    number_of_heliostats, device=device
                ).repeat_interleave(number_of_rays * number_of_points)

                filtered_blocking_primitive_indices = (
                    blocking.lbvh_filter_blocking_planes(
                        points_at_ray_origins=points_at_ray_origins,
                        ray_directions=rays.ray_directions[..., :3],
                        blocking_primitives_corners=blocking_primitives_corners[
                            ..., :3
                        ],
                        ray_to_heliostat_mapping=ray_to_heliostat_mapping,
                        max_stack_size=128,
                        device=device,
                    )
                )

                if filtered_blocking_primitive_indices.numel() > 0:
                    blocked = blocking.soft_ray_blocking_mask(
                        ray_origins=self.heliostat_group.active_surface_points[
                            active_heliostats_mask_batch
                        ],
                        ray_directions=rays.ray_directions,
                        blocking_primitives_corners=blocking_primitives_corners[
                            filtered_blocking_primitive_indices
                        ],
                        blocking_primitives_spans=blocking_primitives_spans[
                            filtered_blocking_primitive_indices
                        ],
                        blocking_primitives_normals=blocking_primitives_normals[
                            filtered_blocking_primitive_indices
                        ],
                        distances_to_target=torch.norm(
                            intersections[..., :3] - points_at_ray_origins, dim=-1
                        ),
                        epsilon=1e-6,
                        softness=50.0,
                    )

            intensities = (
                absolute_intensities * (1 - blocked) * (1 - ray_extinction_factor)
            )

            batch_bitmaps = self.sample_bitmaps(
                intersections=intersections,
                absolute_intensities=intensities,
                active_heliostats_mask=active_heliostats_mask_batch,
                target_area_mask=target_area_mask[active_heliostats_mask_batch],
                device=device,
            )

            flux_distributions.append(batch_bitmaps)

        combined = torch.cat(flux_distributions, dim=0)

        return combined

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
            Tensor of shape [number_of_active_heliostats, number_of_rays, number_of_combined_surface_normals_all_facets].
        distortion_e : torch.Tensor
            The distortions in east direction (angles for scattering).
            Tensor of shape [number_of_active_heliostats, number_of_rays, number_of_combined_surface_normals_all_facets].
        original_ray_direction : torch.Tensor
            The ray direction around which to scatter.
            Tensor of shape [number_of_active_heliostats, number_of_combined_surface_normals_all_facets, 4].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
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
            rotations
            @ original_ray_direction.unsqueeze(
                index_mapping.number_rays_per_point
            ).unsqueeze(-1)
        ).squeeze(-1)

        return Rays(
            ray_directions=scattered_rays,
            ray_magnitudes=torch.ones(
                scattered_rays.shape[: index_mapping.ray_directions], device=device
            ),
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

        The bitmaps are saved for each active heliostat separately.

        Parameters
        ----------
        intersections : torch.Tensor
            The intersections of rays on the target area planes for each heliostat.
            Tensor of shape [number_of_active_heliostats, number_of_rays, number_of_combined_surface_points_all_facets, 4].
        absolute_intensities : torch.Tensor
            The absolute intensities of the rays hitting the target planes for each heliostat.
            Tensor of shape [number_of_active_heliostats, number_of_rays, number_of_combined_surface_points_all_facets].
        active_heliostats_mask : torch.Tensor
            Used to map bitmaps per heliostat to correct index.
            Tensor of shape [number_of_heliostats].
        target_area_mask : torch.Tensor
            The indices of target areas on which each heliostat is raytraced.
            Tensor of shape [number_of_active_heliostats].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The flux density distributions of the reflected rays on the target areas for each active heliostat.
            Tensor of shape [number_of_active_heliostats, bitmap_resolution_e, bitmap_resolution_u].
        """
        device = get_device(device=device)

        plane_widths = (
            self.scenario.target_areas.dimensions[target_area_mask][
                :, index_mapping.target_area_width
            ]
            .unsqueeze(index_mapping.number_rays_per_point)
            .unsqueeze(index_mapping.points_dimension)
        )
        plane_heights = (
            self.scenario.target_areas.dimensions[target_area_mask][
                :, index_mapping.target_area_height
            ]
            .unsqueeze(index_mapping.number_rays_per_point)
            .unsqueeze(index_mapping.points_dimension)
        )
        plane_centers_e = (
            self.scenario.target_areas.centers[target_area_mask][
                :, index_mapping.target_area_center_e
            ]
            .unsqueeze(index_mapping.number_rays_per_point)
            .unsqueeze(index_mapping.points_dimension)
        )
        plane_centers_u = (
            self.scenario.target_areas.centers[target_area_mask][
                :, index_mapping.target_area_center_u
            ]
            .unsqueeze(index_mapping.number_rays_per_point)
            .unsqueeze(index_mapping.points_dimension)
        )
        total_intersections = (
            intersections.shape[index_mapping.number_rays_per_point]
            * intersections.shape[index_mapping.surface_points]
        )
        absolute_intensities = absolute_intensities.reshape(-1, total_intersections)

        # Determine the x- and y-positions of the intersections with the target areas, scaled to the bitmap resolutions.
        dx_intersections = (
            intersections[:, :, :, index_mapping.e] + plane_widths / 2 - plane_centers_e
        )
        dy_intersections = (
            intersections[:, :, :, index_mapping.u]
            + plane_heights / 2
            - plane_centers_u
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
            dx_intersections
            / plane_widths
            * self.bitmap_resolution[index_mapping.unbatched_bitmap_e]
        ).reshape(-1, total_intersections)
        y_intersections = (
            dy_intersections
            / plane_heights
            * self.bitmap_resolution[index_mapping.unbatched_bitmap_u]
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
            (
                intersections.shape[index_mapping.heliostat_dimension],
                total_intersections * 4,
            ),
            device=device,
            dtype=torch.int32,
        )

        x_indices[:, :total_intersections] = x_indices_low
        x_indices[
            :, total_intersections : total_intersections * index_mapping.second_pixel
        ] = x_indices_high
        x_indices[
            :,
            total_intersections * index_mapping.second_pixel : total_intersections
            * index_mapping.third_pixel,
        ] = x_indices_high
        x_indices[:, total_intersections * index_mapping.third_pixel :] = x_indices_low

        y_indices = torch.zeros(
            (
                intersections.shape[index_mapping.heliostat_dimension],
                total_intersections * 4,
            ),
            device=device,
            dtype=torch.int32,
        )

        y_indices[:, :total_intersections] = y_indices_high
        y_indices[
            :, total_intersections : total_intersections * index_mapping.second_pixel
        ] = y_indices_high
        y_indices[
            :,
            total_intersections * index_mapping.second_pixel : total_intersections
            * index_mapping.third_pixel,
        ] = y_indices_low
        y_indices[:, total_intersections * index_mapping.third_pixel :] = y_indices_low

        # When distributing the continuously positioned value/intensity to
        # the discretely positioned pixels, we give the corresponding
        # "influence" of the value to each neighbor. Here, we calculate this
        # influence for each neighbor.

        # x-value influence in 1 and 4.
        x_low_influences = x_indices_high - x_intersections
        # y-value influence in 3 and 4.
        y_low_influences = y_indices_high - y_intersections
        # x-value influence in 2 and 3.
        x_high_influences = x_intersections - x_indices_low
        # y-value influence in 1 and 2.
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
        intensities[
            :, total_intersections : total_intersections * index_mapping.second_pixel
        ] = intensities_pixel_2.reshape(-1, total_intersections)
        intensities[
            :,
            total_intersections * index_mapping.second_pixel : total_intersections
            * index_mapping.third_pixel,
        ] = intensities_pixel_3.reshape(-1, total_intersections)
        intensities[:, total_intersections * index_mapping.third_pixel :] = (
            intensities_pixel_4.reshape(-1, total_intersections)
        )

        # For the distributions, we regarded even those neighboring pixels that are
        # _not_ part of the image but within a little boundary outside of the image as well.
        # That is why here, we set up a mask to choose only those indices that are actually
        # in the bitmap (i.e., we prevent out-of-bounds access).
        intersection_indices_2 = (
            (0 <= x_indices)
            & (x_indices < self.bitmap_resolution[index_mapping.unbatched_bitmap_e])
            & (0 <= y_indices)
            & (y_indices < self.bitmap_resolution[index_mapping.unbatched_bitmap_u])
        )

        final_intersection_indices = (
            intersection_indices_1.reshape(-1, total_intersections).repeat(
                1, self.heliostat_group.number_of_facets_per_heliostat
            )
            & intersection_indices_2
        )

        mask = final_intersection_indices.flatten()

        heliostat_indices = torch.repeat_interleave(
            torch.arange(
                active_heliostats_mask.sum(), device=active_heliostats_mask.device
            ),
            total_intersections * self.heliostat_group.number_of_facets_per_heliostat,
        )

        # Flux density maps for each active heliostat.
        bitmaps_per_heliostat = torch.zeros(
            (
                active_heliostats_mask.sum(),
                self.bitmap_resolution[index_mapping.unbatched_bitmap_u],
                self.bitmap_resolution[index_mapping.unbatched_bitmap_e],
            ),
            dtype=dx_intersections.dtype,
            device=device,
        )

        # Add up all distributed intensities in the corresponding indices.
        bitmaps_per_heliostat.index_put_(
            (
                heliostat_indices[mask],
                self.bitmap_resolution[index_mapping.unbatched_bitmap_u]
                - 1
                - y_indices[final_intersection_indices],
                self.bitmap_resolution[index_mapping.unbatched_bitmap_e]
                - 1
                - x_indices[final_intersection_indices],
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
            Tensor of shape [number_of_active_heliostats, bitmap_resolution_e, bitmap_resolution_u].
        target_area_mask : torch.Tensor
            The mapping from heliostat to target area.
            Tensor of shape [number_of_active_heliostats].
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            Bitmaps per target area.
            Tensor of shape [number_of_target_areas, bitmap_resolution_e, bitmap_resolution_u].
        """
        device = get_device(device=device)

        group_bitmaps_per_target = torch.zeros(
            (
                self.scenario.target_areas.number_of_target_areas,
                self.bitmap_resolution[index_mapping.unbatched_bitmap_e],
                self.bitmap_resolution[index_mapping.unbatched_bitmap_u],
            ),
            device=device,
        )
        for index in range(self.scenario.target_areas.number_of_target_areas):
            mask = target_area_mask == index
            if mask.any():
                group_bitmaps_per_target[index] = bitmaps_per_heliostat[mask].sum(
                    dim=index_mapping.heliostat_dimension
                )

        return group_bitmaps_per_target
