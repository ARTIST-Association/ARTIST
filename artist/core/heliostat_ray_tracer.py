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
        number_of_active_heliostats: int,
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
        number_of_active_heliostats : int
            The number of active heliostats in the scenario.
        random_seed : int
            The random seed used for generating the distortions (default is 7).
        """
        self.distortions_u, self.distortions_e = light_source.get_distortions(
            number_of_points=number_of_points_per_heliostat,
            number_of_active_heliostats=number_of_active_heliostats,
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
    rank_indices : int
        The indices corresponding to the ranks assigned samples.

    See Also
    --------
    :class:`torch.utils.data.Sampler` : Reference to the parent class.
    """

    def __init__(
        self,
        number_of_samples: int,
        number_of_active_heliostats: int,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        """
        Set up a custom distributed sampler to assign data to each rank or leave them idle.

        Parameters
        ----------
        number_of_samples : int
            Length of the dataset or total number of samples.
        number_of_active_heliostats : int
            Number of active heliostats.
        world_size : int
            World size or total number of processes (default is 1).
        rank : int
            Rank of the current process (default is 0).
        """
        super().__init__()
        number_of_active_ranks = min(number_of_active_heliostats, world_size)
        self.rank_indices = []

        if rank < number_of_active_ranks:
            number_of_samples_per_heliostat = (
                number_of_samples // number_of_active_heliostats
            )
            indices: list[int] = []

            for index in range(number_of_active_heliostats):
                if index % number_of_active_ranks == rank:
                    start = index * number_of_samples_per_heliostat
                    end = start + number_of_samples_per_heliostat
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
        Indicates whether blocking is activated.
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
    ray_magnitude : float
        Magnitude of each single ray.
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
        dni: float | None = None,
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
        blocking_active : bool
            Flag indicating whether blocking is activated (default is True).
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
        dni : float | None
            Direct normal irradiance in W/m^2 (default is None -> ray magnitude = 1.0).
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
            number_of_active_heliostats=self.heliostat_group.number_of_active_heliostats,
            random_seed=random_seed,
        )
        # Create restricted distributed sampler.
        self.distortions_sampler = RestrictedDistributedSampler(
            number_of_samples=len(self.distortions_dataset),
            number_of_active_heliostats=(
                self.heliostat_group.active_heliostats_mask > 0
            ).sum(),
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

        if dni is not None:
            # Calculate surface area per heliostat.
            canting_norm = (torch.norm(self.heliostat_group.canting[0], dim=1)[0])[:2]
            dimensions = (canting_norm * 4) + 0.02
            heliostat_surface_area = dimensions[0] * dimensions[1]
            # Calculate ray magnitude.
            power_single_heliostat = dni * heliostat_surface_area
            rays_per_heliostat = (
                self.heliostat_group.surface_points.shape[1]
                * self.scenario.light_sources.light_source_list[0].number_of_rays
            )
            self.ray_magnitude = power_single_heliostat / rays_per_heliostat
        else:
            self.ray_magnitude = 1.0

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
            Amount of global ray extinction, responsible for shading (default is 0.0 -> no extinction).
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
            ray_magnitudes=torch.full(
                (scattered_rays.shape[: index_mapping.ray_directions]),
                self.ray_magnitude,
                device=device,
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

        num_heliostats = active_heliostats_mask.sum()
        bitmap_height = self.bitmap_resolution[index_mapping.unbatched_bitmap_u]
        bitmap_width = self.bitmap_resolution[index_mapping.unbatched_bitmap_e]

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

        # Determine the e- and u-positions of the intersections with the target areas, scaled to the bitmap resolutions.
        # Here we decide that the bottom left corner of the 2D bitmap is the origin of the flux image that is computed.
        target_intersections_e = (
            intersections[:, :, :, index_mapping.e] + plane_widths / 2 - plane_centers_e
        )
        target_intersections_u = (
            intersections[:, :, :, index_mapping.u]
            + plane_heights / 2
            - plane_centers_u
        )

        # target_intersections_e and target_intersections_u contain intersection coordinates ranging from 0 to target_area.plane_e/_u.
        # bitmap_intersections_e and bitmap_intersections_u contain those intersection coordinates scaled to a range from 0 to bitmap_resolution_e/_u - 1.
        # We scale to bitmap_width - 1 and bitmap_height - 1, because the indices start at 0 and end at bitmap_width - 1 or bitmap_height - 1
        bitmap_intersections_e = (
            target_intersections_e / plane_widths * (bitmap_width - 1)
        ).reshape(-1, total_intersections)
        bitmap_intersections_u = (
            target_intersections_u / plane_heights * (bitmap_height - 1)
        ).reshape(-1, total_intersections)

        absolute_intensities = absolute_intensities.reshape(-1, total_intersections)

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
        # The western and lower neighbored pixels are saved in indices_low_e and indices_low_u
        # (for e this corresponds to pixel 1 and 4, for u to 3 and 4).
        # The eastern and upper neighbored pixels are accessed via indices_low_e + 1 and
        # indices_low_u + 1 (for e this corresponds to 2 and 3, for u to 1 and 2).
        # We use the straight-through estimator trick for differentiability.
        indices_floor_e = torch.floor(bitmap_intersections_e)
        indices_floor_u = torch.floor(bitmap_intersections_u)

        indices_low_e = (
            bitmap_intersections_e + (indices_floor_e - bitmap_intersections_e).detach()
        ).to(torch.int32)
        indices_low_u = (
            bitmap_intersections_u + (indices_floor_u - bitmap_intersections_u).detach()
        ).to(torch.int32)

        # When distributing the continuously positioned value/intensity to
        # the discretely positioned pixels, we assign the corresponding
        # contribution to each neighbor based on its distance to the original,
        # continuous intersection point.
        # e-value contribution to 1 and 4.
        contributions_low_e = indices_low_e + 1 - bitmap_intersections_e
        # u-value contribution to 3 and 4.
        contributions_low_u = indices_low_u + 1 - bitmap_intersections_u
        # e-value contribution to 2 and 3.
        contributions_high_e = bitmap_intersections_e - indices_low_e
        # u-value contribution to 1 and 2.
        contributions_high_u = bitmap_intersections_u - indices_low_u

        # We now calculate the distributed intensities for each neighboring
        # pixel and assign the correctly ordered indices to the intensities
        # so we know where to position them. The numbers correspond to the
        # ASCII diagram above.
        intensities_pixel_1 = (
            contributions_low_e * contributions_high_u * absolute_intensities
        )
        intensities_pixel_2 = (
            contributions_high_e * contributions_high_u * absolute_intensities
        )
        intensities_pixel_3 = (
            contributions_high_e * contributions_low_u * absolute_intensities
        )
        intensities_pixel_4 = (
            contributions_low_e * contributions_low_u * absolute_intensities
        )

        # For the distributions, we regarded even those neighboring pixels that are
        # _not_ part of the image.
        # That is why here, we set up a mask to choose only those indices that are actually
        # in the bitmap (i.e., we prevent out-of-bounds access).
        intersection_indices_on_target = (
            (0 <= bitmap_intersections_e)
            & (bitmap_intersections_e < bitmap_width - 1)
            & (0 <= bitmap_intersections_u)
            & (bitmap_intersections_u < bitmap_height - 1)
        )

        # Flux density maps for each active heliostat.
        bitmaps_flat = torch.zeros(
            (num_heliostats, bitmap_height * bitmap_width), device=device
        )

        # scatter_add_ can only handle flat tensors per batch. That is why the bitmaps are flattened.
        # As an example: A bitmap with width = 4 and height = 2 has a total of 8 pixels.
        # Therefore, flattened, the indices range from 0 to 7.
        # 0     1     2     3
        # [0,0] [0,1] [0,2] [0,3]
        # [1,0] [1,1] [1,2] [1,3]
        # 4     5     6     7
        # The element at position [1,2] in the 2D array is at index 6 in the flattened tensor.
        # To convert the pixel indices from their 2D representation to a flattened version we need
        # to compute the row indices times the bitmap_width plus the column indices.
        # In the example this is 1 * 4 + 2 = 6
        # In our more general case that is:
        # flattened_indices = indices_u * bitmap_width + indices_e
        # Since tensor indices have their origin of (0,0) in the top left, but our image indices have their
        # origin in the bottom left, we need to flip the row (u) indices. That is:
        # flattened_indices = ((bitmap_height - 1) - indices_u) * bitmap_width + indices_e
        # The column indices also need to be flipped because the the more intuitive way to look at flux prediction
        # bitmaps, is to imagine oneself to stand in the heliostat field looking at the receiver.
        # This means that we look at the backside of the flux images. This corresponds to a flip of left and right.
        # Therefore our final indices are:
        # flattened_indices = (((bitmap_height - 1) - indices_u) * bitmap_width + ((bitmap_width - 1) - indices_e))
        # (For the flips, we need to subtract 1 from bitmap_height and bitmap_width, because this flips indices, we
        # do not need to subtract the 1 for the multiplication with bitmap_width because there we are interested in the
        # number of elements, not the indices.)
        index_3 = (
            ((bitmap_height - 1) - indices_low_u) * bitmap_width
            + ((bitmap_width - 1) - (indices_low_e + 1))
        ).long()
        index_4 = (
            ((bitmap_height - 1) - indices_low_u) * bitmap_width
            + ((bitmap_width - 1) - indices_low_e)
        ).long()

        # We need to filter out out of bounds indices. scatter_add_ cannot handle advanced indexing in its parameters,
        # therefore we cannot filter out invalid intersections by their indices. Instead we set all out of bounds indices
        # to 0, that way they do not cause index out of bounds errors, and we also set the intensities at these indices
        # to 0 so they do not add to the flux.
        index_3[~intersection_indices_on_target] = 0
        index_4[~intersection_indices_on_target] = 0
        intensities_pixel_1 = intensities_pixel_1 * intersection_indices_on_target
        intensities_pixel_2 = intensities_pixel_2 * intersection_indices_on_target
        intensities_pixel_3 = intensities_pixel_3 * intersection_indices_on_target
        intensities_pixel_4 = intensities_pixel_4 * intersection_indices_on_target

        bitmaps_flat.scatter_add_(1, index_4 + 1, intensities_pixel_1)
        bitmaps_flat.scatter_add_(1, index_3 + 1, intensities_pixel_2)
        bitmaps_flat.scatter_add_(1, index_3, intensities_pixel_3)
        bitmaps_flat.scatter_add_(1, index_4, intensities_pixel_4)

        bitmaps_per_heliostat = bitmaps_flat.view(
            num_heliostats, bitmap_height, bitmap_width
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
