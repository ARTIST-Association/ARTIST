import logging
from typing import TYPE_CHECKING, Iterator

import torch
from torch import Tensor
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
    bitmap_resolution : torch.Tensor
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
    get_bitmaps_per_target()
        Transform bitmaps per heliostat to bitmaps per target area.
    bilinear_splatting()
        Distribute ray intensities onto bitmap pixels using bilinear splatting.
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
            self.blocking_heliostat_surfaces_active = torch.zeros_like(
                self.blocking_heliostat_surfaces
            )
            for group in self.scenario.heliostat_field.heliostat_groups:
                surfaces = group.surface_points + group.positions.unsqueeze(1)
                mask = group.active_heliostats_mask.bool()
                if mask.any():
                    surfaces[mask] = group.active_surface_points
                else:
                    log.warning(
                        "Not all heliostat groups have been aligned yet. "
                        "Using horizontal heliostats as blocking planes."
                    )
                blocking_heliostat_surfaces_active_list.append(surfaces)
            self.blocking_heliostat_surfaces_active = torch.cat(
                blocking_heliostat_surfaces_active_list
            )

        if dni is not None:
            # Calculate surface area per heliostat.
            canting_norm = (torch.norm(self.heliostat_group.canting[0], dim=1)[0])[:2]
            dimensions = (canting_norm * 4) + 0.02
            heliostat_surface_area = (
                dimensions[index_mapping.heliostat_width]
                * dimensions[index_mapping.heliostat_height]
            )
            # Calculate ray magnitude.
            power_single_heliostat = dni * heliostat_surface_area
            rays_per_heliostat = (
                self.heliostat_group.surface_points.shape[1]
                * self.scenario.light_sources.light_source_list[
                    index_mapping.first_light_source
                ].number_of_rays
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
        target_area_indices: torch.Tensor,
        ray_extinction_factor: float = 0.0,
        mirror_reflectivity: float = 0.935,
        device: torch.device | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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
        target_area_indices : torch.Tensor
            The indices of the target areas for each active heliostat.
            Tensor of shape [number_of_active_heliostats].
        ray_extinction_factor : float
            Amount of global ray extinction, responsible for shading (default is 0.0 -> no extinction).
        mirror_reflectivity : float
            Fraction of incident ray intensity reflected by the mirror surface (default is 0.935).
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
        torch.Tensor
            The fraction of rays hitting the target, neglecting blocking effects.
            Shape is [number_of_active_heliostats].
        torch.Tensor
            The fraction of rays not being blocked.
            Shape is [number_of_active_heliostats].
        torch.Tensor
            The fraction of rays actually hitting the target, taking into account blocking effects.
            Shape is [number_of_active_heliostats].
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
            # Compute the heliostat blocking primitives.
            (
                blocking_primitives_corners,
                blocking_primitives_spans,
                blocking_primitives_normals,
            ) = blocking.create_blocking_primitives_rectangles_by_index(
                blocking_heliostats_active_surface_points=self.blocking_heliostat_surfaces_active,
                device=device,
            )

        flux_distributions = torch.empty(
            (
                int(active_heliostats_mask.sum()),
                int(self.bitmap_resolution[index_mapping.unbatched_bitmap_e]),
                int(self.bitmap_resolution[index_mapping.unbatched_bitmap_u]),
            ),
            device=device,
        )
        intercept_factor = torch.empty(active_heliostats_mask.sum(), device=device)
        on_target_factor = torch.empty(active_heliostats_mask.sum(), device=device)
        blocking_factor = torch.empty(active_heliostats_mask.sum(), device=device)

        global_active_indices = torch.nonzero(active_heliostats_mask, as_tuple=True)[0]
        for batch_index, (batch_u, batch_e) in enumerate(self.distortions_loader):
            sampler_indices = list(self.distortions_sampler)
            batch_mask_indices = sampler_indices[
                batch_index * self.batch_size : (batch_index + 1) * self.batch_size
            ]
            active_heliostats_mask_batch = torch.zeros(
                self.heliostat_group.number_of_active_heliostats,
                dtype=torch.bool,
                device=device,
            )
            active_heliostats_mask_batch[batch_mask_indices] = True

            rays = self.scatter_rays(
                distortion_u=batch_u,
                distortion_e=batch_e,
                original_ray_direction=self.heliostat_group.preferred_reflection_directions[
                    active_heliostats_mask_batch
                ],
                device=device,
            )

            # Choose the heliostat indices that are in the active batch and have a planar target area.
            planar_active_mask = (
                active_heliostats_mask_batch
                & (
                    target_area_indices
                    < self.scenario.solar_tower.number_of_target_areas_per_type[
                        index_mapping.planar_target_areas
                    ]
                )
            )[active_heliostats_mask_batch]

            rays_planar_targets = Rays(
                ray_directions=rays.ray_directions[planar_active_mask],
                ray_magnitudes=rays.ray_magnitudes[planar_active_mask],
            )
            rays_cylindrical_targets = Rays(
                ray_directions=rays.ray_directions[~planar_active_mask],
                ray_magnitudes=rays.ray_magnitudes[~planar_active_mask],
            )

            intersection_distances_target = torch.zeros(
                (
                    int(active_heliostats_mask_batch.sum()),
                    self.light_source.number_of_rays,
                    self.heliostat_group.surface_points.shape[1],
                ),
                device=device,
            )
            angle_reduced_intensities = torch.zeros(
                (
                    int(active_heliostats_mask_batch.sum()),
                    self.light_source.number_of_rays,
                    self.heliostat_group.surface_points.shape[1],
                ),
                device=device,
            )
            bitmap_intersections_e = torch.zeros(
                (
                    int(active_heliostats_mask_batch.sum()),
                    self.light_source.number_of_rays,
                    self.heliostat_group.surface_points.shape[1],
                ),
                device=device,
            )
            bitmap_intersections_u = torch.zeros(
                (
                    int(active_heliostats_mask_batch.sum()),
                    self.light_source.number_of_rays,
                    self.heliostat_group.surface_points.shape[1],
                ),
                device=device,
            )

            if planar_active_mask.sum() > 0:
                (
                    bitmap_intersections_e[planar_active_mask],
                    bitmap_intersections_u[planar_active_mask],
                    intersection_distances_target[planar_active_mask],
                    angle_reduced_intensities[planar_active_mask],
                ) = raytracing_utils.line_plane_intersections(
                    rays=rays_planar_targets,
                    points_at_ray_origins=self.heliostat_group.active_surface_points[
                        active_heliostats_mask_batch
                    ][planar_active_mask],
                    target_areas=self.scenario.solar_tower.target_areas[
                        index_mapping.planar_target_areas
                    ],  # type: ignore[arg-type]
                    target_area_indices=target_area_indices[
                        active_heliostats_mask_batch
                    ][planar_active_mask],
                    bitmap_resolution=self.bitmap_resolution,
                    device=device,
                )

            if (~planar_active_mask).sum() > 0:
                (
                    bitmap_intersections_e[~planar_active_mask],
                    bitmap_intersections_u[~planar_active_mask],
                    intersection_distances_target[~planar_active_mask],
                    angle_reduced_intensities[~planar_active_mask],
                ) = raytracing_utils.line_cylinder_intersections(
                    rays=rays_cylindrical_targets,
                    points_at_ray_origins=self.heliostat_group.active_surface_points[
                        active_heliostats_mask_batch
                    ][~planar_active_mask],
                    target_areas=self.scenario.solar_tower.target_areas[
                        index_mapping.cylindrical_target_areas
                    ],  # type: ignore[arg-type]
                    target_area_indices=target_area_indices[
                        active_heliostats_mask_batch
                    ][~planar_active_mask]
                    - self.scenario.solar_tower.number_of_target_areas_per_type[
                        index_mapping.planar_target_areas
                    ],
                    bitmap_resolution=self.bitmap_resolution,
                    device=device,
                )

            # The variable blocked is all zeros if there is no blocking at all in the scene.
            # If blocking was activated in the HeliostatRaytracer, blocking will be computed.
            number_of_heliostats, number_of_rays, number_of_points = (
                intersection_distances_target.shape
            )
            number_of_rays_per_heliostat = number_of_rays * number_of_points
            blocked = torch.zeros(
                (number_of_heliostats, number_of_rays, number_of_points),
                device=device,
            )
            if self.blocking_active:
                batch_global_indices = global_active_indices[batch_mask_indices]
                ray_to_heliostat_mapping = batch_global_indices.repeat_interleave(
                    number_of_rays_per_heliostat
                )

                # Filter out the blocking primitives that are relevant for blocking.
                filtered_blocking_primitive_indices = blocking.lbvh_filter_blocking_planes(
                    points_at_ray_origins=self.heliostat_group.active_surface_points[
                        active_heliostats_mask_batch
                    ],
                    ray_directions=rays.ray_directions,
                    blocking_primitives_corners=blocking_primitives_corners,
                    ray_to_heliostat_mapping=ray_to_heliostat_mapping,
                    intersection_distances_target=intersection_distances_target,
                    device=device,
                )
                # Create the blocked ray mask based on the relevant blocking primitive indices.
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
                        epsilon=1e-12,
                        softness=1000.0,
                    )

            intensities = (
                angle_reduced_intensities
                * (1 - blocked)
                * (1 - ray_extinction_factor)
                * mirror_reflectivity
            )

            bitmaps = self.bilinear_splatting(
                bitmap_intersections_e=bitmap_intersections_e,
                bitmap_intersections_u=bitmap_intersections_u,
                absolute_intensities=intensities,
                device=device,
            )

            flux_distributions[active_heliostats_mask_batch] = bitmaps

            on_target_factor[active_heliostats_mask_batch] = (
                angle_reduced_intensities > 0
            ).sum((1, 2)) / number_of_rays_per_heliostat
            blocking_factor[active_heliostats_mask_batch] = (blocked < 1e-3).sum(
                (1, 2)
            ) / number_of_rays_per_heliostat
            intercept_factor[active_heliostats_mask_batch] = (intensities > 0).sum(
                (1, 2)
            ) / number_of_rays_per_heliostat

        return flux_distributions, intercept_factor, on_target_factor, blocking_factor

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

    def get_bitmaps_per_target(
        self,
        bitmaps_per_heliostat: torch.Tensor,
        target_area_indices: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Transform bitmaps per heliostat to bitmaps per target area.

        Parameters
        ----------
        bitmaps_per_heliostat : torch.Tensor
            Bitmaps per heliostat.
            Tensor of shape [number_of_active_heliostats, bitmap_resolution_e, bitmap_resolution_u].
        target_area_indices : torch.Tensor
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
                int(self.scenario.solar_tower.number_of_target_areas_per_type.sum()),
                int(self.bitmap_resolution[index_mapping.unbatched_bitmap_e]),
                int(self.bitmap_resolution[index_mapping.unbatched_bitmap_u]),
            ),
            device=device,
        )
        for index in range(group_bitmaps_per_target.shape[0]):
            mask = target_area_indices == index
            if mask.any():
                group_bitmaps_per_target[index] = bitmaps_per_heliostat[mask].sum(
                    dim=index_mapping.heliostat_dimension
                )

        return group_bitmaps_per_target

    def bilinear_splatting(
        self,
        bitmap_intersections_e: torch.Tensor,
        bitmap_intersections_u: torch.Tensor,
        absolute_intensities: torch.Tensor,
        device: torch.device | None,
    ) -> torch.Tensor:
        """
        Distribute ray intensities onto bitmap pixels using bilinear splatting.

        Each ray intersection with the target area is treated as a continuously
        positioned value between four neighboring discrete pixels (east/west and
        up/down neighbors). The intensity is split among these four pixels
        proportionally to their proximity to the intersection point, yielding a
        differentiable approximation of the discrete binning operation.

        Parameters
        ----------
        bitmap_intersections_e : torch.Tensor
            The east-component bitmap coordinates of ray intersections.
            Tensor of shape [number_of_active_heliostats, number_of_rays, number_of_surface_points].
        bitmap_intersections_u : torch.Tensor
            The up-component bitmap coordinates of ray intersections.
            Tensor of shape [number_of_active_heliostats, number_of_rays, number_of_surface_points].
        absolute_intensities : torch.Tensor
            The intensity of each ray at its intersection point.
            Tensor of shape [number_of_active_heliostats, number_of_rays, number_of_surface_points].
        device : torch.device | None
            The device on which to perform computations or load tensors and models.
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The flux density bitmaps, one per active heliostat.
            Tensor of shape [number_of_active_heliostats, bitmap_resolution_u, bitmap_resolution_e].
        """
        bitmap_height = self.bitmap_resolution[index_mapping.unbatched_bitmap_u]
        bitmap_width = self.bitmap_resolution[index_mapping.unbatched_bitmap_e]
        num_heliostats = absolute_intensities.shape[0]

        bitmap_intersections_e = bitmap_intersections_e.reshape(num_heliostats, -1)
        bitmap_intersections_u = bitmap_intersections_u.reshape(num_heliostats, -1)
        absolute_intensities = absolute_intensities.reshape(num_heliostats, -1)

        # To ensure differentiability of the ray tracing process, the
        # intensity of each ray is distributed via bilinear splatting.
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
        # (for E this corresponds to pixel 1 and 4, for U to 3 and 4).
        # The eastern and upper neighbored pixels are accessed via indices_low_e + 1 and
        # indices_low_u + 1 (for E this corresponds to 2 and 3, for U to 1 and 2).
        indices_low_e = bitmap_intersections_e.long()
        indices_low_u = bitmap_intersections_u.long()

        # When distributing the continuously positioned value/intensity to
        # the discretely positioned pixels, we assign the corresponding
        # contribution to each neighbor based on its distance to the original,
        # continuous intersection point.
        # Note that the implementation below is already optimized for memory
        # consumption. For improved clarity, the detailed derivation of the
        # splatting weights is sketched below:
        #
        # indices_high_e/u = indices_low_e/u + 1
        # contributions_low_e/u = indices_high_e/u - bitmap_intersections_e/u
        # contributions_high_e/u = bitmap_intersections_e/u - indices_low_e/u
        # weight_pixel_1 = contributions_low_e * contributions_high_u
        # weight_pixel_2 = contributions_high_e * contributions_high_u
        # weight_pixel_3 = contributions_high_e * contributions_low_u
        # weight_pixel_4 = contributions_low_e * contributions_low_u
        #
        # E-value contribution to 1 and 4
        contributions_low_e = indices_low_e + 1 - bitmap_intersections_e
        # U-value contribution to 3 and 4
        contributions_low_u = indices_low_u + 1 - bitmap_intersections_u
        # E-value contribution to 2 and 3
        contributions_high_e = bitmap_intersections_e - indices_low_e
        # U-value contribution to 1 and 2
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
            (0 <= indices_low_e)
            & (indices_low_e + 1 < bitmap_width)
            & (0 <= indices_low_u)
            & (indices_low_u + 1 < bitmap_height)
        )

        # Flux density maps for each active heliostat.
        bitmaps_flat = torch.zeros(
            (int(num_heliostats), int(bitmap_height * bitmap_width)), device=device
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
        index_1 = (indices_low_u + 1) * bitmap_width + indices_low_e
        index_2 = (indices_low_u + 1) * bitmap_width + indices_low_e + 1
        index_3 = indices_low_u * bitmap_width + indices_low_e + 1
        index_4 = indices_low_u * bitmap_width + indices_low_e

        # We need to filter out out-of-bounds indices. `scatter_add_` cannot handle advanced indexing in its parameters,
        # therefore we cannot filter out invalid intersections by their indices. Instead, we set all out-of-bounds
        # indices to 0. That way, they do not cause index-out-of-bounds errors, and we also set the intensities at these
        # indices to 0 so they do not add to the flux.
        index_1[~intersection_indices_on_target] = 0
        index_2[~intersection_indices_on_target] = 0
        index_3[~intersection_indices_on_target] = 0
        index_4[~intersection_indices_on_target] = 0

        intensities_pixel_1 = intensities_pixel_1 * intersection_indices_on_target
        intensities_pixel_2 = intensities_pixel_2 * intersection_indices_on_target
        intensities_pixel_3 = intensities_pixel_3 * intersection_indices_on_target
        intensities_pixel_4 = intensities_pixel_4 * intersection_indices_on_target

        bitmaps_flat.scatter_add_(1, index_1, intensities_pixel_1)
        bitmaps_flat.scatter_add_(1, index_2, intensities_pixel_2)
        bitmaps_flat.scatter_add_(1, index_3, intensities_pixel_3)
        bitmaps_flat.scatter_add_(1, index_4, intensities_pixel_4)

        bitmaps_per_heliostat = bitmaps_flat.view(
            num_heliostats, int(bitmap_height), int(bitmap_width)
        )

        # Since tensor indices have their origin of (0,0) in the top left, but our image indices have their
        # origin in the bottom left, we need to flip the row (u) indices, i.e., up-down flip (flip along axis 1).
        return torch.flip(bitmaps_per_heliostat, [1])
