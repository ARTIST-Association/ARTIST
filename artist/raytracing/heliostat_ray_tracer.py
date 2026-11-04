import logging
from collections.abc import Iterator

import torch
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader

from artist.field.heliostat_group import HeliostatGroup
from artist.geometry import transforms
from artist.raytracing import blocking, geometry
from artist.raytracing.sampling import DistortionsDataset, RestrictedDistributedSampler
from artist.scenario.scenario import Scenario
from artist.scene.rays import Rays
from artist.util import indices
from artist.util.env import get_device

log = logging.getLogger(__name__)
"""A logger for the heliostat ray tracer."""


class HeliostatRayTracer:
    """
    Implement the functionality for heliostat ray tracing.

    Attributes
    ----------
    scenario : Scenario
        The scenario used to perform ray tracing.
    heliostat_group : HeliostatGroup
        The selected heliostat group containing active heliostats.
    occlusion_active : bool
        Indicates whether blocking and shading is activated.
    world_size : int
        The world size, i.e., the overall number of processes.
    rank : int
        The rank, i.e., individual process ID.
    batch_size : int
        The number of samples (heliostats) processed in parallel within a single rank.
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
        Shape is ``[2]``.
    heliostat_primitives_corners : torch.Tensor
        The heliostat primitive corners.
        Shape is ``[number_of_heliostats, 4, 4]``.
    heliostat_primitives_spans : torch.Tensor
        The heliostat primitive spans in u and v direction.
        Shape is ``[number_of_heliostats, 2, 4]``.
    heliostat_primitives_normals : torch.Tensor
        The heliostat primitives normals.
        Shape is ``[number_of_heliostats, 4]``.
    tower_primitives_corners : torch.Tensor
        The tower primitive corners.
        Shape is ``[1, 4, 4]``.
    tower_primitives_spans : torch.Tensor
        The tower primitive spans in u and v direction.
        Shape is ``[1, 2, 4]``.
    tower_primitives_normals : torch.Tensor
        The tower primitives normals.
        Shape is ``[1, 4]``.
    ray_magnitude : float
        Magnitude of each single ray.

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
        heliostat_group: HeliostatGroup,
        occlusion_active: bool = True,
        world_size: int = 1,
        rank: int = 0,
        batch_size: int = 100,
        random_seed: int = 7,
        bitmap_resolution: torch.Tensor = torch.tensor(
            [
                indices.bitmap_resolution,
                indices.bitmap_resolution,
            ]
        ),
        dni: float | None = None,
        use_checkpointing: bool = False,
        device: torch.device | None = None,
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
        occlusion_active : bool
            Flag indicating whether blocking and shading is activated (default is True).
        world_size : int
            The world size, i.e., the overall number of processes (default is 1).
        rank : int
            The rank, i.e., individual process ID (default is 0).
        batch_size : int
            The number of samples (heliostats) processed in parallel within a single rank (default is 100).
        random_seed : int
            The random seed used for generating the distortions (default is 7).
        bitmap_resolution : torch.Tensor
            The resolution of the bitmap in both directions. (default is torch.tensor([256,256])).
            Shape is ``[2]``.
        dni : float | None
            Direct normal irradiance in W/m^2 (default is None -> ray magnitude = 1.0).
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.
        """
        device = get_device(device=device)

        self.scenario = scenario
        self.heliostat_group = heliostat_group
        self.occlusion_active = occlusion_active

        self.world_size = world_size
        self.rank = rank
        self.batch_size = batch_size
        self.use_checkpointing = use_checkpointing
        self._blocking_index_cache: dict[int, torch.Tensor] = {}
        self._will_recompute = False
        groups = self.scenario.heliostat_field.heliostat_groups
        group_index = next(i for i, g in enumerate(groups) if g is heliostat_group)
        self._group_offset = int(
            sum(g.number_of_heliostats for g in groups[:group_index])
        )

        self.light_source = scenario.light_sources.light_source_list[
            indices.first_light_source
        ]

        # Create distortions dataset.
        self.distortions_dataset = DistortionsDataset(
            light_source=self.light_source,
            number_of_points_per_heliostat=self.heliostat_group.active_surface_points.shape[
                indices.number_of_surface_points_dimension
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

        if self.occlusion_active:
            # Compute the heliostat primitives.
            all_heliostat_surfaces = []
            for group in self.scenario.heliostat_field.heliostat_groups:
                surfaces = group.surface_points + group.positions.unsqueeze(1)
                mask = group.active_heliostats_mask.long()
                is_active = mask > 0
                if is_active.any():
                    # Position of the first copy of each heliostat in the active arrays.
                    first_copy = torch.cumsum(mask, dim=0) - mask
                    surfaces[is_active] = group.active_surface_points[
                        first_copy[is_active]
                    ]
                else:
                    log.warning(
                        "Not all heliostat groups have been aligned yet. "
                        "Using horizontal heliostats as blocking planes."
                    )
                all_heliostat_surfaces.append(surfaces)
            heliostat_surfaces = torch.cat(all_heliostat_surfaces).detach()

            (
                self.heliostat_primitives_corners,
                self.heliostat_primitives_spans,
                self.heliostat_primitives_normals,
            ) = blocking.create_heliostat_occlusion_primitives(
                surface_points=heliostat_surfaces,
                device=device,
            )

            (
                self.tower_primitives_corners,
                self.tower_primitives_spans,
                self.tower_primitives_normals,
            ) = blocking.create_tower_occlusion_primitives(device=device)

        if dni is not None:
            # Calculate surface area per heliostat.
            heliostat_surface_area = (
                self.heliostat_group.heliostat_dimensions[indices.heliostat_width]
                * self.heliostat_group.heliostat_dimensions[indices.heliostat_height]
            )
            # Calculate ray magnitude.
            power_single_heliostat = dni * heliostat_surface_area
            rays_per_heliostat = (
                self.heliostat_group.surface_points.shape[1]
                * self.scenario.light_sources.light_source_list[
                    indices.first_light_source
                ].number_of_rays
            )
            self.ray_magnitude = power_single_heliostat / rays_per_heliostat
        else:
            self.ray_magnitude = 1.0

    def _iterate_batches(self, device: torch.device) -> Iterator[torch.Tensor]:
        """
        Yield sorted local active-heliostat indices for each batch on this rank.

        Sorting keeps distortions, surface points, normals and the ray-to-heliostat
        mapping in the same order (the old boolean-mask indexing implied sorted order
        for the geometry but sampler order for the distortions/mapping).
        """
        rank_indices = torch.as_tensor(
            self.distortions_sampler.rank_indices, dtype=torch.long, device=device
        )
        for start in range(0, rank_indices.numel(), self.batch_size):
            yield rank_indices[start : start + self.batch_size].sort().values

    def _run_batch(self, batch_number: int, *args):
        self._will_recompute = self.use_checkpointing and torch.is_grad_enabled()
        if self._will_recompute:
            return checkpoint(
                self._trace_batch, batch_number, *args, use_reentrant=False
            )
        return self._trace_batch(batch_number, *args)

    @torch.no_grad()
    def _get_shading_candidates(
        self,
        batch_number: int,
        batch_indices: torch.Tensor,
        surface_points: torch.Tensor,
        sun_directions: torch.Tensor,
        batch_global_indices: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Shading primitive candidates for one batch, from the cache if available."""
        cache = getattr(self, "_shading_cache", None)
        if cache is not None and batch_number in cache:
            cached_batch_indices, candidates = cache[batch_number]
            assert torch.equal(cached_batch_indices, batch_indices), (
                "Shading cache does not match the current batching "
                "(batch size, rank or world size changed?)."
            )
            return candidates

        candidates = blocking.lbvh_traversal(
            lbvh=self.scenario.heliostat_field.heliostat_field_lbvh,
            points_at_ray_origins=surface_points,
            ray_directions=sun_directions,
            primitives_corners=self.heliostat_primitives_corners,
            heliostat_owners=batch_global_indices,
            max_intersection_distances=torch.full(
                sun_directions.shape[:3], float("inf"), device=device
            ),
            device=device,
        )
        if cache is not None:
            cache[batch_number] = (batch_indices.clone(), candidates)
        return candidates

    def _trace_batch(
        self,
        batch_number: int,
        batch_indices: torch.Tensor,
        batch_global_indices: torch.Tensor,
        incident_ray_directions: torch.Tensor,
        target_area_indices: torch.Tensor,
        ray_extinction_factor: float,
        mirror_reflectivity: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Trace one batch of heliostats.

        Pure function of its inputs (plus read-only attributes), so it can be
        recomputed by ``torch.utils.checkpoint`` in the backward pass.

        Returns
        -------
        torch.Tensor
            Bitmaps for the batch. Shape is ``[batch, bitmap_u, bitmap_e]``.
        torch.Tensor
            Intercept factor for the batch. Shape is ``[batch]``.
        """
        surface_points = self.heliostat_group.active_surface_points[batch_indices]
        surface_normals = self.heliostat_group.active_surface_normals[batch_indices]
        incident = incident_ray_directions[batch_indices]
        batch_target_area_indices = target_area_indices[batch_indices]

        # Reflection per batch instead of for the whole group up front.
        preferred_reflection_directions = geometry.reflect(
            incident_ray_directions=incident[:, None, :],
            reflection_surface_normals=surface_normals,
        )

        # Index distortions directly (no DataLoader collation copies kept alive).
        rays = self.scatter_rays(
            distortion_u=self.distortions_dataset.distortions_u[batch_indices].to(
                device
            ),
            distortion_e=self.distortions_dataset.distortions_e[batch_indices].to(
                device
            ),
            original_ray_direction=preferred_reflection_directions,
            device=device,
        )

        number_of_planar = self.scenario.solar_tower.number_of_target_areas_per_type[
            indices.planar_target_areas
        ]
        planar_active_mask = batch_target_area_indices < number_of_planar
        cylindrical_active_mask = ~planar_active_mask

        number_of_heliostats = batch_indices.numel()
        number_of_rays = self.light_source.number_of_rays
        number_of_points = self.heliostat_group.surface_points.shape[1]
        number_of_rays_per_heliostat = number_of_rays * number_of_points
        shape = (number_of_heliostats, number_of_rays, number_of_points)

        intersection_distances_target = torch.zeros(shape, device=device)
        bitmap_intensities = torch.zeros(shape, device=device)
        bitmap_intersections_e = torch.zeros(shape, device=device)
        bitmap_intersections_u = torch.zeros(shape, device=device)

        if planar_active_mask.any():
            (
                bitmap_intersections_e[planar_active_mask],
                bitmap_intersections_u[planar_active_mask],
                intersection_distances_target[planar_active_mask],
                bitmap_intensities[planar_active_mask],
            ) = geometry.line_plane_intersections(
                rays=rays,
                points_at_ray_origins=surface_points,
                target_areas=self.scenario.solar_tower.target_areas[
                    indices.planar_target_areas
                ],  # type: ignore[arg-type]
                target_area_indices=batch_target_area_indices,
                active_mask=planar_active_mask,
                bitmap_resolution=self.bitmap_resolution,
                device=device,
            )

        if cylindrical_active_mask.any():
            (
                bitmap_intersections_e[cylindrical_active_mask],
                bitmap_intersections_u[cylindrical_active_mask],
                intersection_distances_target[cylindrical_active_mask],
                bitmap_intensities[cylindrical_active_mask],
            ) = geometry.line_cylinder_intersections(
                rays=rays,
                points_at_ray_origins=surface_points,
                target_areas=self.scenario.solar_tower.target_areas[
                    indices.cylindrical_target_areas
                ],  # type: ignore[arg-type]
                target_area_indices=batch_target_area_indices - number_of_planar,
                active_mask=cylindrical_active_mask,
                bitmap_resolution=self.bitmap_resolution,
                device=device,
            )

        blocked = torch.zeros(shape, device=device)
        shaded = torch.zeros(shape, device=device)

        if self.occlusion_active:
            ray_direction_sun = (-incident)[:, None, None, :].expand(
                -1, number_of_rays, number_of_points, -1
            )

            cached = self._blocking_index_cache.pop(batch_number, None)
            if cached is not None:
                # Backward recompute: reuse the forward result (same alignment -> identical).
                filtered_blocking_primitive_indices = cached
            else:
                with torch.no_grad():
                    filtered_blocking_primitive_indices = blocking.lbvh_traversal(
                        lbvh=self.scenario.heliostat_field.heliostat_field_lbvh,
                        points_at_ray_origins=surface_points,
                        ray_directions=rays.ray_directions,
                        primitives_corners=self.heliostat_primitives_corners,
                        heliostat_owners=batch_global_indices,
                        max_intersection_distances=intersection_distances_target,
                        device=device,
                    )
                if self._will_recompute:
                    self._blocking_index_cache[batch_number] = (
                        filtered_blocking_primitive_indices
                    )

            with torch.no_grad():
                filtered_shading_primitive_indices = self._get_shading_candidates(
                    batch_number=batch_number,
                    batch_indices=batch_indices,
                    surface_points=surface_points,
                    sun_directions=ray_direction_sun,
                    batch_global_indices=batch_global_indices,
                    device=device,
                )

            if filtered_blocking_primitive_indices.numel() > 0:
                blocked = blocking.soft_ray_occlusion_mask(
                    ray_origins=surface_points,
                    ray_directions=rays.ray_directions,
                    primitives_corners=self.heliostat_primitives_corners[
                        filtered_blocking_primitive_indices
                    ],
                    primitives_spans=self.heliostat_primitives_spans[
                        filtered_blocking_primitive_indices
                    ],
                    primitives_normals=self.heliostat_primitives_normals[
                        filtered_blocking_primitive_indices
                    ],
                    epsilon=1e-12,
                    softness=1000.0,
                )

            shaded = blocking.soft_ray_occlusion_mask(
                ray_origins=surface_points,
                ray_directions=ray_direction_sun,
                primitives_corners=torch.cat(
                    [
                        self.heliostat_primitives_corners[
                            filtered_shading_primitive_indices
                        ],
                        self.tower_primitives_corners,
                    ]
                ),
                primitives_spans=torch.cat(
                    [
                        self.heliostat_primitives_spans[
                            filtered_shading_primitive_indices
                        ],
                        self.tower_primitives_spans,
                    ]
                ),
                primitives_normals=torch.cat(
                    [
                        self.heliostat_primitives_normals[
                            filtered_shading_primitive_indices
                        ],
                        self.tower_primitives_normals,
                    ]
                ),
                epsilon=1e-12,
                softness=1000.0,
            )

        cosine_factor = (
            (incident[:, None, :3] * surface_normals[..., :3])
            .sum(dim=-1)
            .abs()
            .clamp(0.0, 1.0)[:, None, :]
        )

        # Atmospheric attenuation after Leary & Hankins (1979), MIRVAL.
        d = intersection_distances_target
        atmospheric_attenuation = torch.where(
            d <= 1000,
            0.99321 - 1.176e-4 * d + 1.97e-8 * d**2,
            torch.exp(-1.106e-5 * d),
        )

        optical_loss_factor = (
            mirror_reflectivity * atmospheric_attenuation * cosine_factor
        )

        intensities = (
            bitmap_intensities
            * (1 - blocked)
            * (1 - shaded)
            * (1 - ray_extinction_factor)
            * optical_loss_factor
        )

        bitmaps = self.bilinear_splatting(
            bitmap_intersections_e=bitmap_intersections_e,
            bitmap_intersections_u=bitmap_intersections_u,
            absolute_intensities=intensities,
            device=device,
        )

        with torch.no_grad():
            intercept_factors = bitmaps.sum(dim=(1, 2)) / (
                rays.ray_magnitudes * optical_loss_factor
            ).sum(dim=(1, 2)).clamp_min(1e-12)

        return bitmaps, intercept_factors

    # ------------------------------------------------------------------ #
    # Public tracing APIs
    # ------------------------------------------------------------------ #
    def trace_rays(
        self,
        incident_ray_directions: torch.Tensor,
        active_heliostats_mask: torch.Tensor,
        target_area_indices: torch.Tensor,
        ray_extinction_factor: float = 0.0,
        mirror_reflectivity: float = 0.935,
        shading_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-heliostat bitmaps (unchanged API). Memory scales with N_active * H * W."""
        self._blocking_index_cache.clear()
        device = get_device(device=device)
        assert torch.equal(
            self.heliostat_group.active_heliostats_mask, active_heliostats_mask
        ), "Some heliostats were not aligned and cannot be raytraced."
        number_of_active = int(active_heliostats_mask.sum())
        flux_distributions = torch.zeros(
            (
                number_of_active,
                int(self.bitmap_resolution[indices.unbatched_bitmap_u]),
                int(self.bitmap_resolution[indices.unbatched_bitmap_e]),
            ),
            device=device,
        )
        intercept_factor = torch.zeros(number_of_active, device=device)
        global_active_indices = (
            torch.arange(
                active_heliostats_mask.numel(), device=device
            ).repeat_interleave(active_heliostats_mask.long())
            + self._group_offset
        )

        self._shading_cache = shading_cache

        for batch_number, batch_indices in enumerate(self._iterate_batches(device)):
            bitmaps, intercept = self._run_batch(
                batch_number,
                batch_indices,
                global_active_indices[batch_indices],
                incident_ray_directions,
                target_area_indices,
                ray_extinction_factor,
                mirror_reflectivity,
                device,
            )
            # In-place index_copy_ only saves the index for backward.
            flux_distributions.index_copy_(0, batch_indices, bitmaps)
            intercept_factor.index_copy_(0, batch_indices, intercept)

        return flux_distributions, intercept_factor

    def trace_rays_per_target(
        self,
        incident_ray_directions: torch.Tensor,
        active_heliostats_mask: torch.Tensor,
        target_area_indices: torch.Tensor,
        ray_extinction_factor: float = 0.0,
        mirror_reflectivity: float = 0.935,
        shading_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Trace rays and accumulate flux directly per target area.

        Never materializes the ``[N_active, H, W]`` per-heliostat tensor; memory is
        ``O(number_of_target_areas * H * W)`` plus one (checkpointed) batch.
        Only heliostats sampled on this rank contribute, i.e. the result equals
        ``get_bitmaps_per_target(trace_rays(...)[sampler_indices], ...)``.

        shading_cache : dict[int, tuple[torch.Tensor, torch.Tensor]] | None
            Cache of shading primitive candidates per batch, valid for ONE incident ray
            direction. Filled on first use, reused afterwards. None -> traverse every call.

        Returns
        -------
        torch.Tensor
            Flux per target area. Shape is ``[number_of_target_areas, bitmap_u, bitmap_e]``.
        torch.Tensor
            Intercept factor per active heliostat (zero for heliostats of other ranks).
            Shape is ``[number_of_active_heliostats]``.
        """
        self._blocking_index_cache.clear()
        device = get_device(device=device)
        assert torch.equal(
            self.heliostat_group.active_heliostats_mask, active_heliostats_mask
        ), "Some heliostats were not aligned and cannot be raytraced."

        flux_per_target = torch.zeros(
            (
                int(self.scenario.solar_tower.number_of_target_areas_per_type.sum()),
                int(self.bitmap_resolution[indices.unbatched_bitmap_u]),
                int(self.bitmap_resolution[indices.unbatched_bitmap_e]),
            ),
            device=device,
        )
        intercept_factor = torch.zeros(int(active_heliostats_mask.sum()), device=device)
        global_active_indices = torch.nonzero(active_heliostats_mask, as_tuple=True)[0]
        target_area_indices_long = target_area_indices.long()

        self._shading_cache = shading_cache

        for batch_number, batch_indices in enumerate(self._iterate_batches(device)):
            bitmaps, intercept = self._run_batch(
                batch_number,
                batch_indices,
                global_active_indices[batch_indices],
                incident_ray_directions,
                target_area_indices,
                ray_extinction_factor,
                mirror_reflectivity,
                device,
            )
            flux_per_target.index_add_(
                0, target_area_indices_long[batch_indices], bitmaps
            )
            intercept_factor.index_copy_(0, batch_indices, intercept)

        return flux_per_target, intercept_factor

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
            Shape is ``[number_of_active_heliostats, number_of_rays, number_of_combined_surface_normals_all_facets]``.
        distortion_e : torch.Tensor
            The distortions in east direction (angles for scattering).
            Shape is ``[number_of_active_heliostats, number_of_rays, number_of_combined_surface_normals_all_facets]``.
        original_ray_direction : torch.Tensor
            The ray direction around which to scatter.
            Shape is ``[number_of_active_heliostats, number_of_combined_surface_normals_all_facets, 4]``.
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

        rotations = transforms.rotate_distortions(
            u=distortion_u, e=distortion_e, device=device
        )

        scattered_rays = (
            rotations
            @ original_ray_direction.unsqueeze(indices.number_rays_per_point).unsqueeze(
                -1
            )
        ).squeeze(-1)

        return Rays(
            ray_directions=scattered_rays,
            ray_magnitudes=torch.full(
                (scattered_rays.shape[: indices.ray_directions]),
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
            Shape is ``[number_of_active_heliostats, bitmap_resolution_e, bitmap_resolution_u]``.
        target_area_indices : torch.Tensor
            The mapping from heliostat to target area.
            Shape is ``[number_of_active_heliostats]``.
        device : torch.device | None
            The device on which to perform computations or load tensors and models (default is None).
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            Bitmaps per target area.
            Shape is ``[number_of_target_areas, bitmap_resolution_e, bitmap_resolution_u]``.
        """
        device = get_device(device=device)

        group_bitmaps_per_target = torch.zeros(
            (
                int(self.scenario.solar_tower.number_of_target_areas_per_type.sum()),
                int(self.bitmap_resolution[indices.unbatched_bitmap_u]),
                int(self.bitmap_resolution[indices.unbatched_bitmap_e]),
            ),
            device=device,
        )
        for index in range(group_bitmaps_per_target.shape[0]):
            mask = target_area_indices == index
            if mask.any():
                group_bitmaps_per_target[index] = bitmaps_per_heliostat[mask].sum(
                    dim=indices.heliostat_dimension
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
            Shape is ``[number_of_active_heliostats, number_of_rays, number_of_surface_points]``.
        bitmap_intersections_u : torch.Tensor
            The up-component bitmap coordinates of ray intersections.
            Shape is ``[number_of_active_heliostats, number_of_rays, number_of_surface_points]``.
        absolute_intensities : torch.Tensor
            The intensity of each ray at its intersection point.
            Shape is ``[number_of_active_heliostats, number_of_rays, number_of_surface_points]``.
        device : torch.device | None
            The device on which to perform computations or load tensors and models.
            If None, ``ARTIST`` will automatically select the most appropriate
            device (CUDA or CPU) based on availability and OS.

        Returns
        -------
        torch.Tensor
            The flux density bitmaps, one per active heliostat.
            Shape is ``[number_of_active_heliostats, bitmap_resolution_u, bitmap_resolution_e]``.
        """
        bitmap_height = self.bitmap_resolution[indices.unbatched_bitmap_u]
        bitmap_width = self.bitmap_resolution[indices.unbatched_bitmap_e]
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
