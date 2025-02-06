from typing import TYPE_CHECKING, Iterator, Union

if TYPE_CHECKING:
    from artist.scenario import Scenario

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

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
            self.distortions_u[idx],
            self.distortions_e[idx],
        )


class RestrictedDistributedSampler(Sampler):
    """
    Initializes a custom distributed sampler.

    The ``DistributedSampler`` from torch replicates samples if the size of the dataset
    is smaller than the world size to assign data to each rank. This custom sampler
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
    shuffle : bool
        Shuffled sampling or sequential.
    seed : int
        The seed to replicate random sampling.
    active_replicas : int
        Number of processes that will receive data.
    number_of_samples_per_rank : int
        The number of samples per rank.

    Methods
    -------
    set_seed()
        Set the seed for reproducible shuffling across epochs.

    See Also
    --------
    :class:`torch.utils.data.Sampler` : The parent class.
    """

    def __init__(
        self, dataset: Dataset, world_size: int = 1, rank: int = 0, shuffle: bool = True
    ) -> None:
        """
        Set up a custom distributed sampler to assign data to each rank.

        Parameters
        ----------
        dataset : Dataset
            The dataset from which the samples are taken.
        world_size : int
            The world size or total number of processes (default: 1).
        rank : int
            The rank of the current process (default: 0).
        shuffle : bool
            Shuffled sampling or sequential (default: True).
        """
        super().__init__(dataset)
        self.number_of_samples = len(dataset)
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.seed = 0

        # Adjust num_replicas if dataset is smaller than world_size
        self.active_replicas = min(self.number_of_samples, self.world_size)

        # Only assign data to first `active_replicas` ranks
        self.number_of_samples_per_rank = (
            self.number_of_samples // self.active_replicas
            if self.rank < self.active_replicas
            else 0
        )

    def set_seed(self, seed: int = 0) -> None:
        """
        Set the seed for reproducible shuffling across epochs.

        Parameters
        ----------
        seed: int
            The seed for the random generator.
        """
        self.seed = seed

    def __iter__(self) -> Iterator[int]:
        """
        Generate a sequence of indices for the current rank's portion of the dataset.

        Returns
        -------
        Iterator[int]
            An iterator over (shuffled) indices for the current rank.
        """
        # Generate indices and shuffle them if shuffle=True
        indices = list(range(self.number_of_samples))
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed)
            indices = torch.randperm(len(indices), generator=g).tolist()

        # Split indices only among active ranks
        if self.rank < self.active_replicas:
            start_idx = self.rank * self.number_of_samples_per_rank
            end_idx = start_idx + self.number_of_samples_per_rank
            return iter(indices[start_idx:end_idx])
        else:
            return iter([])

    def __len__(self) -> int:
        """
        Determine the number of samples assigned to the current rank.

        Returns
        -------
        int
            The number of samples for the current rank.
        """
        return (
            self.number_of_samples_per_rank if self.rank < self.active_replicas else 0
        )


class HeliostatRayTracer:
    """
    Implement the functionality for heliostat raytracing.

    Attributes
    ----------
    heliostat : Heliostat
        The heliostat considered for raytracing.
    target_area : TargetArea
        The target area considered for raytracing.
    world_size : int
        The world size i.e., the overall number of processors / ranks.
    rank : int
        The rank, i.e., individual process ID.
    number_of_surface_points : int
        The number of surface points on the heliostat.
    distortions_dataset : DistortionsDataset
        The dataset containing the distortions for ray scattering.
    distortions_loader : DataLoader
        The dataloader that loads the distortions.
    bitmap_resolution_e : int
        The resolution of the bitmap in the east dimension (default: 256).
    bitmap_resolution_u : int
        The resolution of the bitmap in the up dimension (default: 256).

    Methods
    -------
    trace_rays()
        Perform heliostat raytracing.
    scatter_rays()
        Scatter the reflected rays around the preferred ray direction.
    sample_bitmap()
        Sample a bitmap (flux density distribution) of the reflected rays on the target area.
    normalize_bitmap()
        Normalize a bitmap.
    """

    def __init__(
        self,
        scenario: "Scenario",
        aim_point_area: str = "receiver",
        heliostat_index: int = 0,
        world_size: int = 1,
        rank: int = 0,
        batch_size: int = 1,
        random_seed: int = 7,
        shuffle: bool = True,
        bitmap_resolution_e: int = 256,
        bitmap_resolution_u: int = 256,
    ) -> None:
        """
        Initialize the heliostat raytracer.

        "Heliostat"-tracing is one kind of raytracing applied in ARTIST. For this kind of raytracing,
        the rays are initialized on the heliostat. The rays originate in the discrete surface points.
        There they are multiplied, distorted, and scattered, and then they are sent to the target area.
        Letting the rays originate on the heliostat drastically reduces the number of rays that need
        to be traced.

        Parameters
        ----------
        scenario : Scenario
            The scenario used to perform raytracing.
        aim_point_area : str
            The target area on in which the aimpoint is supposed to be.
        heliostat_index : int
            Index of heliostat from the heliostat list (default: 0).
        world_size : int
            The world size (default: 1).
        rank : int
            The rank (default: 0).
        batch_size : int
            The batch size used for raytracing (default: 1).
        random_seed : int
            The random seed used for generating the distortions (default: 7).
        shuffle : bool
            A boolean flag indicating whether to shuffle the data (default: False).
        bitmap_resolution_e : int
            The resolution of the bitmap in the east dimension (default: 256).
        bitmap_resolution_u : int
            The resolution of the bitmap in the up dimension (default: 256).
        """
        self.heliostat = scenario.heliostats.heliostat_list[heliostat_index]
        self.target_area = next(
            area
            for area in scenario.target_areas.target_area_list
            if area.name == aim_point_area
        )
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
        # Create restricted distributed sampler.
        self.distortions_sampler = RestrictedDistributedSampler(
            dataset=self.distortions_dataset,
            world_size=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
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
        device: Union[torch.device, str] = "cuda",
    ) -> torch.Tensor:
        """
        Perform heliostat raytracing.

        Scatter the rays according to the distortions, calculate the line plane intersection, and calculate the
        resulting bitmap on the target area.

        Parameters
        ----------
        incident_ray_direction : torch.Tensor
            The direction of the incident ray as seen from the heliostat.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        torch.Tensor
            The resulting bitmap.
        """
        device = torch.device(device)

        final_bitmap = torch.zeros(
            (self.bitmap_resolution_u, self.bitmap_resolution_e), device=device
        )

        self.heliostat.set_preferred_reflection_direction(rays=-incident_ray_direction)

        self.distortions_sampler.set_seed(0)
        for batch_u, batch_e in self.distortions_loader:
            rays = self.scatter_rays(batch_u, batch_e, device)

            intersections = raytracing_utils.line_plane_intersections(
                ray_directions=rays.ray_directions,
                plane_normal_vectors=self.target_area.normal_vector,
                plane_center=self.target_area.center,
                points_at_ray_origin=self.heliostat.current_aligned_surface_points,
            )

            dx_ints = (
                intersections[:, :, :, 0]
                + self.target_area.plane_e / 2
                - self.target_area.center[0]
            )
            dy_ints = (
                intersections[:, :, :, 2]
                + self.target_area.plane_u / 2
                - self.target_area.center[2]
            )

            indices = (
                (-1 <= dx_ints)
                & (dx_ints < self.target_area.plane_e + 1)
                & (-1 <= dy_ints)
                & (dy_ints < self.target_area.plane_u + 1)
            )

            total_bitmap = self.sample_bitmap(dx_ints, dy_ints, indices, device=device)

            final_bitmap = final_bitmap + total_bitmap

        return final_bitmap

    def scatter_rays(
        self,
        distortion_u: torch.Tensor,
        distortion_e: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> Rays:
        """
        Scatter the reflected rays around the preferred ray direction.

        Parameters
        ----------
        distortion_u : torch.Tensor
            The distortions in up direction (angles for scattering).
        distortion_e : torch.Tensor
            The distortions in east direction (angles for scattering).
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        Rays
            Scattered rays around the preferred direction.
        """
        device = torch.device(device)
        ray_directions = self.heliostat.preferred_reflection_direction[
            :, :, :3
        ] / torch.linalg.norm(
            self.heliostat.preferred_reflection_direction[:, :, :3],
            ord=2,
            dim=-1,
            keepdim=True,
        )

        ray_directions = torch.cat(
            (ray_directions, torch.zeros(4, ray_directions.size(1), 1, device=device)),
            dim=-1,
        )

        rotations = utils.rotate_distortions(
            u=distortion_u, e=distortion_e, device=device
        )

        scattered_rays = (rotations @ ray_directions.unsqueeze(-1)).squeeze(-1)

        return Rays(
            ray_directions=scattered_rays,
            ray_magnitudes=torch.ones(
                scattered_rays.size(dim=0),
                scattered_rays.size(dim=1),
                scattered_rays.size(dim=2),
                device=device,
            ),
        )

    def sample_bitmap(
        self,
        dx_ints: torch.Tensor,
        dy_ints: torch.Tensor,
        indices: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> torch.Tensor:
        """
        Sample a bitmap (flux density distribution) of the reflected rays on the target area.

        Parameters
        ----------
        dx_ints : torch.Tensor
            x position of intersection with the target area of shape (N, 1), where N is the resolution of
            the target area along the x-axis.
        dy_ints : torch.Tensor
            y position of intersection with the target area of shape (N, 1), where N is the resolution of
            the target area along the y-axis.
        indices : torch.Tensor
            Index of the pixel.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        torch.Tensor
            The flux density distribution of the reflected rays on the target area.
        """
        device = torch.device(device)

        x_ints = dx_ints[indices] / self.target_area.plane_e * self.bitmap_resolution_e
        y_ints = dy_ints[indices] / self.target_area.plane_u * self.bitmap_resolution_u

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

        # The lower-valued neighboring pixels (for x this corresponds to 4
        # and 3, for y to 1 and 4).
        x_inds_low = x_ints.floor().long()
        y_inds_low = y_ints.floor().long()
        # The higher-valued neighboring pixels (for x this corresponds to 2
        # and 3, for y to 1 and 2).
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

        x_inds_3 = x_inds_high
        y_inds_3 = y_inds_high
        ints_3 = x_ints_high * y_ints_high

        x_inds_4 = x_inds_low
        y_inds_4 = y_inds_high
        ints_4 = x_ints_low * y_ints_high

        # Combine all indices and intensities in the correct order.
        x_inds = torch.hstack([x_inds_4, x_inds_3, x_inds_2, x_inds_1]).long().ravel()

        y_inds = torch.hstack([y_inds_4, y_inds_3, y_inds_2, y_inds_1]).long().ravel()

        ints = torch.hstack([ints_4, ints_3, ints_2, ints_1]).ravel()

        # For distribution, we regard even those neighboring pixels that are
        # _not_ part of the image. That is why here, we set up a mask to
        # choose only those indices that are actually in the bitmap (i.e. we
        # prevent out-of-bounds access).
        indices = (
            (0 <= x_inds)
            & (x_inds < self.bitmap_resolution_e)
            & (0 <= y_inds)
            & (y_inds < self.bitmap_resolution_u)
        )

        # Flux density map for heliostat field
        total_bitmap = torch.zeros(
            [self.bitmap_resolution_u, self.bitmap_resolution_e],
            dtype=dx_ints.dtype,
            device=device,
        )
        # Add up all distributed intensities in the corresponding indices.
        total_bitmap.index_put_(
            (
                self.bitmap_resolution_u - 1 - y_inds[indices],
                self.bitmap_resolution_e - 1 - x_inds[indices],
            ),
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

        plane_area = self.target_area.plane_e * self.target_area.plane_u
        num_pixels = bitmap_height * bitmap_width
        plane_area_per_pixel = plane_area / num_pixels

        return bitmap / (
            self.distortions_dataset.distortions_u.numel() * plane_area_per_pixel
        )
