from typing import List, Tuple

import torch
from .light_source import ALightSource
from ...util import utils


class Sun(ALightSource):
    """
    Implementation of the sun as a specific light source.

    See Also
    :class:ALightSource : Reference to the parent class
    """

    def __init__(
        self,
        dist_type: str,
        ray_count: int,
        mean: List[float],
        cov: List[float],
        device: torch.device,
    ) -> None:
        """
        Initialize a sun as a light source

        Parameters
        ----------
        dist_type : str
            Type of the distribution to be implemented.

        ray_count : int
            The amount of rays send out.

        mean : List[float]
            The mean that describes the normal distribution.

        cov : List[float]
            The covariance that describes the normal distribution.

        device : torch.device
            Specifies the device type responsible to load tensors into memory.

        Raises
        ------
        ValueError
            If the chosen dist_type is unknown
        """
        super(Sun, self).__init__()
        self.dist_type: str = dist_type
        self.num_rays: int = ray_count

        dtype = torch.get_default_dtype()
        if self.dist_type == "Normal":
            self.mean = torch.tensor(
                mean,
                dtype=dtype,
                device=device,
            )
            self.cov = torch.tensor(
                cov,
                dtype=dtype,
                device=device,
            )
            self.distribution = torch.distributions.MultivariateNormal(
                self.mean, self.cov
            )

        elif self.dist_type == "Pillbox":
            raise ValueError("Not Implemented Yet")
        else:
            raise ValueError("unknown light distribution type")

    def sample(
        self,
        num_rays_on_hel: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample rays from a given distribution.

        Parameters
        ----------
        num_rays_on_hel : int
            Defines the number of rays on the heliostat.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Returns the distortion in x and y direction.

        Raises
        ------
        ValueError
            Raised if the distribution type is not valid, currently only the normal distribution is implemented.
        """
        if self.dist_type == "Normal":
            distortion_x_dir, distortion_y_dir = (
                self.distribution.sample(
                    (self.num_rays, num_rays_on_hel),
                )
                .transpose(0, 1)
                .permute(2, 1, 0)
            )
            return distortion_x_dir, distortion_y_dir
        else:
            raise ValueError("unknown light distribution type")

    def compute_rays(
        self,
        plane_normal: torch.Tensor,
        plane_point: torch.Tensor,
        ray_directions: torch.Tensor,
        surface_points: torch.Tensor,
        distortion_x_dir: torch.Tensor,
        distortion_y_dir: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the scattered rays for points on a surface.

        Parameters
        ----------
        plane_normal : torch.Tensor
            The normal vector of the intersecting plane (normal vector of the receiver).

        plane_point : torch.Tensor
            Point on the plane (Center point of the receiver).

        ray_directions : torch.Tensor
            Directions of the reflected sun light.

        surface_points : torch.Tensor
            Points on which the rays are to be reflected.

        distortion_x_dir : torch.Tensor
            Distortion of the rays in x direction.

        distortion_y_dir : torch.Tensor
            Distortion of the rays in y direction.

        Returns
        -------
        torch.Tensor
            Returns the scattered rays.

        """
        intersections = self.line_plane_intersections(
            plane_normal=plane_normal,
            plane_point=plane_point,
            ray_directions=ray_directions,
            surface_points=surface_points,
        )
        as_ = intersections
        has = as_ - surface_points
        # TODO Wieder der Vektor von vorher?
        #      Evtl. ist dieser Aufruf von `line_plane_intersections` unnötig
        has = has / torch.linalg.norm(has, dim=1).unsqueeze(-1)

        # rotate: Calculate 3D rotationmatrix in heliostat system.
        # 1 axis is pointing towards the receiver, the other are orthogonal
        rotates_x = torch.stack(
            [has[:, 0], has[:, 1], has[:, 2]],
            -1,
        )
        rotates_x = rotates_x / torch.linalg.norm(rotates_x, dim=-1).unsqueeze(-1)
        rotates_y = torch.stack(
            [
                has[:, 1],
                -has[:, 0],
                torch.zeros(has.shape[:1], device=as_.device),
            ],
            -1,
        )
        rotates_y = rotates_y / torch.linalg.norm(rotates_y, dim=-1).unsqueeze(-1)
        rotates_z = torch.stack(
            [
                has[:, 2] * has[:, 0],
                has[:, 2] * has[:, 1],
                -has[:, 0] ** 2 - has[:, 1] ** 2,
            ],
            -1,
        )
        rotates_z = rotates_z / torch.linalg.norm(rotates_z, dim=-1).unsqueeze(-1)
        rotates = torch.hstack([rotates_x, rotates_y, rotates_z]).reshape(
            rotates_x.shape[0],
            rotates_x.shape[1],
            -1,
        )
        inv_rot = torch.linalg.inv(rotates)  # inverse matrix
        # rays_tmp = torch.tensor(ha, device=device)
        # print(rays_tmp.shape)

        # rays_tmp: first rotate aimpoint in right coord system,
        # apply xi, yi distortion, rotate back
        rotated_has = torch.matmul(rotates, has.unsqueeze(-1))

        # rays = rotated_has.transpose(0, -1).transpose(1, -1)
        rot_y = self.Ry(distortion_x_dir, mat=(rotated_has.to(torch.float)))
        rot_z = self.Rz(distortion_y_dir, rot_y).transpose(0, -1).squeeze(0)
        rays = (
            torch.matmul(inv_rot.to(torch.float), rot_z)
            .transpose(0, -1)
            .transpose(1, -1)
        )

        return rays

    def line_plane_intersections(
        self,
        plane_normal: torch.Tensor,
        plane_point: torch.Tensor,
        ray_directions: torch.Tensor,
        surface_points: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> torch.Tensor:
        """
        Compute line-plane intersections of ray directions and the (receiver) plane

        Parameters
        ----------
        plane_normal : torch.Tensor
            The normal vector of the intersecting plane (normal vector of the receiver).

        plane_point : torch.Tensor
            Point on the plane (center point of the receiver).

        ray_directions : torch.Tensor
            Direction of the reflected sun light.

        surface_points : torch.Tensor
            Points on which the rays are to be reflected.

        epsilon : float
            small value, upper limit.

        Returns
        -------
        torch.Tensor
            Return the intersections of the lines and plane.

        Raises
        ------
        RuntimeError
            Raised when there are no intersections between the line and the plane.
        """
        ndotu = ray_directions.matmul(plane_normal)
        if (torch.abs(ndotu) < epsilon).any():
            raise RuntimeError("no intersection or line is within plane")
        ds = (plane_point - surface_points).matmul(plane_normal.to(torch.float)) / ndotu

        return surface_points + ray_directions * ds.unsqueeze(-1)

    def reflect_rays_(self, rays: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
        """
        Reflect incoming rays according to a noraml vector.

        Parameters
        ----------
        rays : torch.Tensor
            The incoming rays (from the sun) to be reflected.

        normals : Torch.Tensor
            Surface normals on which the rays are reflected.

        Returns
        -------
        torch.Tensor
            Returns the reflected rays.
        """
        return rays - 2 * utils.batch_dot(rays, normals) * normals

    def Ry(self, alpha: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
        """
        Create rotation matrices and rotate the input along the y-axis in the heliostat coordination system.

        Parameters
        ----------
        alpha : torch.Tensor

        mat : torch.Tensor

        Returns
        -------
        torch.Tensor
            Returns rotation matrix.
        """
        zeros = torch.zeros_like(alpha)
        coss = torch.cos(alpha)
        sins = torch.sin(alpha)
        rots_x = torch.stack([coss, zeros, sins], -1)
        rots_y = torch.stack([zeros, torch.ones_like(alpha), zeros], -1)
        rots_z = torch.stack([-sins, zeros, coss], -1)
        rots = torch.stack([rots_x, rots_y, rots_z], -1).reshape(rots_x.shape + (-1,))
        return torch.matmul(rots, mat)

    def Rz(self, alpha: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
        """
        Create rotation matrices and rotate the input along the z-axis in the heliostat coordination system.

        Parameters
        ----------
        alpha : torch.Tensor

        mat : torch.Tensor

        Returns
        -------
        torch.Tensor
            Returns rotation matrix.
        """
        zeros = torch.zeros_like(alpha)
        coss = torch.cos(alpha)
        sins = torch.sin(alpha)
        rots_x = torch.stack([coss, -sins, zeros], -1)
        rots_y = torch.stack([sins, coss, zeros], -1)
        rots_z = torch.stack([zeros, zeros, torch.ones_like(alpha)], -1)
        rots = torch.stack([rots_x, rots_y, rots_z], -1).reshape(rots_x.shape + (-1,))
        return torch.matmul(rots, mat)

    def sample_bitmap(
        self,
        dx_ints: torch.Tensor,
        dy_ints: torch.Tensor,
        indices: torch.Tensor,
        plane_x: float,
        plane_y: float,
        bitmap_height: int,
        bitmap_width: int,
    ) -> torch.Tensor:
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
            resolution of the resulting bitmap (x direction) -> height

        bitmap_width : int
            resolution of the resulting bitmap (y direction) -> width

        Returns
        -------
        torch.Tensor
            Returns the flux density distribution of the reflected rays on the receiver
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