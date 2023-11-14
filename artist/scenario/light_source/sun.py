from typing import Tuple

import torch
from .ALightSource import ALightSource
from ...util import utils


class Sun(ALightSource):
    def __init__(self, dist_type, ray_count, mean, cov, device):
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
                self.mean, self.cov)

        elif self.dist_type == "Pillbox":
            raise ValueError("Not Implemented Yet")
        else:
            raise ValueError("unknown light distribution type")
    
    def sample(self, num_rays_on_hel: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.dist_type == "Normal":
            distortion_x_dir, distortion_y_dir = self.distribution.sample(
                (self.num_rays, num_rays_on_hel),
            ).transpose(0, 1).permute(2, 1, 0)
            return distortion_x_dir, distortion_y_dir
        else:
            raise ValueError('unknown light distribution type')

    def Ry(self, alpha: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(alpha)
        coss = torch.cos(alpha)
        sins = torch.sin(alpha)
        rots_x = torch.stack([coss, zeros, sins], -1)
        rots_y = torch.stack([zeros, torch.ones_like(alpha), zeros], -1)
        rots_z = torch.stack([-sins, zeros, coss], -1)
        rots = torch.stack([rots_x, rots_y, rots_z], -1).reshape(rots_x.shape + (-1,))
        return torch.matmul(rots, mat)


    def Rz(self, alpha: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(alpha)
        coss = torch.cos(alpha)
        sins = torch.sin(alpha)
        rots_x = torch.stack([coss, -sins, zeros], -1)
        rots_y = torch.stack([sins, coss, zeros], -1)
        rots_z = torch.stack([zeros, zeros, torch.ones_like(alpha)], -1)
        rots = torch.stack([rots_x, rots_y, rots_z], -1).reshape(rots_x.shape + (-1,))
        return torch.matmul(rots, mat)


    def line_plane_intersections(self, 
        planeNormal: torch.Tensor,
        planePoint: torch.Tensor,
        rayDirections: torch.Tensor,
        rayPoints: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> torch.Tensor:
        ndotu = rayDirections.matmul(planeNormal)
        if (torch.abs(ndotu) < epsilon).any():
            raise RuntimeError("no intersection or line is within plane")
        ds = (planePoint - rayPoints).matmul(planeNormal) / ndotu

        return rayPoints + rayDirections * ds.unsqueeze(-1)

    def compute_rays(self,
        planeNormal: torch.Tensor,
        planePoint: torch.Tensor,
        ray_directions: torch.Tensor,
        hel_in_field: torch.Tensor,
        distortion_x_dir: torch.Tensor,
        distortion_y_dir: torch.Tensor,
    ) -> torch.Tensor:
        intersections = self.line_plane_intersections(planeNormal=planeNormal, 
                                                      planePoint=planePoint,
                                                      rayDirections=ray_directions, 
                                                      rayPoints=hel_in_field)
        as_ = intersections
        has = as_ - hel_in_field
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
                -has[:, 0]**2 - has[:, 1]**2,
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
        rot_y = self.Ry(alpha=distortion_x_dir, mat=(rotated_has.to(torch.float)))
        rot_z = self.Rz(distortion_y_dir, rot_y).transpose(0, -1).squeeze(0)
        rays = torch.matmul(inv_rot.to(torch.float), rot_z).transpose(0, -1).transpose(1, -1)

        return rays
        # return rotated_has

    def reflect_rays_(self, 
                      rays: torch.Tensor, 
                      normals: torch.Tensor) -> torch.Tensor:
        return rays - 2 * utils.batch_dot(rays, normals) * normals
    
    def compute_receiver_intersections(self,
        planeNormal: torch.Tensor,
        planePoint: torch.Tensor,
        ray_directions: torch.Tensor,
        hel_in_field: torch.Tensor,
    ) -> torch.Tensor:
        # Execute the kernel
        intersections = self.line_plane_intersections(
            planeNormal, planePoint, ray_directions, hel_in_field, epsilon=1e-6)
        # print(intersections)
        return intersections
    
    def line_plane_intersections(self,
        planeNormal: torch.Tensor,
        planePoint: torch.Tensor,
        rayDirections: torch.Tensor,
        rayPoints: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> torch.Tensor:
        ndotu = rayDirections.matmul(planeNormal)
        if (torch.abs(ndotu) < epsilon).any():
            raise RuntimeError("no intersection or line is within plane")
        ds = (planePoint - rayPoints).matmul(planeNormal.to(torch.float)) / ndotu

        return rayPoints + rayDirections * ds.unsqueeze(-1)

    def sample_bitmap(self,
        dx_ints: torch.Tensor,
        dy_ints: torch.Tensor,
        indices: torch.Tensor,
        planex: float,
        planey: float,
        bitmap_height: int,
        bitmap_width: int,
    ) -> torch.Tensor:

        x_ints = dx_ints[indices] / planex * bitmap_height
        y_ints = dy_ints[indices] / planey * bitmap_width

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

