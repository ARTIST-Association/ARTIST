from typing import Optional, Tuple, Union
import torch
import torch as th

from environment import Environment
from heliostat_models import AbstractHeliostat


def Rx(alpha: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    zeros = th.zeros_like(alpha)
    coss = th.cos(alpha)
    sins = th.sin(alpha)
    rots_x = th.stack([th.ones_like(alpha), zeros, zeros], -1)
    rots_y = th.stack([zeros, coss, -sins], -1)
    rots_z = th.stack([zeros, sins, coss], -1)
    rots = th.stack([rots_x, rots_y, rots_z], -1).reshape(rots_x.shape + (-1,))
    return th.matmul(rots, mat)


def Ry(alpha: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    zeros = th.zeros_like(alpha)
    coss = th.cos(alpha)
    sins = th.sin(alpha)
    rots_x = th.stack([coss, zeros, sins], -1)
    rots_y = th.stack([zeros, th.ones_like(alpha), zeros], -1)
    rots_z = th.stack([-sins, zeros, coss], -1)
    rots = th.stack([rots_x, rots_y, rots_z], -1).reshape(rots_x.shape + (-1,))
    return th.matmul(rots, mat)


def Rz(alpha: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    zeros = th.zeros_like(alpha)
    coss = th.cos(alpha)
    sins = th.sin(alpha)
    rots_x = th.stack([coss, -sins, zeros], -1)
    rots_y = th.stack([sins, coss, zeros], -1)
    rots_z = th.stack([zeros, zeros, th.ones_like(alpha)], -1)
    rots = th.stack([rots_x, rots_y, rots_z], -1).reshape(rots_x.shape + (-1,))
    return th.matmul(rots, mat)


def line_plane_intersections(
        planeNormal: torch.Tensor,
        planePoint: torch.Tensor,
        rayDirections: torch.Tensor,
        rayPoints: torch.Tensor,
        epsilon: float = 1e-6,
) -> torch.Tensor:
    ndotu = rayDirections.matmul(planeNormal)
    if (th.abs(ndotu) < epsilon).any():
        raise RuntimeError("no intersection or line is within plane")
    ds = (planePoint - rayPoints).matmul(planeNormal) / ndotu

    return rayPoints + rayDirections * ds.unsqueeze(-1)


def compute_ray_directions(
        planeNormal: torch.Tensor,
        planePoint: torch.Tensor,
        ray_directions: torch.Tensor,
        hel_in_field: torch.Tensor,
        xi: torch.Tensor,
        yi: torch.Tensor,
) -> torch.Tensor:
    intersections = line_plane_intersections(
        planeNormal, planePoint, ray_directions, hel_in_field)
    as_ = intersections
    has = as_ - hel_in_field
    # TODO Wieder der Vektor von vorher?
    #      Evtl. ist dieser Aufruf von `line_plane_intersections` unnötig
    has = has / th.linalg.norm(has, dim=1).unsqueeze(-1)

    # rotate: Calculate 3D rotationmatrix in heliostat system.
    # 1 axis is pointing towards the receiver, the other are orthogonal
    rotates_x = th.stack(
        [has[:, 0], has[:, 1], has[:, 2]],
        -1,
    )
    rotates_x = rotates_x / th.linalg.norm(rotates_x, dim=-1).unsqueeze(-1)
    rotates_y = th.stack(
        [
            has[:, 1],
            -has[:, 0],
            th.zeros(has.shape[:1], device=as_.device),
        ],
        -1,
    )
    rotates_y = rotates_y / th.linalg.norm(rotates_y, dim=-1).unsqueeze(-1)
    rotates_z = th.stack(
        [
            has[:, 2] * has[:, 0],
            has[:, 2] * has[:, 1],
            -has[:, 0]**2 - has[:, 1]**2,
        ],
        -1,
    )
    rotates_z = rotates_z / th.linalg.norm(rotates_z, dim=-1).unsqueeze(-1)
    rotates = th.hstack([rotates_x, rotates_y, rotates_z]).reshape(
        rotates_x.shape[0],
        rotates_x.shape[1],
        -1,
    )
    inv_rot = th.linalg.inv(rotates)  # inverse matrix
    # rays_tmp = th.tensor(ha, device=device)
    # print(rays_tmp.shape)

    # rays_tmp: first rotate aimpoint in right coord system,
    # apply xi, yi distortion, rotate back
    rotated_has = th.matmul(rotates, has.unsqueeze(-1))

    # rays = rotated_has.transpose(0, -1).transpose(1, -1)
    rot_y = Ry(xi, rotated_has)
    rot_z = Rz(yi, rot_y).transpose(0, -1).squeeze(0)
    rays = th.matmul(inv_rot, rot_z).transpose(0, -1).transpose(1, -1)

    return rays
    # return rotated_has


def compute_receiver_intersections(
        planeNormal: torch.Tensor,
        planePoint: torch.Tensor,
        ray_directions: torch.Tensor,
        hel_in_field: torch.Tensor,
) -> torch.Tensor:
    # Execute the kernel
    intersections = line_plane_intersections(
        planeNormal, planePoint, ray_directions, hel_in_field, epsilon=1e-6)
    # print(intersections)
    return intersections


def sample_bitmap(
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
    x_inds = th.hstack([x_inds_1, x_inds_2, x_inds_3, x_inds_4]).long().ravel()
    del x_inds_1
    del x_inds_2
    del x_inds_3
    del x_inds_4

    y_inds = th.hstack([y_inds_1, y_inds_2, y_inds_3, y_inds_4]).long().ravel()
    del y_inds_1
    del y_inds_2
    del y_inds_3
    del y_inds_4

    ints = th.hstack([ints_1, ints_2, ints_3, ints_4]).ravel()
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
    total_bitmap = th.zeros(
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


def normalize_bitmap(
        bitmap: torch.Tensor,
        num_rays: Union[int, torch.Tensor],
        planex: float,
        planey: float,
        bitmap_height: int,
        bitmap_width: int,
) -> torch.Tensor:
    plane_area = planex * planey
    num_pixels = bitmap_height * bitmap_width
    plane_area_per_pixel = plane_area / num_pixels

    return bitmap / (num_rays * plane_area_per_pixel)


class Renderer(object):
    def __init__(
            self,
            heliostat: AbstractHeliostat,
            environment: Environment,
    ) -> None:
        self.H = heliostat
        self.ENV = environment
        self.redraw_random_variables: bool = \
            self.ENV.cfg.SUN.REDRAW_RANDOM_VARIABLES
        # plotter.plot_raytracer(
        #     self.H.discrete_points,
        #     self.ENV.receiver_center,
        #     self.ENV.sun_direction,
        # )
        # plotter.plot_normal_vectors(self.H.discrete_points, self.H._normals)

        if not self.redraw_random_variables:
            self.xi, self.yi = self.ENV.sun.sample(len(self.H))

    def render(
            self,
            heliostat: Optional[AbstractHeliostat] = None,
            return_extras: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[
            torch.Tensor,
            Tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ],
    ]:
        if heliostat is None:
            heliostat = self.H
        if self.redraw_random_variables:
            xi, yi = self.ENV.sun.sample(len(heliostat))
        else:
            xi = self.xi
            yi = self.yi
        surface_points, ray_directions = heliostat()
        # TODO Maybe another name?

        distorted_ray_directions = compute_ray_directions(
            self.ENV.receiver_plane_normal,  # Intersection plane
            self.ENV.receiver_center,  # Point on plane
            ray_directions,  # line directions
            surface_points,  # points on line
            xi,
            yi
        )

        intersections = compute_receiver_intersections(
            self.ENV.receiver_plane_normal,
            self.ENV.receiver_center,
            distorted_ray_directions,
            surface_points,
        )

        dx_ints = (  # TODO Make dependent on receiver plane
            intersections[:, :, 0]
            + self.ENV.receiver_plane_x / 2
            - self.ENV.receiver_center[0]
        )
        dy_ints = (
            intersections[:, :, 2]
            + self.ENV.receiver_plane_y / 2
            - self.ENV.receiver_center[2]
        )
        indices = (
            (-1 <= dx_ints)
            & (dx_ints < self.ENV.receiver_plane_x + 1)
            & (-1 <= dy_ints)
            & (dy_ints < self.ENV.receiver_plane_y + 1)
        )
        total_bitmap = sample_bitmap(
            dx_ints,
            dy_ints,
            indices,
            self.ENV.receiver_plane_x,
            self.ENV.receiver_plane_y,
            self.ENV.receiver_resolution_x,
            self.ENV.receiver_resolution_y,
        )

        total_bitmap = normalize_bitmap(
            total_bitmap,
            xi.numel(),
            self.ENV.receiver_plane_x,
            self.ENV.receiver_plane_y,
            self.ENV.receiver_resolution_x,
            self.ENV.receiver_resolution_y,
        )

        target_num_missed = indices.numel() - indices.count_nonzero()

        if target_num_missed > 0:
            print(
                'Missed for target:', target_num_missed.detach().cpu().item())

        if return_extras:
            return (
                total_bitmap,
                (
                    distorted_ray_directions,
                    dx_ints,
                    dy_ints,
                    indices,
                    xi,
                    yi,
                ),
            )
        return total_bitmap
