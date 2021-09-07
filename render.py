import torch as th

def batch_dot(x, y):
    return (x * y).sum(-1).unsqueeze(-1)

def reflect_rays_(rays, normals):
    return rays - 2 * batch_dot(rays, normals) * normals

def reflect_rays(rays, normals):
    normals = normals / th.linalg.norm(normals, dim=-1).unsqueeze(-1)
    return reflect_rays_(rays, normals)

def Rx(alpha, mat):
    zeros = th.zeros_like(alpha)
    coss = th.cos(alpha)
    sins = th.sin(alpha)
    rots_x = th.hstack(list(map(lambda t: t.unsqueeze(-1), [th.ones_like(alpha), zeros, zeros])))
    rots_y = th.hstack(list(map(lambda t: t.unsqueeze(-1), [zeros, coss, -sins])))
    rots_z = th.hstack(list(map(lambda t: t.unsqueeze(-1), [zeros, sins, coss])))
    rots = th.hstack([rots_x, rots_y, rots_z]).reshape(rots_x.shape[0], rots_x.shape[1], -1)
    return th.matmul(rots,mat)

def Ry(alpha, mat):
    zeros = th.zeros_like(alpha)
    coss = th.cos(alpha)
    sins = th.sin(alpha)
    rots_x = th.hstack(list(map(lambda t: t.unsqueeze(-1), [coss, zeros, sins])))
    rots_y = th.hstack(list(map(lambda t: t.unsqueeze(-1), [zeros, th.ones_like(alpha), zeros])))
    rots_z = th.hstack(list(map(lambda t: t.unsqueeze(-1), [-sins, zeros, coss])))
    rots = th.hstack([rots_x, rots_y, rots_z]).reshape(rots_x.shape[0], rots_x.shape[1], -1)
    return th.matmul(rots,mat.transpose(0, -1))

def Rz(alpha, mat):
    zeros = th.zeros_like(alpha)
    coss = th.cos(alpha)
    sins = th.sin(alpha)
    rots_x = th.hstack(list(map(lambda t: t.unsqueeze(-1), [coss, -sins, zeros])))
    rots_y = th.hstack(list(map(lambda t: t.unsqueeze(-1), [sins, coss, zeros])))
    rots_z = th.hstack(list(map(lambda t: t.unsqueeze(-1), [zeros, zeros, th.ones_like(alpha)])))
    rots = th.hstack([rots_x, rots_y, rots_z]).reshape(rots_x.shape[0], rots_x.shape[1], -1)
    return th.matmul(rots,mat)

def LinePlaneCollision(planeNormal, planePoint, rayDirections, rayPoints, epsilon=1e-6):

    ndotu = rayDirections.matmul(planeNormal)
    if (th.abs(ndotu) < epsilon).any():
        raise RuntimeError("no intersection or line is within plane")

    ws = rayPoints - planePoint
    sis = -ws.matmul(planeNormal) / ndotu
    Psis = ws + sis.unsqueeze(-1) * rayDirections + planePoint
    return Psis

def compute_receiver_intersections(
        planeNormal,
        planePoint,
        ray_directions,
        hel_in_field,
        xi,
        yi,
):
    intersections = LinePlaneCollision(planeNormal, planePoint, ray_directions, hel_in_field)
    as_ = intersections
    has = as_-hel_in_field

    # TODO Max: remove/use for ray reflection instead
    # rotate: Calculate 3D rotationmatrix in heliostat system. 1 axis is pointin towards the receiver, the other are orthogonal
    rotates_x = th.hstack(list(map(lambda t: t.unsqueeze(-1), [has[:, 0],has[:, 1],has[:, 2]])))
    rotates_x = rotates_x / th.linalg.norm(rotates_x, dim=-1).unsqueeze(-1)
    rotates_y = th.hstack(list(map(lambda t: t.unsqueeze(-1), [has[:, 1],-has[:, 0],th.zeros(has.shape[:1], device=as_.device)])))
    rotates_y = rotates_y / th.linalg.norm(rotates_y, dim=-1).unsqueeze(-1)
    rotates_z = th.hstack(list(map(lambda t: t.unsqueeze(-1), [has[:, 2]*has[:, 0],has[:, 2]*has[:, 1],-has[:, 0]**2-has[:, 1]**2])))
    rotates_z = rotates_z / th.linalg.norm(rotates_z, dim=-1).unsqueeze(-1)
    rotates = th.hstack([rotates_x, rotates_y, rotates_z]).reshape(rotates_x.shape[0], rotates_x.shape[1], -1)
    inv_rot = th.linalg.inv(rotates) #inverse matrix
    # rays_tmp = th.tensor(ha, device=device)
    # print(rays_tmp.shape)

    # rays_tmp: first rotate aimpoint in right coord system, aplay xi,yi distortion, rotate back
    rotated_has = th.matmul(rotates, has.unsqueeze(-1))
    # rays = rotated_has.transpose(0, -1).transpose(1, -1)
    rays = th.matmul(inv_rot,
                                Rz(yi,
                                  Ry(xi,
                                      rotated_has
                                      )
                                  ).transpose(0, -1)
                                ).transpose(0, -1).transpose(1, -1)


    # rays = rays.to(th.float32)

    # Execute the kernel
    intersections = LinePlaneCollision(planeNormal, planePoint, rays, hel_in_field, epsilon=1e-6)
    # print(intersections)
    return intersections

def sample_bitmap_(dx_ints, dy_ints, indices, planex, planey, bitmap_height, bitmap_width):

    x_ints = dx_ints[indices]/planex*bitmap_height
    y_ints = dy_ints[indices]/planey*bitmap_width

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
    x_inds_low = x_ints.long()
    y_inds_low = y_ints.long()
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
    # Normalize
    if len(ints) > 0:
        ints = ints / th.max(ints)

    # For distribution, we regard even those neighboring pixels that are
    # _not_ part of the image. That is why here, we set up a mask to
    # choose only those indices that are actually in the bitmap (i.e. we
    # prevent out-of-bounds access).
    indices = (0 <= x_inds) & (x_inds < bitmap_width) & (0 <= y_inds) & (y_inds < bitmap_height)

    total_bitmap = th.zeros([bitmap_height, bitmap_width], dtype=th.float32, device=dx_ints.device) # Flux density map for heliostat field
    # Add up all distributed intensities in the corresponding indices.
    total_bitmap.index_put_(
        (x_inds[indices], y_inds[indices]),
        ints[indices],
        accumulate=True,
    )
    return total_bitmap

class Renderer(object):
    def __init__(self, Heliostat, Environment):
        self.H = Heliostat
        self.ENV = Environment
        from_sun = self.H.position_on_field - self.ENV.sun_origin #TODO Evtl auf H.Discrete Points umstellen
        from_sun /= from_sun.norm()
        from_sun = from_sun.unsqueeze(0)
        self.ray_directions = reflect_rays_(from_sun, self.H.normals)#
        self.xi, self.yi = self.ENV.Sun.sample_() # Evtl. in render jedesmal aufrufen
    def render(self):
        # TODO Max: use for reflection instead
        
        intersections = compute_receiver_intersections(
            self.ENV.receiver_plane_normal, #Intersection plane
            self.ENV.receiver_center, # Point on plane
            self.ray_directions,  # line directions
            self.H.discrete_points, # points on line
            self.xi, 
            self.yi
            )
        
        dx_ints = intersections[:, :, 1] + self.ENV.receiver_plane_x/2 - self.ENV.receiver_center[1]
        dy_ints = intersections[:, :, 2] + self.ENV.receiver_plane_y/2 - self.ENV.receiver_center[2]
        indices = (-1 <= dx_ints) & (dx_ints < self.ENV.receiver_plane_x + 1) & (-1 <= dy_ints) & (dy_ints < self.ENV.receiver_plane_y + 1)
        total_bitmap = sample_bitmap_(
            dx_ints, 
            dy_ints, 
            indices, 
            self.ENV.receiver_plane_x, 
            self.ENV.receiver_plane_y, 
            self.ENV.receiver_resolution_x, 
            self.ENV.receiver_resolution_y
            )
        # target_num_missed = indices.numel() - indices.count_nonzero()
        # print('Missed for target:', target_num_missed.detach().cpu().item())
        return total_bitmap
    
    