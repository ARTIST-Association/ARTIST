import os

import torch as th

from render import Renderer
import utils


def create_target(
        H,
        ENV,
        sun_direction,
        save_path=None,
):
    device = H.device
    if save_path:
        target_save_data = (
            H.position_on_field,
            th.tensor(
                H.cfg.IDEAL.NORMAL_VECS,
                dtype=th.get_default_dtype(),
                device=device,
            ),
            H.discrete_points,
            H.normals,
            None,  # TODO

            ENV.receiver_center,
            ENV.receiver_plane_x,
            ENV.receiver_plane_y,
            ENV.receiver_plane_normal,
            None,  # TODO

            sun_direction,
            ENV.sun.num_rays,
            ENV.sun.mean,
            ENV.sun.cov,
        )

    H_aligned = H.align(sun_direction, ENV.receiver_center)
    R = Renderer(H_aligned, ENV)
    if save_path:
        if R.redraw_random_variables:
            xi = None
            yi = None
        else:
            xi = R.xi
            yi = R.yi

        utils.save_target(
            *(
                target_save_data
                + (
                    xi,
                    yi,

                    # We need the heliostat to be aligned here.
                    H_aligned.get_ray_directions(),
                    H_aligned.discrete_points,
                    save_path,
                )
            )
        )

    # Render Step
    # ===========
    target_bitmap = R.render()
    return target_bitmap


@th.no_grad()
def generate_dataset(H, ENV, sun_directions, save_dir, writer=None, prefix=''):

    if save_dir:
        save_path = os.path.join(save_dir, 'target.pt')
    else:
        save_path = None

    device = H.device
    
    targets = None
    for i, sun_direction in enumerate(sun_directions):
        target_bitmap = create_target(
            H,
            ENV,
            sun_direction,
            save_path=save_path,
        )
        if targets is None:
            targets = th.empty(
                (len(sun_directions),) + target_bitmap.shape,
                dtype=th.get_default_dtype(),
                device=device,
            )
        targets[i] = target_bitmap
        if writer:
            writer.add_image(
                f"{prefix}target_{i}/originals",
                utils.colorize(target_bitmap),
            )
    return targets





