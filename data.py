import os

import torch as th

from render import Renderer
import utils


def create_target(
        H,
        ENV,
        sun_origin,
        sun_origin_normed,
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

            sun_origin,
            ENV.sun.num_rays,
            ENV.sun.mean,
            ENV.sun.cov,
        )

    H_aligned = H.align(sun_origin_normed, ENV.receiver_center)
    R = Renderer(H_aligned, ENV)
    if save_path:
        utils.save_target(
            *(
                target_save_data
                + (
                    R.xi,
                    R.yi,

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


def generate_dataset(cfg, H, ENV, save_dir, writer=None):
    if save_dir:
        save_path = os.path.join(save_dir, 'target.pt')
    else:
        save_path = None

    device = H.device
    sun_origins = cfg.AC.SUN.ORIGIN
    if not isinstance(sun_origins[0], list):
        sun_origins = [sun_origins]
    sun_origins = th.tensor(
        sun_origins, dtype=th.get_default_dtype(), device=device)
    sun_origins_normed = \
        sun_origins / th.linalg.norm(sun_origins, dim=1).unsqueeze(-1)

    targets = None
    for (i, (sun_origin, sun_origin_normed)) in enumerate(zip(
            sun_origins,
            sun_origins_normed,
    )):
        target_bitmap = create_target(
            H,
            ENV,
            sun_origin,
            sun_origin_normed,
            save_path=save_path,
        )
        if targets is None:
            targets = th.empty(
                (len(sun_origins),) + target_bitmap.shape,
                dtype=th.get_default_dtype(),
                device=device,
            )
        targets[i] = target_bitmap
        if writer:
            writer.add_image(
                f"target_{i}/originals",
                utils.colorize(target_bitmap),
            )

        # Plot and Save Stuff
        # ===================
        # print(H._normals_orig.shape)
        # im = plt.imshow(target_bitmap.detach().cpu(),cmap = "jet")
    return targets, sun_origins_normed
