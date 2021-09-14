import argparse
import os

import matplotlib.pyplot as plt
import torch as th

import nurbs
import utils


class Target:
    __slots__ = [
        'heliostat_origin_center',
        'heliostat_face_normal',
        'heliostat_points',
        'heliostat_normals',
        'heliostat_up_dir',

        'receiver_origin_center',
        'receiver_width',
        'receiver_height',
        'receiver_normal',
        'receiver_up_dir',

        'sun',
        'num_rays',
        'mean',
        'cov',
        'xi',
        'yi',
    ]

    def __init__(self, checkpoint_dict):
        # We iterate the slots instead of the dictionary so missing data
        # is caught easily and redundant data is ignored.
        for key in self.__slots__:
            value = checkpoint_dict[key]
            setattr(self, key, value)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model_path',
        type=str,
        help='Model to load for testing.',
    )
    parser.add_argument(
        'target_path',
        type=str,
        help='Target to load for testing.',
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Seed to use for RNG initialization.',
    )
    parser.add_argument(
        '--use_gpu',
        type=bool,
        default=th.cuda.is_available(),
        help='Whether to use the GPU.',
    )
    parser.add_argument(
        '--bitmap_dims',
        type=int,
        nargs=2,
        default=[256, 256],
        help='Dimensions of the bitmap to generate.',
    )
    parser.add_argument(
        '--num_tests',
        type=int,
        default=32,
        help='How many different sun positions to test.',
    )

    args = parser.parse_args()
    return args


def load_model(path, device):
    cp = th.load(path, map_location=device)
    use_splines = 'ctrl_points' in cp
    if use_splines:
        degree_x = cp['degree_x']
        degree_y = cp['degree_y']
        ctrl_points = cp['ctrl_points']
        ctrl_weights = cp['ctrl_weights']
        knots_x = cp['knots_x']
        knots_y = cp['knots_y']

        model = nurbs.NURBSSurface(
            degree_x,
            degree_y,
            ctrl_points,
            ctrl_weights,
            knots_x,
            knots_y,
        )
    elif 'heliostat_normals' in cp:
        model = cp['heliostat_normals']
    else:
        if 'opt' in cp:
            raise ValueError(
                f'trying to load an optimizer checkpoint; '
                f'please remove the suffix "_opt" from {path}'
            )
        else:
            raise ValueError(f'unknown checkpoint format in {path}')
    model_xyi = (cp['xi'], cp['yi'])
    return model, model_xyi, use_splines


def load_target(path, device):
    cp = th.load(path, map_location=device)
    target = Target(cp)
    return target


def main():
    args = parse_args()

    th.manual_seed(args.seed)
    device = th.device(
        'cuda'
        if args.use_gpu and th.cuda.is_available()
        else 'cpu'
    )
    bitmap_height, bitmap_width = args.bitmap_dims

    target = load_target(args.target_path, device)
    model, (model_xi, model_yi), use_splines = load_model(
        args.model_path, device)
    if use_splines:
        rows, cols = model.control_points.shape[:-1]
        eval_points = utils.initialize_spline_eval_points(rows, cols, device)

    target_sun = target.sun / target.sun.norm()
    for test in range(args.num_tests):
        sun = th.rand_like(target.sun)
        sun /= sun.norm()

        from_sun = target.heliostat_origin_center - sun
        from_sun /= from_sun.norm()
        from_sun = from_sun.unsqueeze(0)

        target_hel_coords = th.stack(utils.heliostat_coord_system(
            target.heliostat_origin_center,
            sun,
            target.receiver_origin_center,
            verbose=False,
        ))
        target_hel_rotated = utils.rotate_heliostat(
            target.heliostat_points, target_hel_coords)
        target_hel_in_field = (
            target_hel_rotated
            + target.heliostat_origin_center
        )

        target_normals = utils.rotate_heliostat(
            target.heliostat_normals, target_hel_coords)
        target_normals /= target_normals.norm(dim=-1).unsqueeze(-1)

        target_ray_directions = utils.reflect_rays_(from_sun, target_normals)

        intersections = utils.compute_receiver_intersections(
            target.receiver_normal,
            target.receiver_origin_center,
            target_ray_directions,
            target_hel_in_field,
            target.xi,
            target.yi,
        )
        dx_ints, dy_ints, indices = utils.get_intensities_and_sampling_indices(
            intersections,
            target.receiver_origin_center,
            target.receiver_width,
            target.receiver_height,
        )
        target_bitmap = utils.sample_bitmap_(
            dx_ints,
            dy_ints,
            indices,
            target.receiver_width,
            target.receiver_height,
            bitmap_height,
            bitmap_width,
        )
        target_num_missed = indices.numel() - indices.count_nonzero()

        # Model
        if use_splines:
            hel_origin, normals = (
                nurbs.calc_normals_and_surface_slow(
                    eval_points[:, 0],
                    eval_points[:, 1],
                    model.degree_x,
                    model.degree_y,
                    model.control_points,
                    model.control_point_weights,
                    model.knots_x,
                    model.knots_y,
                )
            )

            hel_rotated = utils.rotate_heliostat(hel_origin, target_hel_coords)
            hel_in_field = hel_rotated + target.heliostat_origin_center

            normals = utils.rotate_heliostat(normals, target_hel_coords)
            normals = normals / normals.norm(dim=-1).unsqueeze(-1)
            ray_directions = utils.reflect_rays_(from_sun, normals)
        else:
            hel_in_field = target_hel_in_field
            normals = utils.rotate_heliostat(model, target_hel_coords)
            normals /= normals.norm(dim=-1).unsqueeze(-1)
            ray_directions = utils.reflect_rays_(from_sun, normals)

        intersections = utils.compute_receiver_intersections(
            target.receiver_normal,
            target.receiver_origin_center,
            ray_directions,
            hel_in_field,
            model_xi,
            model_yi,
        )
        dx_ints, dy_ints, indices = utils.get_intensities_and_sampling_indices(
            intersections,
            target.receiver_origin_center,
            target.receiver_width,
            target.receiver_height,
        )
        bitmap = utils.sample_bitmap_(
            dx_ints,
            dy_ints,
            indices,
            target.receiver_width,
            target.receiver_height,
            bitmap_height,
            bitmap_width,
        )

        num_missed = indices.numel() - indices.count_nonzero()
        ray_dir_diff = utils.calc_ray_diffs(
            ray_directions,
            target_ray_directions,
        )
        normal_diff = utils.calc_ray_diffs(
            normals,
            target_normals,
        )
        normal_cos_diff = th.mean(utils.batch_dot(
            normals,
            target_normals,
        ))
        loss = th.nn.functional.mse_loss(bitmap, target_bitmap, 0.1)

        # if first_plot:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8.5, 7.2))

        ax1.set_title('Target')
        ax1.imshow(
            target_bitmap.detach().cpu().numpy(),
            cmap='jet',
        )

        ax2.set_title('Prediction')
        ax2.imshow(
            bitmap.detach().cpu().numpy(),
            cmap='jet',
        )

        ax3.set_title('Difference target/prediction')
        im = ax3.imshow(
            (
                th.abs(target_bitmap - bitmap)
            ).detach().cpu().numpy(),
            cmap='gray',
        )
        fig.colorbar(im, ax=ax3)

        ax4.set_title('Difference normalized by target intensity')
        im = ax4.imshow(
            (
                th.abs(target_bitmap - bitmap)
                / target_bitmap
            ).detach().cpu().numpy(),
            cmap='gray',
        )
        fig.colorbar(im, ax=ax4)
        plt.savefig(os.path.join('images', f'test_{test}.png'))
        plt.close(fig)

        print(
            f'test {test}: '
            f'sun {sun.detach().cpu().numpy()}, '
            f'loss {loss.detach().cpu().numpy()}, '
            f'target missed {target_num_missed.detach().cpu().numpy()}, '
            f'missed {num_missed.detach().cpu().numpy()}, '
            f'ray dir diff {ray_dir_diff.detach().cpu().numpy()}, '
            f'normal diff {normal_diff.detach().cpu().numpy()}, '
            f'normal cos diff {normal_cos_diff.detach().cpu().numpy()}'
        )
        # print()


if __name__ == '__main__':
    main()
