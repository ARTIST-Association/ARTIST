import argparse
import os

import matplotlib.pyplot as plt
import torch as th

import defaults
from environment import Environment
from heliostat_models import Heliostat
from nurbs import NURBSSurface
from nurbs_heliostat import NURBSHeliostat
from render import Renderer
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


def _parse_bool(string):
    assert string == 'False' or string == 'True', \
        'please only use "False" or "True" as boolean arguments.'
    return string != 'False'


def parse_args(cfg):
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
        '--num_tests',
        type=int,
        default=32,
        help='How many different sun positions to test.',
    )

    # Values with defaults given by configuration.
    parser.add_argument(
        '--seed',
        type=int,
        default=cfg.SEED,
        help='Seed to use for RNG initialization.',
    )
    parser.add_argument(
        '--use_gpu',
        type=_parse_bool,
        default=cfg.USE_GPU and th.cuda.is_available(),
        help='Whether to use the GPU.',
    )
    parser.add_argument(
        '--bitmap_dims',
        type=int,
        nargs=2,
        default=[
            cfg.AC.RECEIVER.RESOLUTION_X,
            cfg.AC.RECEIVER.RESOLUTION_Y,
        ],
        help='Dimensions of the bitmap to generate.',
    )

    args = parser.parse_args()
    return args


def load_model(path, device):
    cp = th.load(path, map_location=device)
    use_splines = 'control_points' in cp
    if use_splines:
        degree_x = cp['degree_x']
        degree_y = cp['degree_y']
        ctrl_points = cp['control_points']
        ctrl_weights = cp['control_point_weights']
        knots_x = cp['knots_x']
        knots_y = cp['knots_y']

        model = NURBSSurface(
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
    cfg = defaults.get_cfg_defaults()

    args = parse_args(cfg)

    th.manual_seed(args.seed)
    device = th.device(
        'cuda'
        if args.use_gpu and th.cuda.is_available()
        else 'cpu'
    )

    # Set up config from arguments
    bitmap_height, bitmap_width = args.bitmap_dims
    cfg.merge_from_list([
        'SEED',
        args.seed,
        'USE_GPU',
        args.use_gpu,
        'AC.RECEIVER.RESOLUTION_X',
        bitmap_height,
        'AC.RECEIVER.RESOLUTION_Y',
        bitmap_width,
    ])

    # Set up config from target
    target = load_target(args.target_path, device)
    cfg.merge_from_list([
        'H.IDEAL.POSITION_ON_FIELD',
        target.heliostat_origin_center.tolist(),
        'H.IDEAL.NORMAL_VECS',
        target.heliostat_face_normal.tolist(),
        # We assign this as well just in case.
        # 'H.NURBS.NORMAL_VECS',
        # target.heliostat_face_normal.tolist(),
        # TODO Missing `heliostat_up_dir`

        'AC.RECEIVER.CENTER',
        target.receiver_origin_center.tolist(),
        'AC.RECEIVER.PLANE_X',
        target.receiver_width,
        'AC.RECEIVER.PLANE_Y',
        target.receiver_height,
        'AC.RECEIVER.PLANE_NORMAL',
        target.receiver_normal.tolist(),
        # TODO Missing `receiver_up_dir`

        'AC.SUN.DIRECTION',
        target.sun.tolist(),
        'AC.SUN.GENERATE_N_RAYS',
        target.num_rays,
        'AC.SUN.NORMAL_DIST.MEAN',
        target.mean.tolist(),
        'AC.SUN.NORMAL_DIST.COV',
        target.cov.tolist(),
    ])
    target_cfg = cfg.clone()

    # Set up target
    target_heliostat = Heliostat(target_cfg.H, device)

    # Set up model
    model, (model_xi, model_yi), use_splines = load_model(
        args.model_path, device)

    if use_splines:
        rows, cols = model.control_points.shape[:-1]

        cfg.merge_from_list([
            'NURBS.SET_UP_WITH_KNOWLEDGE',
            True,
            'NURBS.SPLINE_DEGREE',
            model.degree_x,
            'H.NURBS.ROWS',
            rows,
            'H.NURBS.COLS',
            cols,
        ])

        heliostat = NURBSHeliostat(cfg.H, cfg.NURBS, device)
        heliostat.ctrl_points_xy = model.control_points[:, :, :-1]
        heliostat.ctrl_points_z = model.control_points[:, :, -1:]
        heliostat.ctrl_weights = model.control_point_weights
        heliostat.knots_x = model.knots_x
        heliostat.knots_y = model.knots_y
    else:
        heliostat = Heliostat(cfg.H, cfg.NURBS, device)
        heliostat._normals = model

    for test in range(args.num_tests):
        sun = th.rand_like(target.sun)
        # Allow negative x and y values
        sun[:-1] -= 0.5
        sun /= th.linalg.norm(sun)

        target_cfg.merge_from_list([
            'AC.SUN.DIRECTION',
            sun.tolist(),
        ])

        env = Environment(target_cfg.AC, device)
        target_heliostat_aligned = target_heliostat.align(
            env.sun_direction, env.receiver_center)
        heliostat_aligned = heliostat.align(
            env.sun_direction, env.receiver_center)

        target_renderer = Renderer(target_heliostat_aligned, env)
        renderer = Renderer(heliostat_aligned, env)
        for renderer_ in [target_renderer, renderer]:
            renderer_.xi = model_xi
            renderer_.yi = model_yi

        target_bitmap = target_renderer.render()
        bitmap = renderer.render()

        # ray_dir_diff = utils.calc_ray_diffs(
        #     ray_directions,
        #     target_ray_directions,
        # )
        # normal_diff = utils.calc_ray_diffs(
        #     normals,
        #     target_normals,
        # )
        # normal_cos_diff = th.mean(utils.batch_dot(
        #     normals,
        #     target_normals,
        # ))
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
            f'loss {loss.detach().cpu().numpy()}'
            # f'target missed {target_num_missed.detach().cpu().numpy()}, '
            # f'missed {num_missed.detach().cpu().numpy()}, '
            # f'ray dir diff {ray_dir_diff.detach().cpu().numpy()}, '
            # f'normal diff {normal_diff.detach().cpu().numpy()}, '
            # f'normal cos diff {normal_cos_diff.detach().cpu().numpy()}'
        )
        # print()


if __name__ == '__main__':
    main()
